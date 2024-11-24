from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from rl4co.data.utils import (
    load_npz_to_tensordict,
    load_solomon_instance,
    load_solomon_solution,
)
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.utils.ops import gather_by_index, get_distance

from ..cvrp.generator import CVRPGenerator
from .generator import EVRPTWGenerator
from .render import render


class EVRPTWEnv(CVRPEnv):
    """Electric Vehicle Routing Problem with Time Windows (EVRPTW) environment.
    Inherits from the CVRPEnv class in which customers are considered.
    Additionally considers time windows within which a service has to be started,
    and the vehicle has fuel limit.

    Observations:
        - location of the depot.
        - locations and demand of each customer.
        - current location of the vehicle.
        - the remaining customer of the vehicle.
        - the current time.
        - the remaining fuel of the vehicle.
        - service durations of each location.
        - time windows of each location.

    Constraints:
        - the tour starts and ends at the depot.
        - each customer must be visited exactly once.
        - the vehicle cannot visit customers exceed the remaining customer.
        - the vehicle can return to the depot to refill the customer.
        - the vehicle must start the service within the time window of each location.

    Finish Condition:
        - the vehicle has visited all customers and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPTWGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "evrptw"

    def __init__(
        self,
        generator: EVRPTWGenerator = None,
        generator_params: dict = {},
        soft: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = EVRPTWGenerator(**generator_params)
        self.generator = generator
        self.soft = soft
        self._make_spec(self.generator)

    def _make_spec(self, generator: EVRPTWGenerator):
        if isinstance(generator, CVRPGenerator):
            super()._make_spec(generator)
        else:
            current_time = UnboundedContinuousTensorSpec(shape=(1), dtype=torch.float32, device=self.device)
            current_loc = UnboundedContinuousTensorSpec(shape=(2), dtype=torch.float32, device=self.device)
            current_limit = BoundedTensorSpec(
                low=0,
                high=generator.vehicle_limit,
                shape=(1,),
                dtype=torch.int64,
                device=self.device,
            )
            current_fuel = BoundedTensorSpec(
                low=0.0,
                high=generator.max_fuel,
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            )
            durations = BoundedTensorSpec(
                low=generator.min_time,
                high=generator.max_time,
                shape=(generator.num_loc, 1),
                dtype=torch.int64,
                device=self.device,
            )
            time_windows = BoundedTensorSpec(
                low=generator.min_time,
                high=generator.max_time,
                shape=(
                    generator.num_loc,
                    2,
                ),  # Each location has a 2D time window (start, end)
                dtype=torch.int64,
                device=self.device,
            )
            self.action_spec = BoundedTensorSpec(
                shape=(1,),
                dtype=torch.int64,
                low=0,
                high=generator.num_loc + generator.num_station + 1,
            )
            # Extend observation specs
            self.observation_spec = CompositeSpec(
                **self.observation_spec,
                current_time=current_time,
                current_loc=current_loc,
                current_fuel=current_fuel,
                current_limit=current_limit,
                durations=durations,
                time_windows=time_windows,
            )

    def get_action_mask(self, td: TensorDict, soft=False) -> torch.Tensor:
        """In addition to the constraints considered in the CVRPEnv, the time windows are considered.
        The vehicle can only visit a location if it can reach it in time, i.e. before its time window ends.
        Also, the vehicle can only visit a location if it has enough fuel to reach it.
        """
        if not soft:
            # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
            exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"]

            current_loc = gather_by_index(td["locs"], td["current_node"])
            # Calculate the distance to each location
            dist_to_loc = get_distance(current_loc[..., None, :], td["locs"])
            dist_to_depot = get_distance(td["locs"][..., 0, :].unsqueeze(1), td["locs"])
            # Calculate the time to each location
            time_to_loc = dist_to_loc / td["vehicle_speed"]

            exceeds_time = (td["current_time"] + time_to_loc > td["time_windows"][..., 1])
            exceeds_fuel_station = td["current_fuel"] - dist_to_loc < 0
            exceeds_fuel = td["current_fuel"] - dist_to_loc - dist_to_depot < 0
            # Nodes that cannot be visited are already visited or too much demand to be served now
            station_index = td["locs"].shape[-2] - td["stations"].shape[-2]
            num_station = td["stations"].shape[-2]

            mask_loc = (td["visited"].to(exceeds_cap.dtype)
                        | exceeds_cap
                        | exceeds_time[..., 1:station_index]
                        | exceeds_fuel[..., 1:station_index])
            
            unserved_reachable = ((mask_loc == 0).int().sum(-1) > 0)[:, None]
            # Cannot visit the depot if there are unserved nodes reachable and the last node is the depot
            mask_depot = (unserved_reachable
                        | exceeds_fuel[..., None, 0]
                        | exceeds_time[..., None, 0]).to(mask_loc.dtype)
            
            # Cannot visit a station if just visited a station
            max_fuel = torch.full_like(td["current_fuel"], self.generator.max_fuel)
            is_max_fuel = td["current_fuel"] == max_fuel
            mask_station = (is_max_fuel
                            | exceeds_fuel_station[..., station_index:]
                            | exceeds_time[..., station_index:])
            mask_station = mask_station.expand(-1, num_station).to(mask_loc.dtype)  # Expand to match the number of stations

            not_masked = ~torch.cat((mask_depot, mask_loc, mask_station), -1)

            td.update({"current_loc": current_loc, "distances": dist_to_loc})
            # print(f"\nvisitable: {not_masked}, visited: {td['visited']}, current node: {td['current_node']}")
            # print(f"\nCurrent time: {td['current_time']}, current fuel: {td['current_fuel']}")
            # print(f"\nExceeds cap: {exceeds_cap}, exceeds time: {exceeds_time[..., None, 0]}, exceeds fuel: {exceeds_fuel[..., None, 0]}")
            # print(f"\nUnserved reachable: {unserved_reachable}, currently at depot: {td['current_node']==0}, mask depot: {mask_depot}")
            return not_masked
        else:
            current_loc = gather_by_index(td["locs"], td["current_node"])
            dist_to_loc = get_distance(current_loc[..., None, :], td["locs"])
            mask_loc = td["visited"].bool()

            mask_depot = torch.zeros((mask_loc.shape[0], 1), dtype=mask_loc.dtype)
            default_mask_station = torch.zeros((mask_loc.shape[0], td["stations"].shape[-2]), dtype=mask_loc.dtype)
            # Cannot visit a station if just visited a station
            max_fuel = torch.full_like(td["current_fuel"], self.generator.max_fuel)
            is_max_fuel = td["current_fuel"] == max_fuel
            mask_station = (is_max_fuel.cpu() | default_mask_station)
                            
            not_masked = ~torch.cat((mask_depot, mask_loc.cpu(), mask_station), -1)
            td.update({"current_loc": current_loc, "distances": dist_to_loc})
            return not_masked

    
    def _step(self, td: TensorDict) -> TensorDict:
        """In addition to the calculations in the CVRPEnv, the current time is
        updated to keep track of which nodes are still reachable in time.
        The current_node is updated in the parent class' _step() function.
        """
        batch_size = td["locs"].shape[0]
        # Check if the action is visiting a station
        station_index = td["locs"].shape[-2] - td["stations"].shape[-2]
        is_station = td["action"][:, None] >= station_index
        is_depot = td["action"][:, None] == 0

        # update current_time
        distance = gather_by_index(td["distances"], td["action"]).reshape([batch_size, 1])
        charging_percentage = 1 - (td["current_fuel"] - distance)/self.generator.max_fuel
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        duration[is_station] *= charging_percentage[is_station]  # Recharge time at station

        time_to_loc = distance / td["vehicle_speed"]  # Time to reach the location

        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape([batch_size, 1])
        end_times = gather_by_index(td["time_windows"], td["action"])[..., 1].reshape([batch_size, 1])
        current_time = torch.where(is_depot,
                                   self.generator.min_time,  # Reset time to 0 at depot
                                   torch.max(td["current_time"] + time_to_loc, start_times) + duration)
        
        # Calculate time penalty: if current_time exceeds end_time, compute the violation amount
        penalty_time = torch.max(torch.zeros_like(current_time), current_time - end_times).sum(dim=1, keepdim=True)

        # If visiting a station, replenish the fuel to max fuel, otherwise deduct fuel based on distance
        current_fuel = torch.where((is_station | is_depot),
                                   self.generator.max_fuel,  # Full fuel replenishment at station
                                   td["current_fuel"] - distance)  # Deduct fuel based on distance traveled)
        
        # Calculate battery penalty: if current_fuel goes below zero, compute the violation amount
        penalty_battery = torch.max(torch.zeros_like(current_fuel), -current_fuel).sum(dim=1, keepdim=True)

        # current_node is updated to the selected action
        current_node = td["action"][:, None]  # Add dimension for step
        # print("Current node in step(): ", current_node)
        n_loc = td["demand"].size(-1)  # Excludes depot

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = ~is_station*gather_by_index(
            td["demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )

        # Increase used capacity if depot is not visited, otherwise set to 0
        used_capacity = torch.where(is_depot,
                                    0.0,
                                    (td["used_capacity"] + selected_demand)).float()

        # Calculate cargo penalty: if used_capacity exceeds vehicle capacity, compute the violation amount
        penalty_cargo = torch.max(torch.zeros_like(used_capacity), used_capacity - td["vehicle_capacity"]).sum(dim=1, keepdim=True)

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"]

        # Create a mask tensor where it is 1 if current_node is within the range of visited, 0 otherwise
        valid_mask = (current_node > 0) & (current_node <= visited.size(-1))

        # Set update_tensor values to 1 at the specified positions
        update_tensor = torch.zeros_like(visited)

        # update_tensor.scatter_(-1, current_node[valid_mask].unsqueeze(-1)-1, 1)
        rows = torch.arange(valid_mask.size(0))[valid_mask.squeeze(-1).cpu()]
        cols = (current_node[valid_mask] - 1).squeeze(-1)
        update_tensor[rows, cols] = 1

        # Perform element-wise max to get the resulting visited tensor
        visited = torch.max(visited, update_tensor)

        # Considered finished, if visited all customers
        finished = visited.sum(-1) == visited.size(-1)

        # SECTION: get done
        done = finished & is_depot.squeeze()
        reward = torch.zeros_like(done)

        # If visiting a depot, deduct one from current vehicle limit if it is not done
        current_limit = td["current_limit"]

        # Setting a hard limit (minus 10) to vehicles; if limit is reached, the instance has failed and thus done
        done = done | (current_limit <= -10).squeeze()

        # If depot is visited, deduct one from the limit if not done yet; we allow the vehicle to remain in depot
        #    if it has finished all its tasks, or it has already failed
        current_limit[is_depot & ~done.unsqueeze(1)] -= 1

        td.update(
            {
                "current_node": current_node,
                "current_fuel": current_fuel,
                "current_time": current_time,
                "current_limit": current_limit,
                "used_capacity": used_capacity,
                "visited": visited,
                "finished": finished,
                "reward": reward,
                "done": done,
                "penalty_time": penalty_time,
                "penalty_battery": penalty_battery,
                "penalty_cargo": penalty_cargo,
            }
        )
        # td.set("action_mask", self.get_action_mask(td))
        td.set("action_mask", self.get_action_mask(td, self.soft))
        return td
    

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device
        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "depot": td["depot"],
                "stations": td["stations"],
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "vehicle_speed": torch.full(
                    (*batch_size, 1), self.generator.vehicle_speed, device=device
                ),
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "current_limit": torch.full(
                    (*batch_size, 1), self.generator.vehicle_limit - 1, device=device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.generator.vehicle_capacity, device=device
                ),
                "current_fuel": torch.full(
                    (*batch_size, 1), self.generator.max_fuel, device=device
                ),
                "max_fuel": torch.full(
                    (*batch_size, 1), self.generator.max_fuel, device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2] - td["stations"].shape[-2] - 1),
                    dtype=torch.uint8,
                    device=device,
                ),  # Only record the customers; visits to the station/depot can repeat
                "finished": torch.zeros(
                    *batch_size, 1, dtype=torch.uint8, device=device
                ),  # Whether the vehicle has met the demand of all customers
                "durations": td["durations"],
                "time_windows": td["time_windows"],
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
    
    def _get_reward(self, td: TensorDict, actions: torch.Tensor, train: bool = False, feasibility_check = False,
                    beta_1: int = 1, beta_2: int = 10000) -> torch.Tensor:
        """The reward is the negative tour length minus the cost
        of invoking additional EVs (beyond limit)."""
        # Energy consumption which is proportional to the distance traveled
        negative_tour_length = super()._get_reward(td, actions)
        # Cost of invoking additional EVs beyond EV limit
        ev_cost = beta_1 * torch.clamp(td['current_limit'], max=0)

        # Cost of not visiting all customers
        unvisited = torch.sum(~td["visited"].bool(), dim=1)
        unvisited_cost = -beta_2 * unvisited

        # print(negative_tour_length.shape, ev_cost.shape, td['current_limit'].shape)
        if train:
            return negative_tour_length + ev_cost.squeeze() + unvisited_cost.squeeze()
        else:
            if feasibility_check:
                # 1. Vehicle limit
                limit_violate = (td['current_limit'] < 0)
                # 2. EVs limit violoation
                ev_violation_cost = -beta_2 * limit_violate
                # 3. Time, Bettery, Cargo violation (done in the _step() function)
                return negative_tour_length + ev_violation_cost.squeeze() + unvisited_cost.squeeze()
            else:
                return negative_tour_length

    def calculate_time_penalty(self, info):
        """Calculate penalty for time window violations."""
        current_time = info["current_time"]
        time_windows = info["time_windows"]

        # Calculate if the current time exceeds the allowed time window at the current location
        start_time, end_time = time_windows[..., 0], time_windows[..., 1]
        time_violations = torch.max(torch.zeros_like(current_time), current_time - end_time)
        penalty_time = time_violations.sum()
        
        return penalty_time

    def calculate_battery_penalty(self, info):
        """Calculate penalty for battery constraint violations."""
        current_fuel = info["current_fuel"]

        # If fuel is below 0, calculate the penalty as the absolute deficit
        fuel_deficit = torch.max(torch.zeros_like(current_fuel), -current_fuel)
        penalty_battery = fuel_deficit.sum()
        
        return penalty_battery

    def calculate_cargo_penalty(self, info):
        """Calculate penalty for cargo capacity violations."""
        used_capacity = info["used_capacity"]
        vehicle_capacity = info["vehicle_capacity"]

        # If capacity is exceeded, calculate the penalty as the excess load
        cargo_excess = torch.max(torch.zeros_like(used_capacity), used_capacity - vehicle_capacity)
        penalty_cargo = cargo_excess.sum()
        
        return penalty_cargo
    
    # @staticmethod
    # def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
    #     # print("Actions are: ", actions)
    #     # print("To be implemented.")
    #     # CVRPEnv.check_solution_validity(td, actions)
    #     # batch_size = td["locs"].shape[0]
    #     # # distances to depot
    #     # distances = get_distance(
    #     #     td["locs"][..., 0, :], td["locs"].transpose(0, 1)
    #     # ).transpose(0, 1)
    #     # # basic checks on time windows
    #     # assert torch.all(distances >= 0.0), "Distances must be non-negative."
    #     # assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
    #     # assert torch.all(
    #     #     td["time_windows"][..., :, 0] + distances + td["durations"]
    #     #     <= td["time_windows"][..., 0, 1][0]  # max_time is the same for all batches
    #     # ), "vehicle cannot perform service and get back to depot in time."
    #     # assert torch.all(
    #     #     td["durations"] >= 0.0
    #     # ), "Service durations must be non-negative."
    #     # assert torch.all(
    #     #     td["time_windows"][..., 0] < td["time_windows"][..., 1]
    #     # ), "there are unfeasible time windows"
    #     # assert torch.all(
    #     #     td["current_fuel"] >= 0
    #     # ), "There are unreachable locations due to fuel limit."
    #     # # check vehicles can meet deadlines
    #     # curr_time = torch.zeros(batch_size, 1, dtype=torch.float32, device=td.device)
    #     # curr_node = torch.zeros_like(curr_time, dtype=torch.int64, device=td.device)
    #     # for ii in range(actions.size(1)):
    #     #     next_node = actions[:, ii]
    #     #     dist = get_distance(
    #     #         gather_by_index(td["locs"], curr_node).reshape([batch_size, 2]),
    #     #         gather_by_index(td["locs"], next_node).reshape([batch_size, 2]),
    #     #     ).reshape([batch_size, 1])
    #     #     curr_time = torch.max(
    #     #         (curr_time + dist).int(),
    #     #         gather_by_index(td["time_windows"], next_node)[..., 0].reshape(
    #     #             [batch_size, 1]
    #     #         ),
    #     #     )
    #     #     assert torch.all(
    #     #         curr_time
    #     #         <= gather_by_index(td["time_windows"], next_node)[..., 1].reshape(
    #     #             [batch_size, 1]
    #     #         )
    #     #     ), "vehicle cannot start service before deadline"
    #     #     curr_time = curr_time + gather_by_index(td["durations"], next_node).reshape(
    #     #         [batch_size, 1]
    #     #     )
    #     #     curr_node = next_node
    #     #     curr_time[curr_node == 0] = 0.0  # reset time for depot
    #     return

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        render(td, actions, ax)

    # @staticmethod
    # def load_data(
    #     name: str,
    #     solomon=False,
    #     path_instances: str = None,
    #     type: str = None,
    #     compute_edge_weights: bool = False,
    # ):
    #     if solomon:
    #         assert type in [
    #             "instance",
    #             "solution",
    #         ], "type must be either 'instance' or 'solution'"
    #         if type == "instance":
    #             instance = load_solomon_instance(
    #                 name=name, path=path_instances, edge_weights=compute_edge_weights
    #             )
    #         elif type == "solution":
    #             instance = load_solomon_solution(name=name, path=path_instances)
    #         return instance
    #     return load_npz_to_tensordict(filename=name)
    #
    # def extract_from_solomon(self, instance: dict, batch_size: int = 1):
    #     # extract parameters for the environment from the Solomon instance
    #     self.min_demand = instance["demand"][1:].min()
    #     self.max_demand = instance["demand"][1:].max()
    #     self.vehicle_capacity = instance["capacity"]
    #     self.min_loc = instance["node_coord"][1:].min()
    #     self.max_loc = instance["node_coord"][1:].max()
    #     self.min_time = instance["time_window"][:, 0].min()
    #     self.max_time = instance["time_window"][:, 1].max()
    #     # assert the time window of the depot starts at 0 and ends at max_time
    #     assert self.min_time == 0, "Time window of depot must start at 0."
    #     assert (
    #         self.max_time == instance["time_window"][0, 1]
    #     ), "Depot must have latest end time."
    #     # convert to format used in CVRPTWEnv
    #     td = TensorDict(
    #         {
    #             "depot": torch.tensor(
    #                 instance["node_coord"][0],
    #                 dtype=torch.float32,
    #                 device=self.device,
    #             ).repeat(batch_size, 1),
    #             "locs": torch.tensor(
    #                 instance["node_coord"][1:],
    #                 dtype=torch.float32,
    #                 device=self.device,
    #             ).repeat(batch_size, 1, 1),
    #             "demand": torch.tensor(
    #                 instance["demand"][1:],
    #                 dtype=torch.float32,
    #                 device=self.device,
    #             ).repeat(batch_size, 1),
    #             "durations": torch.tensor(
    #                 instance["service_time"],
    #                 dtype=torch.int64,
    #                 device=self.device,
    #             ).repeat(batch_size, 1),
    #             "time_windows": torch.tensor(
    #                 instance["time_window"],
    #                 dtype=torch.int64,
    #                 device=self.device,
    #             ).repeat(batch_size, 1, 1),
    #         },
    #         batch_size=1,  # we assume batch_size will always be 1 for loaded instances
    #     )
    #     return self.reset(td, batch_size=batch_size)

#%%
