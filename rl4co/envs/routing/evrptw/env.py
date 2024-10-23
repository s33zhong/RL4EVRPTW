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
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = EVRPTWGenerator(**generator_params)
        self.generator = generator
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

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """In addition to the constraints considered in the CVRPEnv, the time windows are considered.
        The vehicle can only visit a location if it can reach it in time, i.e. before its time window ends.
        Also, the vehicle can only visit a location if it has enough fuel to reach it.
        """
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"]

        current_loc = gather_by_index(td["locs"], td["current_node"])
        dist = get_distance(current_loc[..., None, :], td["locs"])
        exceeds_time = td["current_time"] + dist > td["time_windows"][..., 1]
        exceeds_fuel = td["current_fuel"] - dist < 0
        # Nodes that cannot be visited are already visited or too much demand to be served now
        station_index = td["locs"].shape[-2] - td["stations"].shape[-2]
        num_station = td["stations"].shape[-2]
        mask_loc = (td["visited"].to(exceeds_cap.dtype)
                    | exceeds_cap
                    | exceeds_time[..., 1:station_index]
                    | exceeds_fuel[..., 1:station_index])
        unserved_reachable = ((mask_loc == 0).int().sum(-1) > 0)[:, None]
        # Cannot visit the depot if there are unserved nodes reachable and the vehicle limit is reached
        mask_depot = (unserved_reachable
                      | exceeds_fuel[..., None, 0]
                      | exceeds_time[..., None, 0]).to(mask_loc.dtype)
        # Cannot visit the station if just visited a station and still unserved nodes reachable
        mask_station = ((td["current_node"] >= station_index & unserved_reachable)
                        | exceeds_fuel[..., station_index:]
                        | exceeds_time[..., station_index:])
        mask_station = mask_station.expand(-1, num_station).to(mask_loc.dtype)  # Expand to match the number of stations

        not_masked = ~torch.cat((mask_depot, mask_loc, mask_station), -1)
        # Mask the current node (whether depot, station, or regular location)
        not_masked = not_masked.scatter(1, td["current_node"], 0)
        td.update({"current_loc": current_loc, "distances": dist})

        # print("exceeds_time: ", exceeds_time)
        # print("exceeds_fuel: ", exceeds_fuel)
        # print("exceeds_cap: ", exceeds_cap)
        # print("not_masked: ", not_masked)
        # print("mask_depot: ", mask_depot, "mask_loc: ", mask_loc, "mask_station: ", mask_station)
        return not_masked

    def _step(self, td: TensorDict) -> TensorDict:
        """In addition to the calculations in the CVRPEnv, the current time is
        updated to keep track of which nodes are still reachable in time.
        The current_node is updated in the parent class' _step() function.
        """
        batch_size = td["locs"].shape[0]
        # print(td["action"])
        # Check if the action is visiting a station
        station_index = td["locs"].shape[-2] - td["stations"].shape[-2]
        # print("Station index: ", station_index)
        is_station = td["action"][:, None] >= station_index
        is_depot = td["action"][:, None] == 0

        charging_percentage = 1 - td["current_fuel"]/self.generator.max_fuel

        # update current_time
        distance = gather_by_index(td["distances"], td["action"]).reshape([batch_size, 1])
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        duration = torch.where(is_station,
                               charging_percentage*duration, # Recharge time at station
                               duration)

        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        current_time = torch.where(is_depot,
                                   self.generator.min_time,  # Reset time to 0 at depot
                                   torch.max(td["current_time"] + distance, start_times) + duration)

        # If visiting a station, replenish the fuel to max fuel, otherwise deduct fuel based on distance
        current_fuel = torch.where((is_station | is_depot),
                                   self.generator.max_fuel,  # Full fuel replenishment at station
                                   td["current_fuel"] - distance)  # Deduct fuel based on distance traveled)

        # If visiting a depot, deduct one from current vehicle limit
        current_limit = torch.where(is_depot,
                                    td["current_limit"] - 1,
                                    td["current_limit"])

        # current_node is updated to the selected action
        current_node = td["action"][:, None]  # Add dimension for step
        # print("Current node in step(): ", current_node)
        n_loc = td["demand"].size(-1)  # Excludes depot

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )

        # Increase used capacity if depot is not visited, otherwise set to 0
        used_capacity = torch.where(is_depot,
                                    0.0,
                                    (td["used_capacity"] + selected_demand)).float()

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"]

        # Create a mask tensor where it is 1 if current_node is within the range of visited, 0 otherwise
        valid_mask = (current_node > 0) & (current_node <= visited.size(-1))

        # Create a tensor to apply updates, but only update where valid_mask is 1
        update_tensor = torch.zeros_like(visited)
        update_tensor.scatter_(-1, current_node[valid_mask].unsqueeze(-1)-1, 1)

        # Perform element-wise max to get the resulting visited tensor
        visited = torch.max(visited, update_tensor)
        finished = visited.sum(-1) == visited.size(-1)  # visited all customers
        # SECTION: get done
        returned = torch.where(is_depot & finished,
                               1,
                               0)
        done = finished & returned
        # print("Done: ", done)
        reward = torch.zeros_like(done) - 1

        td.update(
            {
                "current_node": current_node,
                "current_fuel": current_fuel,
                "current_time": current_time,
                "current_limit": current_limit,
                "used_capacity": used_capacity,
                "visited": visited,
                "finished": finished,
                "returned": returned,
                "reward": reward,
                "done": done,

            }
        )
        td.set("action_mask", self.get_action_mask(td))
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
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "current_limit": torch.full(
                    (*batch_size, 1), self.generator.vehicle_limit, device=device
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

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """The reward is the negative tour length. Time windows
        are not considered for the calculation of the reward."""
        return super()._get_reward(td, actions)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        print("Actions are: ", actions)
        print("To be implemented.")
        # CVRPEnv.check_solution_validity(td, actions)
        # batch_size = td["locs"].shape[0]
        # # distances to depot
        # distances = get_distance(
        #     td["locs"][..., 0, :], td["locs"].transpose(0, 1)
        # ).transpose(0, 1)
        # # basic checks on time windows
        # assert torch.all(distances >= 0.0), "Distances must be non-negative."
        # assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        # assert torch.all(
        #     td["time_windows"][..., :, 0] + distances + td["durations"]
        #     <= td["time_windows"][..., 0, 1][0]  # max_time is the same for all batches
        # ), "vehicle cannot perform service and get back to depot in time."
        # assert torch.all(
        #     td["durations"] >= 0.0
        # ), "Service durations must be non-negative."
        # assert torch.all(
        #     td["time_windows"][..., 0] < td["time_windows"][..., 1]
        # ), "there are unfeasible time windows"
        # assert torch.all(
        #     td["current_fuel"] >= 0
        # ), "There are unreachable locations due to fuel limit."
        # # check vehicles can meet deadlines
        # curr_time = torch.zeros(batch_size, 1, dtype=torch.float32, device=td.device)
        # curr_node = torch.zeros_like(curr_time, dtype=torch.int64, device=td.device)
        # for ii in range(actions.size(1)):
        #     next_node = actions[:, ii]
        #     dist = get_distance(
        #         gather_by_index(td["locs"], curr_node).reshape([batch_size, 2]),
        #         gather_by_index(td["locs"], next_node).reshape([batch_size, 2]),
        #     ).reshape([batch_size, 1])
        #     curr_time = torch.max(
        #         (curr_time + dist).int(),
        #         gather_by_index(td["time_windows"], next_node)[..., 0].reshape(
        #             [batch_size, 1]
        #         ),
        #     )
        #     assert torch.all(
        #         curr_time
        #         <= gather_by_index(td["time_windows"], next_node)[..., 1].reshape(
        #             [batch_size, 1]
        #         )
        #     ), "vehicle cannot start service before deadline"
        #     curr_time = curr_time + gather_by_index(td["durations"], next_node).reshape(
        #         [batch_size, 1]
        #     )
        #     curr_node = next_node
        #     curr_time[curr_node == 0] = 0.0  # reset time for depot

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
