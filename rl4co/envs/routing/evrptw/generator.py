from typing import Union, Callable

import torch

from torch.distributions import Uniform
from tensordict.tensordict import TensorDict
from rl4co.envs.common.utils import get_sampler

from rl4co.envs.routing.cvrp.generator import CVRPGenerator
from rl4co.utils.ops import get_distance


class EVRPTWGenerator(CVRPGenerator):
    """Data generator for the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) environment
    Generates time windows and service durations for the locations. The depot has a time window of [0, self.max_time].
    The time windows define the time span within which a service has to be started. To reach the depot in time from the last node,
    the end time of each node is bounded by the service duration and the distance back to the depot.
    The start times of the time windows are bounded by how long it takes to travel there from the depot.

    Args:
        num_loc: number of locations (customers) in the VRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates, default is 150 insted of 1.0, will be scaled
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations
        min_demand: minimum value for the demand of each customer
        max_demand: maximum value for the demand of each customer
        demand_distribution: distribution for the demand of each customer
        capacity: capacity of the vehicle
        max_time: maximum time for the vehicle to complete the tour
        scale: if True, the locations, time windows, and service durations will be scaled to [0, 1]. Default to False

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each city
            depot [batch_size, 2]: location of the depot
            demand [batch_size, num_loc]: demand of each customer
                while the demand of the depot is a placeholder
            capacity [batch_size, 1]: capacity of the vehicle
            durations [batch_size, num_loc]: service durations of each location
            time_windows [batch_size, num_loc, 2]: time windows of each location
    """

    def __init__(
            self,
            num_loc: int = 20,
            min_loc: float = 0.0,
            max_loc: float = 150.0,
            loc_distribution: Union[
                int, float, str, type, Callable
            ] = Uniform,
            depot_distribution: Union[
                int, float, str, type, Callable
            ] = None,
            num_station: int = 5,
            station_distribution: Union[
                int, float, str, type, Callable
            ] = Uniform,
            min_demand: int = 1,
            max_demand: int = 10,
            demand_distribution: Union[
                int, float, type, Callable
            ] = Uniform,
            vehicle_capacity: float = 1.0,
            vehicle_limit: int = 10,
            max_fuel: float = 500,
            fuel_consumption_rate: float = 1.0,
            inverse_recharge_rate: float = 1.0,
            capacity: float = None,
            max_time: float = 480,
            horizon: float = 480,
            scale: bool = False,
            **kwargs,
    ):
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            depot_distribution=depot_distribution,
            min_demand=min_demand,
            max_demand=max_demand,
            demand_distribution=demand_distribution,
            vehicle_capacity=vehicle_capacity,
            capacity=capacity,
            **kwargs,
        )
        self.max_loc = max_loc
        self.vehicle_limit = vehicle_limit
        self.min_time = 0.0
        self.num_station = num_station
        self.max_time = max_time
        self.horizon = horizon
        self.scale = scale
        self.max_fuel = max_fuel/fuel_consumption_rate
        self.recharge_time = max_fuel * inverse_recharge_rate
        # Depot distribution
        if kwargs.get("station_distribution", None) is not None:
            self.station_sampler = kwargs["station_distribution"]
        else:
            self.station_sampler = get_sampler(
                "station", station_distribution, min_loc, max_loc, **kwargs
            ) if station_distribution is not None else None

    def _generate(self, batch_size) -> TensorDict:

        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        if self.depot_sampler is None:
            loc = self.loc_sampler.sample((*batch_size, self.num_loc+1, 2))
            depot = loc[..., 0, :]
            customer = loc[..., 1:, :]
        else:
            depot = self.depot_sampler.sample((*batch_size, 2))
            customer = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        station = self.station_sampler.sample((*batch_size, self.num_station, 2)) \
            if self.station_sampler is not None and self.num_station != 0 else None

        # Sample demands
        demand = self.demand_sampler.sample((*batch_size, self.num_loc))
        demand = (demand.int() + 1).float()

        # Sample capacities
        capacity = torch.full((*batch_size, 1), self.capacity)

        ## define service durations
        # Assume service duration is instantaneous
        durations = torch.zeros(
            *batch_size, self.num_loc + 1 + self.num_station, dtype=torch.float32
        )
        # Add recharge duration
        durations[..., -self.num_station:] = self.recharge_time

        ## define time windows
        # 1. get distances from depot
        customer_dist = get_distance(depot, customer.transpose(0, 1)).transpose(0, 1)
        station_dist = get_distance(depot, station.transpose(0, 1)).transpose(0, 1)
        dist = torch.cat((torch.zeros(*batch_size, 1), customer_dist, station_dist), dim=1)  # 0 being the depot,
        # 1 ~ num_loc being the customers, num_loc+1 ~ num_loc+1+num_station being the customers

        # 2. define upper bound for time windows to make sure the vehicle can get back to the depot in time
        upper_bound = self.max_time - dist - durations

        # 3. create random values between 0 and 1
        ts_1 = torch.rand(*batch_size, self.num_loc + self.num_station + 1)
        ts_2 = torch.rand(*batch_size, self.num_loc + self.num_station + 1)

        # 4. scale values to lie between their respective min_time and max_time and convert to integer values
        min_ts = (dist + (upper_bound - dist) * ts_1).int()
        max_ts = (dist + (upper_bound - dist) * ts_2).int()

        # 5. set the lower value to min, the higher to max
        min_times = torch.min(min_ts, max_ts)
        max_times = torch.max(min_ts, max_ts)

        # 6. reset times for depot and stations
        min_times[..., :, 0] = 0.0
        min_times[..., :, -self.num_station:] = 0.0
        max_times[..., :, 0] = self.horizon
        max_times[..., :, -self.num_station:] = self.horizon

        # 7. ensure min_times < max_times to prevent numerical errors in attention.py
        # min_times == max_times may lead to nan values in _inner_mha()
        mask = min_times == max_times
        if torch.any(mask):
            min_tmp = min_times.clone()
            min_tmp[mask] = torch.max(
                dist[mask].int(), min_tmp[mask] - 1
            )  # we are handling integer values, so we can simply substract 1
            min_times = min_tmp

            mask = min_times == max_times  # update mask to new min_times
            if torch.any(mask):
                max_tmp = max_times.clone()
                max_tmp[mask] = torch.min(
                    torch.floor(upper_bound[mask]).int(),
                    torch.max(
                        torch.ceil(min_tmp[mask] + durations[mask]).int(),
                        max_tmp[mask] + 1,
                    ),
                )
                max_times = max_tmp

        # Scale to [0, 1]
        if self.scale:
            durations = durations / self.horizon
            min_times = min_times / self.horizon
            max_times = max_times / self.horizon
            depot = depot / self.horizon
            customer = customer / self.horizon
            station = station / self.horizon

        # 8. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)

        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        # Reset duration at depot to 0
        durations[:, 0] = 0.0

        # Concatenate depot, customers, and stations
        if self.station_sampler is None:
            locs = torch.cat((depot.unsqueeze(1), customer), dim=1)
        else:
            locs = torch.cat((depot.unsqueeze(1), customer, station), dim=1)
        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "stations": station,
                "demand": demand / self.capacity,
                "capacity": capacity,
                "durations": durations,
                "time_windows": time_windows,
            },
            batch_size=batch_size,
        )

