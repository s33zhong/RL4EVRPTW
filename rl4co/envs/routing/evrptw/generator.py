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
            max_loc: float = 1.0,
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
            min_demand: float = 0.05,
            max_demand: float = 0.20,
            demand_distribution: Union[
                int, float, type, Callable
            ] = Uniform,
            vehicle_capacity: float = 1.0,
            vehicle_limit: int = 10,
            vehicle_speed: float = 2,
            max_fuel: float = 1,
            fuel_consumption_rate: float = 0.25,
            inverse_recharge_rate: float = 0.25,
            max_time: float = 1,
            horizon: float = 1,
            scale: bool = False,
            **kwargs,
    ):
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            depot_distribution=depot_distribution,
            demand_distribution=demand_distribution,
            vehicle_capacity=vehicle_capacity,
            **kwargs,
        )
        self.max_loc = max_loc
        self.vehicle_limit = vehicle_limit
        self.vehicle_speed = vehicle_speed
        self.min_time = 0.0
        self.min_demand = min_demand
        self.max_demand = max_demand
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

        # Sample demands and scale to [min_demand, max_demand]
        demand = (self.min_demand + (self.max_demand - self.min_demand) *
                  torch.rand(*batch_size, self.num_loc))

        # Create capacities
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

        lower_bound = dist/self.vehicle_speed
        upper_bound = torch.max(self.horizon - dist/self.vehicle_speed - durations, lower_bound)

        # 3. create two random values between 0 and 1 for time windows, and scale them to their upper bound
        tw_1 = torch.rand(*batch_size, self.num_loc + self.num_station + 1) * upper_bound
        tw_2 = torch.rand(*batch_size, self.num_loc + self.num_station + 1) * upper_bound

        # 4. set the lower value to min, the higher to max
        max_times = torch.clamp(torch.max(tw_1+0.05, tw_2+0.05), max=upper_bound, min=lower_bound)
        min_times = torch.clamp(torch.min(tw_1-0.05, tw_2-0.05), max=max_times, min=torch.zeros_like(max_times))

        # 5. set times for depot and stations; make them always available
        min_times[..., :, 0] = 0.0
        min_times[..., :, -self.num_station:] = 0.0
        max_times[..., :, 0] = self.horizon
        max_times[..., :, -self.num_station:] = self.horizon

        # 6. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)

        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        assert torch.all(
            0 <= min_times
        ), "Please make sure the time windows are non-negative."

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
                "demand": demand,
                "capacity": capacity,
                "durations": durations,
                "time_windows": time_windows,
            },
            batch_size=batch_size,
        )

