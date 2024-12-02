import logging
import math
from typing import List, Tuple

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring
from nuplan.planning.simulation.planner.abstract_idm_planner import AbstractIDMPlanner
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy

from IDM_simulator import BayesianOptimizer, OptunaOptimizer

class NewIDMPlanner(IDMPlanner):
    def __init__(self, target_velocity: float, min_gap_to_lead_agent: float, headway_time: float, accel_max: float, decel_max: float, planned_trajectory_samples: int, planned_trajectory_sample_interval: float, occupancy_map_radius: float, scenario=None):
        super(NewIDMPlanner, self).__init__(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max, planned_trajectory_samples, planned_trajectory_sample_interval, occupancy_map_radius)
        self._initialized = False
        self._scenario = scenario
        self._optimizer = OptunaOptimizer(scenario, planned_trajectory_samples)
        self._opt_params = None
        self._token_recorded = False

    def set_parameters(self, target_velocity: float, min_gap_to_lead_agent: float, headway_time: float, accel_max: float, decel_max: float) -> None:
        self._policy = IDMPolicy(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max)

    def transform_to_absolute_parameters(self, param, current_input: PlannerInput):
        target_velocity = min(max(1.0, param[0] + current_input.history.current_state[0].dynamic_car_state.speed), 30)
        res_param = [target_velocity, param[1], param[2], param[3], param[4]]
        return res_param
            
    def get_parameters(self, current_input: PlannerInput):
        if self._opt_params is not None:
            return self.transform_to_absolute_parameters(self._opt_params, current_input) 
        param, error = self._optimizer.optimize(current_input)
        # print(f'Optimized parameters: {param.x}, error: {error}')
        if error <= 1.0:
            with open('optimized_parameters_2.json', 'a') as f:
                f.write(f'{param}\n')
            f.close()
        else:
           self._opt_params = param
        return self.transform_to_absolute_parameters(param, current_input)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        ego_state, observations = current_input.history.current_state
        self.set_parameters(*self.get_parameters(current_input))

        if not self._initialized:
            self._initialize_ego_path(ego_state)
            self._initialized = True

        occupancy_map, unique_observations = self._construct_occupancy_map(ego_state, observations)

        traffic_light_data = current_input.traffic_light_data
        self._annotate_occupancy_map(traffic_light_data, occupancy_map)

        return self._get_planned_trajectory(ego_state, occupancy_map, unique_observations)
