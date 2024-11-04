from unittest.mock import MagicMock, Mock, call, patch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class IDMSimulator:

    def __init__(self, scenario, planned_trajectory_samples=8) -> None:
        self.scenario = scenario
        self.planned_trajectory_samples = planned_trajectory_samples
        self.planner = None
        
    def setUp(self,
              target_velocity,
              min_gap_to_lead_agent,
              headway_time,
              accel_max,
              decel_max) -> None:

        self.planner = IDMPlanner(
            target_velocity=target_velocity,
            min_gap_to_lead_agent=min_gap_to_lead_agent,
            headway_time=headway_time,
            accel_max=accel_max,
            decel_max=decel_max,
            planned_trajectory_samples=self.planned_trajectory_samples * 10,
            planned_trajectory_sample_interval=0.1,
            occupancy_map_radius=20,
        )


    def compute_trajectory(self) -> None:
        """Test the IDMPlanner in full using mock data"""
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(10, self.scenario, DetectionsTracks)

        self.planner.initialize(
            PlannerInitialization(
                self.scenario.get_route_roadblock_ids(),
                self.scenario.get_mission_goal(),
                self.scenario.map_api,
            )
        )
        trajectories = self.planner.compute_trajectory(
            PlannerInput(
                SimulationIteration(self.scenario.get_time_point(0), 0),
                history_buffer,
                list(self.scenario.get_traffic_light_status_at_iteration(0)),
            )
        )

        return trajectories
    

class BayesianOptimizer:
    def __init__(self, scenario, planned_trajectory_samples=8) -> None:
        self.idm_simulator = IDMSimulator(scenario, planned_trajectory_samples)
        self.scenario = scenario
        self.planned_trajectory_samples = planned_trajectory_samples
        self.idm_params = {"target_velocity_range": [2.0, 30.0], 
                           "min_gap_to_lead_agent_range": [1.0, 5.0],
                           "headway_time_range": [1.0, 5.0],
                           "accel_max_range": [1.0, 2.0],
                           "decel_max_range": [1.0, 5.0]}
        
    def objective_function(self, param, debug=False):
        target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max = param
        self.idm_simulator.setUp(target_velocity=target_velocity, 
                                 min_gap_to_lead_agent=min_gap_to_lead_agent, 
                                 headway_time=headway_time, 
                                 accel_max=accel_max, 
                                 decel_max=decel_max)
        idm_trajectory: InterpolatedTrajectory = self.idm_simulator.compute_trajectory()
        idm_trajectory: list[EgoState] = idm_trajectory.get_sampled_trajectory()
        gt_trajectory_absolute = self.scenario.get_ego_future_trajectory(
            iteration=0, 
            num_samples=self.planned_trajectory_samples*10, 
            time_horizon=self.planned_trajectory_samples
        )
        gt_trajectory_absolute: list[EgoState] = [state for state in gt_trajectory_absolute]

        if debug:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot([state.center.x for state in idm_trajectory], 
                     [state.center.y for state in idm_trajectory], 
                     label="IDM")
            state_list = [state.center for state in gt_trajectory_absolute]
            plt.plot([idm_trajectory[0].center.x] + [state.x for state in state_list],
                     [idm_trajectory[0].center.y] + [state.y for state in state_list],
                     label="GT")
            plt.legend()
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"idm_vs_gt_{timestamp}.png", dpi=600)

        error = 0
        for idm_state, gt_state in zip(idm_trajectory[1:], gt_trajectory_absolute):
            error += idm_state.center.distance_to(gt_state.center)

        return error / len(idm_trajectory)
    
    def optimize(self):
        from skopt import gp_minimize
        result = gp_minimize(self.objective_function, 
                             [(self.idm_params["target_velocity_range"][0], self.idm_params["target_velocity_range"][1]),
                              (self.idm_params["min_gap_to_lead_agent_range"][0], self.idm_params["min_gap_to_lead_agent_range"][1]),
                              (self.idm_params["headway_time_range"][0], self.idm_params["headway_time_range"][1]),
                              (self.idm_params["accel_max_range"][0], self.idm_params["accel_max_range"][1]),
                              (self.idm_params["decel_max_range"][0], self.idm_params["decel_max_range"][1])],
                             n_calls=50,
                             n_random_starts=50,
                             random_state=1234)
        error = self.objective_function(result.x, debug=True)
        return result, error

        