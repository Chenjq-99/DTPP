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

    def __init__(self, scenario, planned_trajectory_samples=40) -> None:
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
            planned_trajectory_samples=self.planned_trajectory_samples,
            planned_trajectory_sample_interval=0.1,
            occupancy_map_radius=20,
        )


    def compute_trajectory(self, current_input) -> None:
        self.planner.initialize(
            PlannerInitialization(
                self.scenario.get_route_roadblock_ids(),
                self.scenario.get_mission_goal(),
                self.scenario.map_api,
            )
        )
        trajectories = self.planner.compute_trajectory(current_input=current_input)
        return trajectories

class BayesianOptimizer:
    def __init__(self, scenario, planned_trajectory_samples=40) -> None:
        self.idm_simulator = IDMSimulator(scenario, planned_trajectory_samples)
        self.scenario = scenario
        self.planned_trajectory_samples = planned_trajectory_samples
        from skopt.space import Categorical
        self.idm_params = [
            Categorical([-10, -7, -5, -3, 0, 3, 5, 7, 10], name="target_velocity"),
            Categorical([0.5, 1.0, 1.5, 2.0], name="min_gap_to_lead_agent"),
            Categorical([1.0, 1.5, 2.0, 2.5], name="headway_time"),
            Categorical([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], name="accel_max"),
            Categorical([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], name="decel_max")
        ] 
        
    def objective_function(self, param, current_input:PlannerInput=None, debug=False) -> float:
        target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max = param
        self.idm_simulator.setUp(target_velocity=target_velocity + current_input.history.current_state[0].dynamic_car_state.speed,
                                 min_gap_to_lead_agent=min_gap_to_lead_agent, 
                                 headway_time=headway_time, 
                                 accel_max=accel_max, 
                                 decel_max=decel_max)
        idm_trajectory: InterpolatedTrajectory = self.idm_simulator.compute_trajectory(current_input)
        idm_trajectory: list[EgoState] = idm_trajectory.get_sampled_trajectory()
        gt_trajectory_absolute = self.scenario.get_ego_future_trajectory(
            iteration=current_input.iteration.index, 
            num_samples=self.planned_trajectory_samples, 
            time_horizon=self.planned_trajectory_samples * 0.1
        )
        gt_trajectory_absolute: list[EgoState] = [state for state in gt_trajectory_absolute]

        if debug:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot([state.center.x for state in idm_trajectory[1:]], 
                     [state.center.y for state in idm_trajectory[1:]], 
                     label="IDM")
            plt.plot([state.center.x for state in gt_trajectory_absolute],
                     [state.center.y for state in gt_trajectory_absolute],
                     label="GT")
            plt.legend()
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"idm_pic/xy_idm_vs_gt_{timestamp}.png", dpi=600)
            plt.close()

            plt.figure()
            plt.plot([t * 0.1 for t in range(1, len(idm_trajectory))],
                     [state.dynamic_car_state.speed for state in idm_trajectory[1:]], 
                     label="IDM")
            plt.plot([t * 0.1 for t in range(1, len(idm_trajectory))],
                     [state.dynamic_car_state.speed for state in gt_trajectory_absolute],
                     label="GT")
            plt.legend()
            plt.savefig(f"idm_pic/speed_idm_vs_gt_{timestamp}.png", dpi=600)
            plt.close()

            plt.figure()
            plt.plot([t * 0.1 for t in range(1, len(idm_trajectory))],
                     [state.dynamic_car_state.acceleration for state in idm_trajectory[1:]], 
                     label="IDM")
            plt.plot([t * 0.1 for t in range(1, len(idm_trajectory))],
                     [state.dynamic_car_state.acceleration for state in gt_trajectory_absolute],
                        label="GT")
            plt.legend()
            plt.savefig(f"idm_pic/acceleration_idm_vs_gt_{timestamp}.png", dpi=600)
            plt.close()

        error = 0
        for idm_state, gt_state in zip(idm_trajectory[1:], gt_trajectory_absolute):
            error += idm_state.center.distance_to(gt_state.center)

        return error / len(idm_trajectory)
    
    def optimize(self, current_input):
        from skopt import gp_minimize
        from functools import partial
        objective_function_with_iteration = partial(self.objective_function, current_input=current_input)
        result = gp_minimize(objective_function_with_iteration, 
                             self.idm_params,
                             n_calls=80,
                             n_random_starts=50,
                             random_state=666)
        error = self.objective_function(result.x, current_input=current_input, debug=False)
        return result, error


import optuna
class OptunaOptimizer:
    def __init__(self, scenario, planned_trajectory_samples=40) -> None:
        self.idm_simulator = IDMSimulator(scenario, planned_trajectory_samples)
        self.scenario = scenario
        self.planned_trajectory_samples = planned_trajectory_samples
        
    def objective_function(self, trail=None, current_input:PlannerInput=None, debug=False) -> float:
        target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max = \
        trail.suggest_categorical('target_velocity', [-10.0, -9.0, -8.0, -7.0,  -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), \
        trail.suggest_categorical('min_gap_to_lead_agent', [0.5, 1.0, 1.5, 2.0]), \
        trail.suggest_categorical('headway_time', [1.0, 1.5, 2.0, 2.5]), \
        trail.suggest_categorical('accel_max', [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]), \
        trail.suggest_categorical('decel_max', [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        target_velocity_ = min(max(1.0, target_velocity + current_input.history.current_state[0].dynamic_car_state.speed), 30)
        self.idm_simulator.setUp(target_velocity=target_velocity_,
                                 min_gap_to_lead_agent=min_gap_to_lead_agent, 
                                 headway_time=headway_time, 
                                 accel_max=accel_max, 
                                 decel_max=decel_max)
        idm_trajectory: InterpolatedTrajectory = self.idm_simulator.compute_trajectory(current_input)
        idm_trajectory: list[EgoState] = idm_trajectory.get_sampled_trajectory()
        gt_trajectory_absolute = self.scenario.get_ego_future_trajectory(
            iteration=current_input.iteration.index, 
            num_samples=self.planned_trajectory_samples, 
            time_horizon=self.planned_trajectory_samples * 0.1
        )
        gt_trajectory_absolute: list[EgoState] = [state for state in gt_trajectory_absolute]

        error = 0
        for idm_state, gt_state in zip(idm_trajectory[1:], gt_trajectory_absolute):
            error += idm_state.center.distance_to(gt_state.center)
        
        return error / len(idm_trajectory)

    def error(self, param, current_input:PlannerInput=None, debug=False) -> float:
        target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max = param.values()
        target_velocity_ = min(max(1.0, target_velocity + current_input.history.current_state[0].dynamic_car_state.speed), 30)
        self.idm_simulator.setUp(target_velocity=target_velocity_,
                                 min_gap_to_lead_agent=min_gap_to_lead_agent, 
                                 headway_time=headway_time, 
                                 accel_max=accel_max, 
                                 decel_max=decel_max)
        idm_trajectory: InterpolatedTrajectory = self.idm_simulator.compute_trajectory(current_input)
        idm_trajectory: list[EgoState] = idm_trajectory.get_sampled_trajectory()
        gt_trajectory_absolute = self.scenario.get_ego_future_trajectory(
            iteration=current_input.iteration.index, 
            num_samples=self.planned_trajectory_samples, 
            time_horizon=self.planned_trajectory_samples * 0.1
        )
        gt_trajectory_absolute: list[EgoState] = [state for state in gt_trajectory_absolute]

        error = 0
        for idm_state, gt_state in zip(idm_trajectory[1:], gt_trajectory_absolute):
            error += idm_state.center.distance_to(gt_state.center)
        error /= len(idm_trajectory)
        # if debug or error <= 0.8:
        if False:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot([state.center.x for state in idm_trajectory[1:]], 
                     [state.center.y for state in idm_trajectory[1:]], 
                     label="IDM")
            plt.plot([state.center.x for state in gt_trajectory_absolute],
                     [state.center.y for state in gt_trajectory_absolute],
                     label="GT")
            #添加标题和坐标轴标签
            plt.xlabel("x m")
            plt.ylabel("y m")
            plt.legend()
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"idm_pic_1/xy_idm_vs_gt_{timestamp}.png", dpi=600)
            plt.close()

            plt.figure()
            plt.plot([t * 0.1 for t in range(1, len(idm_trajectory))],
                     [state.dynamic_car_state.speed for state in idm_trajectory[1:]], 
                     label="IDM")
            plt.plot([t * 0.1 for t in range(1, len(idm_trajectory))],
                     [state.dynamic_car_state.speed for state in gt_trajectory_absolute],
                     label="GT")
            plt.xlabel("time s")
            plt.ylabel("speed m/s")
            plt.legend()
            plt.savefig(f"idm_pic_1/speed_idm_vs_gt_{timestamp}.png", dpi=600)
            plt.close()


        return error

    def optimize(self, current_input):
        study = optuna.create_study(direction="minimize")
        from functools import partial
        objective_function_with_iteration = partial(self.objective_function, current_input=current_input)
        study.optimize(objective_function_with_iteration, n_trials=200)
        error = self.error(study.best_params, current_input=current_input, debug=False)
        return list(study.best_params.values()), error