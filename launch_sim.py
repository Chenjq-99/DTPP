import yaml
import datetime
import torch
import argparse
import warnings
from tqdm import tqdm
from planner import Planner
from common_utils import *
warnings.filterwarnings("ignore") 

from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.nuboard.base.data_class import NuBoardFile

from common_utils import get_filter_parameters, get_scenario_map, get_filter_parameters_for_changing_lane

def build_nuboard(scenario_builder, simulation_path):
    nuboard = NuBoard(
        nuboard_paths=simulation_path,
        scenario_builder=scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=5006
    )

    nuboard.run()

def main(args):
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)
    simulation_file = [args.nuboard_file if args.nuboard_file else None]

    # show metrics and scenarios
    build_nuboard(builder, simulation_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='~/nuplan/dataset/nuplan-v1.1/splits/train')
    parser.add_argument('--map_path', type=str, default='~/nuplan/dataset/maps')
    parser.add_argument('--nuboard_file', type=str)
    args = parser.parse_args()

    main(args)
