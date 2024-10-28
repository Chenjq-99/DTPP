#!/bin/bash

python data_process.py \
--data_path ~/nuplan/dataset/nuplan-v1.1/splits/train \
--map_path ~/nuplan/dataset/maps \
--save_path ~/nuplan/processed_data/train/all_changing_lane_sample_9 || true

python data_process.py \
--data_path ~/nuplan/dataset/nuplan-v1.1/splits/train_vegas_1 \
--map_path ~/nuplan/dataset/maps \
--save_path ~/nuplan/processed_data/train/all_changing_lane_sample_9 || true

python data_process.py \
--data_path ~/nuplan/dataset/nuplan-v1.1/splits/train_vegas_2 \
--map_path ~/nuplan/dataset/maps \
--save_path ~/nuplan/processed_data/train/all_changing_lane_sample_9 || true

python data_process.py \
--data_path ~/nuplan/dataset/nuplan-v1.1/splits/train_vegas_3 \
--map_path ~/nuplan/dataset/maps \
--save_path ~/nuplan/processed_data/train/all_changing_lane_sample_9 || true

python data_process.py \
--data_path ~/nuplan/dataset/nuplan-v1.1/splits/train_vegas_4 \
--map_path ~/nuplan/dataset/maps \
--save_path ~/nuplan/processed_data/train/all_changing_lane_sample_9 || true