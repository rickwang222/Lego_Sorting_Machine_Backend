#!/bin/bash

export PATH="/home/mart4322/blender-2.93.8-linux-x64:/home/mart4322/.local/bin:/home/mart4322/miniconda3/bin:/home/mart4322/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

source /home/mart4322/miniconda3/etc/profile.d/conda.sh
conda activate py310yolov5
cd /home/mart4322/InferenceServer/
python /home/mart4322/InferenceServer/server.py
