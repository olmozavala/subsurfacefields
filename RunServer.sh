#!/bin/bash

PYTHONPATH=""
PYTHONPATH=$PYTHONPATH:/home/olmozavala/Dropbox/MyProjects/OZ_LIB/image_visualization
PYTHONPATH=$PYTHONPATH:/home/olmozavala/Dropbox/MyProjects/OZ_LIB/eoas_preprocessing
export PYTHONPATH=$PYTHONPATH:/home/olmozavala/Dropbox/MyProjects/OZ_LIB/AI_Common

echo $PYTHONPATH
SRC_PATH="/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/SubsurfaceFields"

echo '############################ Running server.. ############################ '
python $SRC_PATH/7_Dash_Error_Stats.py