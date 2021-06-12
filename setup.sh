#!/bin/bash

conda install python=3.8.10 -y
pip install tensorflow-model-analysis
conda install imageio tqdm matplotlib IPython pandas -y
pip install git+https://github.com/tensorflow/docs
conda install -c conda-forge tensorflow-hub opencv matplotlib -y