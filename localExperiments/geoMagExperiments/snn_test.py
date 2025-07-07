# Reproduce Table-5 LSTM / GRU baseline models 
import os, random, time, sys

# Fix the path to correctly import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
import spikingjelly as sj
from spikingjelly.event_driven import neuron
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import logger and checkpoint utilities
from utils.utils import get_logger, save_checkpoint, plot_confusion_matrix

# ------------------------ 0. Setup logging and result directory ------------- #
RESULT_DIR = "localExperiments/geoMagExperiments/model_result/seqModelResult"
CHECKPOINT_DIR = "localExperiments/geoMagExperiments/model_result/checkPoints"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize logger
logger = get_logger(
    filename=os.path.join(RESULT_DIR, "lstm_gru_experiment.log"),
    name="LSTM_GRU_Experiment",
    level="INFO",
    overwrite=True,
    to_stdout=True
)