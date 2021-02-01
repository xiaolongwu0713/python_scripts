from src.data import DataBuildClassifier
import shutil
import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import single_auc_loging
from src.utils import prepare_dirs,write_results_table, separte_last_block
from src.model_torch import train_model_eegnet
import shutil
from sklearn.model_selection import StratifiedKFold