import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

import composer
from composer.core import Evaluator
from composer import Time, TimeUnit
from composer.utils import dist
from composer.models.huggingface import HuggingFaceModel
from torchmetrics.classification import MulticlassAccuracy, MulticlassMatthewsCorrCoef, MulticlassF1Score
from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef
from composer import Trainer
from composer.callbacks import LRMonitor, RuntimeEstimator
from composer.loggers import WandBLogger
from pruners.PMGP import PMGP_Algorithm
from pruners.PLATON import PLATON_Algorithm