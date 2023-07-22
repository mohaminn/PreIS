from transformers import AutoTokenizer
import torch
import numpy as np
import random

# Configurations
RANDOM_SEED = 42
DEVICE = "cuda"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 200
ACCUMULATION_STEPS = 2
GAMMA_L = 0.1
GAMMA_G = 0.4
TRAIN_DATA_PATH = "./Data/train.csv"
VLID_DATA_PATH = "./Data/valid.csv"
TEST_DATA_PATH = "./Data/test.csv"
RITA_DIR = "./RITA/models--lightonai--RITA_s/snapshots/fced662eadd2b7099a3b92a88365dfc3c98eb3da"
SAVE_MODEL_TO = ""
SDA = False
FINE_TUNNING = True
tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_s")
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

