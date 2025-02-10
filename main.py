import torch  # noqa
import torch.nn as nn  # noqa
import torch.optim as optim  # noqa
import itertools  # noqa
import matplotlib.pyplot as plt  # noqa
import random
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

# =====================
# Training Configuration
# =====================
# SEQ_LEN = 64  # Examples per function (B14/B16 in paper)
# TASKS_PER_BATCH = 128  # Number of functions per meta-batch
SEQ_LEN = 40  # Examples per function (B14/B16 in paper)
TASKS_PER_BATCH = 40  # Number of functions per meta-batch
EPOCHS = 800  # Training epochs

# =====================
# Model Configuration
# =====================
HIDDEN_SIZE = 6  # LSTM: 6 hidden units, 6 memory cells (size 1)
LR = 0.01  # Learning rate (Table 1)
WEIGHT_RANGE = 0.1  # Initialization range [-0.1, 0.1]

# =====================
# Generate All 16 Boolean Functions
# =====================
# Create all possible 4-output combinations for 2-input Boolean functions
# We need them to be 4-output because there are 4 possible inputs for a 2-bit input Boolean function
# and therefore we need one bit for each input = 4 bits
ALL_TRUTH_TABLES = torch.tensor(
    list(itertools.product([0.0, 1.0], repeat=4)),  # 16 functions
    dtype=torch.float32,
)

# =====================
# TRAIN_TEST_SPLIT = 4 / 16  # 75% train, 25% test
TRAIN_TEST_SPLIT = 14 / 16  # 75% train, 25% test
EVAL_EVERY = 50  # Evaluate on test set every 50 epochs
TEST_EPISODES = 100  # Number of test episodes per evaluation

# Split boolean functions into train/test
num_train = int(len(ALL_TRUTH_TABLES) * TRAIN_TEST_SPLIT)
indices = torch.randperm(len(ALL_TRUTH_TABLES))
TRAIN_INDICES = indices[:num_train]
TEST_INDICES = indices[num_train:]


def generate_boolean_batch(batch_size, seq_len, train=True):
    """Generate a batch of boolean function tasks with their expected outputs.
    Each sequence uses a single boolean function throughout.

    Args:
        batch_size: Number of sequences (each using a different boolean function)
        seq_len: Length of each sequence
        train: If True, use training functions; if False, use test functions
    """
    # Step 1: Select one function for each sequence
    available_functions = TRAIN_INDICES if train else TEST_INDICES
    # Each sequence gets one function
    selected_function_indices = torch.randint(
        low=0,
        high=len(available_functions),
        size=(batch_size,),  # shape: (batch_size,)
    )
    selected_functions = available_functions[selected_function_indices]

    # Get the truth tables for our selected functions
    # Shape: (batch_size, 4) - each row is a complete truth table for one function
    selected_truth_tables = ALL_TRUTH_TABLES[selected_functions]

    # Step 2: Create input pairs
    all_possible_inputs = torch.tensor(
        [
            [0, 0],  # False False
            [0, 1],  # False True
            [1, 0],  # True False
            [1, 1],  # True True
        ],
        dtype=torch.float32,
    )

    # Generate random input pairs for each sequence
    random_input_choices = torch.randint(
        low=0,
        high=4,  # 4 possible input pairs
        size=(batch_size, seq_len),  # shape: (batch_size, seq_len)
    )

    # Generate inputs and outputs
    inputs = all_possible_inputs[
        random_input_choices
    ]  # shape: (batch_size, seq_len, 2)
    outputs = selected_truth_tables.gather(1, random_input_choices).unsqueeze(
        -1
    )  # shape: (batch_size, seq_len, 1)

    # Return the function used for each sequence
    return inputs, outputs


# =====================
# Paper-Accurate LSTM (6/6(1) Architecture)
# =====================
class MetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

        self._init_weights()

    def _init_weights(self):
        # 1. Uniform weight initialization
        for p in self.parameters():
            p.data.uniform_(-WEIGHT_RANGE, WEIGHT_RANGE)

        # 2. Input gate bias = -1.0 (critical for LSTM performance)
        with torch.no_grad():
            self.lstm.bias_ih_l0[:HIDDEN_SIZE].fill_(-1.0)
            self.lstm.bias_hh_l0[:HIDDEN_SIZE].fill_(-1.0)

    def squash_cell_input(self, x):
        """Squash input to [-2, 2] range using sigmoid"""
        return 4 * torch.sigmoid(x) - 2

    def squash_cell_output(self, x):
        """Squash output to [-1, 1] range using sigmoid"""
        return 2 * torch.sigmoid(x) - 1

    def forward(self, x, y_prev, hidden=None):
        inputs = torch.cat([x, y_prev], dim=-1)

        # Squash the inputs to [-2, 2] BEFORE the LSTM
        inputs = self.squash_cell_input(inputs)

        out, hidden = self.lstm(inputs, hidden)

        # Linear layer followed by output squashing to [-1, 1]
        out = self.fc(out)
        out = self.squash_cell_output(out)

        return out, hidden
