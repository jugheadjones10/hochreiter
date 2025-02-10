# Load model and eval

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from main import (  # Adjust imports as needed
    SEQ_LEN,
    TASKS_PER_BATCH,
    MetaLearner,
    generate_boolean_batch,
)

TASKS_PER_BATCH = 5


def evaluate_model(model):
    """Evaluate model on test set, matching per-task iteration used in training."""
    criterion = nn.MSELoss()
    mse_history = []
    accuracy_history = []

    with torch.no_grad():
        # Generate a batch of tasks (just like in training)
        x, y_true = generate_boolean_batch(TASKS_PER_BATCH, SEQ_LEN, train=False)

        # Loop over each task in the batch individually
        for task_i in range(TASKS_PER_BATCH):
            # Initialize hidden state for this task
            hidden = None
            y_prev = torch.zeros(1, 1, 1)

            # Roll through the sequence for this single task
            for t in range(SEQ_LEN):
                x_t = x[task_i : task_i + 1, t : t + 1, :]
                y_pred_t, hidden = model(x_t, y_prev, hidden)

                # Compute MSE for this timestep
                mse_t = criterion(y_pred_t, y_true[task_i : task_i + 1, t : t + 1, :])
                mse_history.append(mse_t.item())

                # Compute binary accuracy at threshold=0.5
                predictions = (y_pred_t > 0.0).float()

                # Update y_prev for next timestep
                y_prev = y_pred_t.detach()
                # Detach hidden to avoid unnecessary graph accumulation
                hidden = tuple(h.detach() for h in hidden)

    return mse_history, accuracy_history


# 1. Create an instance of the model with the same architecture
model = MetaLearner()

# 2. Load the saved weights
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# 3. Run evaluation
mse_history, accuracy_history = evaluate_model(model)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(mse_history, label="MSE")
plt.xlabel("Cycle")
plt.ylabel("MSE")
plt.title("MSE Over Time")
plt.legend()
plt.show()
