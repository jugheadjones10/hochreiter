import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from main import (
    EPOCHS,
    LR,
    SEQ_LEN,
    TASKS_PER_BATCH,
    TEST_INDICES,
    TRAIN_INDICES,
    MetaLearner,
    generate_boolean_batch,
)

# =====================
# Training Loop with Evaluation
# =====================
model = MetaLearner()
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(
    f"Training on {len(TRAIN_INDICES)} functions, Testing on {len(TEST_INDICES)} functions"
)

loss_history = []
for epoch in range(EPOCHS):
    # Generate a batch of tasks/functions
    # x shape: (TASKS_PER_BATCH, SEQ_LEN, 2)
    # y shape: (TASKS_PER_BATCH, SEQ_LEN, 1)
    # On each epoch, we generate a new shuffled batch of tasks (drawn from train set)
    x, y_true = generate_boolean_batch(TASKS_PER_BATCH, SEQ_LEN, train=True)

    # We'll accumulate loss for this one task before doing a single optimizer step
    task_loss = torch.tensor(0.0)

    # Loop through each task in the batch individually
    for task_i in range(TASKS_PER_BATCH):
        # Initialize hidden state for this specific task
        hidden = None
        # LSTM expects a (batch_size, 1, 1) shape for y_prev, so here batch_size=1 for a single task
        y_prev = torch.zeros(1, 1, 1)

        # Roll through the sequence for this task
        for t in range(SEQ_LEN):
            x_t = x[task_i : task_i + 1, t : t + 1, :]  # shape: (1, 1, input_dim)
            y_pred_t, hidden = model(
                x_t, y_prev, hidden
            )  # forward pass for one timestep

            # Transform y_true from [0,1] to [-1,1] to match model's output range
            y_true_scaled = 2 * y_true[task_i : task_i + 1, t : t + 1, :] - 1

            # Calculate loss at this timestep (both predictions and targets now in [-1,1])
            loss_t = criterion(y_pred_t, y_true_scaled)
            task_loss += loss_t

            y_prev = y_pred_t

    optimizer.zero_grad()
    task_loss.backward()
    optimizer.step()

    # (Optional) If you want a running average each epoch, you can log it here
    print(f"Epoch {epoch + 1}, Training Loss: {task_loss.item() / TASKS_PER_BATCH:.4f}")
    loss_history.append(task_loss.item() / TASKS_PER_BATCH)

    # Periodic evaluation
    # if (epoch + 1) % EVAL_EVERY == 0:
    #     test_mse, test_accuracy = evaluate_model(model, TEST_EPISODES)
    #     avg_task_loss = total_loss / TASKS_PER_BATCH
    #     print(f"Epoch {epoch + 1}")
    #     print(f"Training Loss (avg per task): {avg_task_loss:.4f}")
    #     print(f"Test MSE: {test_mse:.4f}")
    #     print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    #     print("-" * 40)

# Save model
torch.save(model.state_dict(), "model.pth")


# 3. After training completed, plot the loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()
