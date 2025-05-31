import matplotlib.pyplot as plt
import re

val_losses = []
train_losses = []

# Open the output file and read its lines
with open('cora_tune8.out', 'r') as file:
    for line in file:
        # Find lines that contain 'val loss' or 'train loss'
        if 'Val loss' in line:
            # Extract the number after 'val loss' using regex
            loss = float(re.search('Val loss ([0-9\.]+)', line).group(1))
            val_losses.append(loss)
            print("Val loss", loss)
        elif 'Train loss' in line:
            # Extract the number after 'train loss' using regex
            loss = float(re.search('Train loss ([0-9\.]+)', line).group(1))
            train_losses.append(loss)
            print("train loss:", loss)

# Plot the losses
epochs = range(1, len(val_losses) + 1)
plt.plot(epochs, val_losses, color='blue', label='Validation Loss')
plt.plot(epochs, train_losses, color='red', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("train_loss_val.png")