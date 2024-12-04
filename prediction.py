import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Hyperparameters
input_size = 6  # Same as the number of features used during training
hidden_size = 10
num_classes = 1  # Regression problem, single output value
model_path = 'Datasets/FF_Net_1.pth'  # Path to your trained model

# Define the model structure (same as during training)
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_size, 50)
        self.l3 = torch.nn.Linear(50, 20)
        self.l4 = torch.nn.Linear(20, 5)
        self.l5 = torch.nn.Linear(5, num_classes)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.l5(x)
        return x

# Load the trained model
model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Prepare the scaler used during training
scaler = MinMaxScaler()
# Here you should fit the scaler on the training data or load the fitted scaler if saved
# For demonstration, I'm using a dummy fit, you need to fit it on the same training data
# scaler.fit(training_data)  # Replace with actual training data used during model training

# Example input data (feature vector)
new_data = np.array([[0.5, 0.7, 0.1, 0.3, 0.6, 0.2]])  # Example feature vector

# Convert the input data to a PyTorch tensor
input_tensor = torch.from_numpy(new_data).float()

# Make the prediction
with torch.no_grad():  # No need to track gradients for inference
    prediction = model(input_tensor)

# Print the prediction
print("Predicted RUL (Remaining Useful Life):", prediction.item())