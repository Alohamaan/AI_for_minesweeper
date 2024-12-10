import torch
import torch.nn as nn
import torch.nn.functional as F
# маму разрабов тф ебал

# Define the custom model
class CustomModel(nn.Module):
    def __init__(self, dim1, dim2):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv2d(11, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = F.relu(self.conv1(x1))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = self.sigmoid(self.conv6(x))  # Apply sigmoid for activation
        output = x * x2  # Element-wise multiplication
        return output

# Define input shapes
dim1 = 16
dim2 = 30
input_shape = (11, dim1, dim2)

# Create model instance
model = CustomModel(dim1, dim2)

# Example inputs
in1 = torch.rand(1, 11, dim1, dim2)  # Batch size 1, 11 channels
in2 = torch.rand(1, 1, dim1, dim2)   # Batch size 1, 1 channel

# Forward pass
output = model(in1, in2)

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example loss calculation
target = torch.rand(1, 1, dim1, dim2)  # Random target for demonstration
loss = criterion(output, target)
print(f"Loss: {loss.item()}")

# Backward pass
loss.backward()
optimizer.step()
