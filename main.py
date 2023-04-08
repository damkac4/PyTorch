from pathlib import Path
import torch
from torch import nn


# Create *known* parameters
weight = 0.7
bias = 0.3
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))  # 80% of data used for training set, 20% for testing
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Create a Linear Regression model class
# class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(
#             torch.randn(1, dtype=torch.float))
#
#         self.bias = nn.Parameter(
#             torch.randn(1, dtype=torch.float))
#
#     # Forward defines the computation in the model
#     def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
#         return self.weights * x + self.bias  # <- this is the linear regression formula (y = m*x + b)

# Create a Linear Regression model class (different way)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
         # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)




# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()
print(model_0.state_dict())

# Check the nn.Parameter(s) within the nn.Module subclass we created
print("The model original values for weights and bias:")
print(model_0.state_dict())

# Set model to GPU if it's availalble, otherwise it'll default to CPU
model_0.to(device)

# Create the loss function
loss_fn = nn.L1Loss()

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01)


# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

### Training


# Set the number of epochs (how many times the model will pass over the training data)
epochs = 300

for epoch in range(epochs):

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside
    y_pred = model_0(X_train)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. The optimizers gradients are set to zero (they are accumulated by default) so they can be recalculated for the specific training step.
    optimizer.zero_grad()

    # 4. Computes the gradient of the loss with respect for every model parameter to be updated (each parameter with requires_grad=True). This is known as backpropagation.
    loss.backward()

    # 5. Udates parameters with requires_grad=True with respect to the loss gradients in order to improve them.
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode for testing
    model_0.eval()
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model_0(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")










# Find our model's learned parameters
print("\nThe model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")






###### SAVING AND LOADING ######

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "pytorch_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"\nSaving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print("\nThe model loaded values for weights and bias:")
print(loaded_model_0.state_dict())