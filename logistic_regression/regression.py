import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegressorBasic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressorBasic, self).__init__()
        self.w = nn.Parameter(
            torch.randn(output_dim, input_dim, dtype=torch.float32)
        )
        self.b = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))

    def forward(self, x):
        return torch.matmul(x, self.w.T) + self.b
    
class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
def plot_predictions(X, y, y_pred, title):
    sns.set_theme('notebook')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X.squeeze(), y=y.squeeze(), color='blue', label='true values')
    sns.lineplot(x=X.squeeze(), y=y_pred.squeeze(), color='red', label='predictions')
    plt.title(title)
    plt.legend()
    plt.show()

def train_model(model, X, y, X_val, y_val, lr=0.01, epochs=100):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # validation loss
        with torch.inference_mode():
            y_pred = model(X_val)
            loss = criterion(y_pred, y_val)
            val_losses.append(loss.item())

        print(f'Epoch {epoch}, train loss: {train_losses[-1]}, val loss: {val_losses[-1]}') 
    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    sns.set_theme('notebook')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_losses, label='train loss')
    sns.lineplot(data=val_losses, label='val loss')
    plt.legend()
    plt.show()