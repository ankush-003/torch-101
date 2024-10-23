import torch
from tqdm.notebook import tqdm
from .functions import *

class NeuralNetwork:
    def __init__(self, layer_dims: list[int]):
        """
        Initialize the neural network with the given layer dimensions.
        layer_dims: list containing the number of neurons in each layer.
        """
        self.params = {}
        self.device = torch.device('cpu') # CPU by default
        for i in range(1, len(layer_dims)):
            self.params['W' + str(i)] = torch.randn(layer_dims[i], layer_dims[i-1]) / torch.sqrt(torch.tensor(layer_dims[i-1], dtype=torch.float32))
            self.params['b' + str(i)] = torch.zeros((layer_dims[i], 1))
    
    def to(self, device):
        """
        Move the model parameters to the specified device (CPU or GPU).
        device: The device to move the model parameters to ('cpu' or 'cuda').
        """
        self.device = device
        for key in self.params:
            self.params[key] = self.params[key].to(self.device)
        return self
    
    def train(self, X: torch.Tensor, Y: torch.Tensor, num_iterations: int = 100, learning_rate: float = 0.001):
        """
        Train the neural network using gradient descent.
        X: input data.
        Y: output data.
        learning_rate: the learning rate for gradient descent.
        num_iterations: the number of iterations to train the model.
        """
        # Reshape X to (n_features, n_samples)
        X.to(self.device)
        Y.to(self.device)

        X = X.T
        Y = Y.T

        costs = []
        for i in tqdm(range(num_iterations), desc='Epochs ', leave=False):
            # Forward propagation
            AL, caches = model_forward(X, self.params)

            # Compute cost
            if Y.shape[0] == 1:
                cost = binary_cross_entropy_loss(AL, Y, device=self.device)
            else:
                cost = categorical_cross_entropy_loss(AL, Y, device=self.device)

            # Backward propagation
            grads = model_backward(AL, Y, caches, device=self.device)

            for key in grads:
                    torch.nn.utils.clip_grad_norm_(grads[key], max_norm=1.0)
                    
            self.params = update_parameters(self.params, grads, learning_rate)

            # Print the cost every 100 iterations
            if i % 100 == 0 or i == num_iterations - 1:
                print(f'Cost at iteration {i}: {cost.item()}')

            costs.append(cost.item())
        
        return costs
    
    def predict(self, X: torch.Tensor):
        """
        Predict the output given the input.
        X: input data.
        Returns: the predicted output.
        """
        X.to(self.device)
        X = X.T  # Reshape X to (n_features, n_samples)
        AL, _ = model_forward(X, self.params)
        return AL.T  # Reshape output back to (n_samples, n_classes)
    
    def evaluate(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Evaluate the model's performance.
        X: input data.
        Y: true labels.
        Returns: accuracy of the model.
        """
        predictions = self.predict(X)
        Y = Y.view(-1, 1)
        predicted_classes = (predictions > 0.5).float()
        accuracy = (predicted_classes == Y).float().mean()
        return accuracy.item()