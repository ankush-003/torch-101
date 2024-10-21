import numpy as np
import torch

def linear_forward(A: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    """
    Perform the linear part of a layer's forward propagation.
    A: activations from the previous layer.
    W: weights matrix.
    b: bias vector.
    Returns: 
    - Z = W*A + b
    - cache: a tuple containing A, W, and b (for backpropagation).
    """

    assert W.shape[1] == A.shape[0], "W and A have incompatible shapes."

    Z = torch.matmul(W, A) + b
    cache = (A, W, b)
    return Z, cache

def sigmoid(Z: torch.Tensor):
    """
    Compute the sigmoid activation function.
    Z: output of the linear layer.
    Returns:
    - A: the sigmoid of Z.
    - cache: Z (for backpropagation).
    """
    # A = 1 / (1 + torch.exp(-Z))
    func = torch.nn.Sigmoid()
    A = func(Z)
    cache = Z

    return A, cache

def relu(Z: torch.Tensor):
    """
    Compute the ReLU activation function.
    Z: output of the linear layer.
    Returns:
    - A: the ReLU of Z.
    - cache: Z (for backpropagation).
    """
    # A = torch.max(0, Z)
    func = torch.nn.ReLU()
    A = func(Z)
    cache = Z

    return A, cache

def sigmoid_backward(dA: torch.Tensor, cache):
    """
    Compute the gradient of the cost with respect to the sigmoid activation.
    dA: post-activation gradient.
    cache: 'Z' where we store for computing backward propagation efficiently.
    Returns: the gradient of the cost with respect to Z.
    """

    Z = cache
    # s = 1 / (1 + torch.exp(-Z))
    func = torch.nn.Sigmoid()
    s = func(Z)
    dZ = dA * s * (1 - s)

    return dZ

def relu_backward(dA: torch.Tensor, cache):
    """
    Compute the gradient of the cost with respect to the ReLU activation.
    dA: post-activation gradient.
    cache: 'Z' where we store for computing backward propagation efficiently.
    Returns: the gradient of the cost with respect to Z.
    """

    Z = cache
    dZ = torch.clone(dA) # copy dA to dZ
    dZ[Z <= 0] = 0 # when Z <= 0, set dZ to 0

    return dZ

def softmax(Z: torch.Tensor):
    """
    Compute the softmax activation function.
    
    Args:
    Z (torch.Tensor): The input tensor.
    
    Returns:
    tuple: (A, cache)
        A (torch.Tensor): Output of the softmax function.
        cache (torch.Tensor): Input Z, stored for backward pass.
    """
    exp_Z = torch.exp(Z - torch.max(Z, dim=0, keepdim=True)[0])  # for numerical stability
    A = exp_Z / torch.sum(exp_Z, dim=0, keepdim=True)
    cache = Z
    return A, cache

def softmax_backward(dA: torch.Tensor, cache: torch.Tensor):
    """
    Compute the gradient of the cost with respect to Z for the softmax function.
    
    Args:
    dA (torch.Tensor): Gradient of the cost with respect to the output A.
    cache (torch.Tensor): Input Z from the forward pass.
    
    Returns:
    torch.Tensor: Gradient of the cost with respect to Z.
    """
    Z = cache
    A, _ = softmax(Z)
    # Compute the gradient
    dZ = A - dA
    return dZ

def linear_activation_forward(A_prev: torch.Tensor, W: torch.Tensor, b: torch.Tensor, activation: str):
    """
    Linear -> Activation
    Perform the linear and activation parts of a layer's forward propagation.
    A_prev: activations from the previous layer.
    W: weights matrix.
    b: bias vector.
    activation: the activation function to use.
    Returns:
    - A: the output of the activation function.
    - cache: a tuple containing the linear cache and activation cache (for backpropagation).
    """

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def model_forward(X: torch.Tensor, params: dict):
    """
    Forward propagation for the entire neural network.
    X: input data.
    params: dictionary containing the weights and biases for each layer.
    Returns:
    - AL: the output of the last layer.
    - caches: a list of caches containing every cache of linear_activation_forward().
    """

    caches = []
    # Input layer
    A = X
    L = len(params) // 2 # number of layers in the neural network

    # Hidden layers
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, params[f'W{l}'], params[f'b{l}'], "relu")
        caches.append(cache)

    # Output layer
    AL, cache = linear_activation_forward(A, params[f'W{L}'], params[f'b{L}'], "sigmoid")
    caches.append(cache)

    return AL, caches

def binary_cross_entropy_loss(AL: torch.Tensor, Y: torch.Tensor):
    """
    Compute the binary cross-entropy loss.
    AL: probability vector corresponding to the label predictions.
    Y: true "label" vector.
    Returns: the loss.
    """
    m = Y.shape[1]
    epsilon = 1e-15
    AL = torch.clamp(AL, epsilon, 1 - epsilon)  # Avoid log(0)
    loss = -1/m * torch.sum(Y * torch.log(AL) + (1 - Y) * torch.log(1 - AL))
    return loss.squeeze()

def binary_cross_entropy_loss_backward(AL: torch.Tensor, Y: torch.Tensor):
    """
    Compute the gradient of the cost with respect to AL for binary cross-entropy loss.
    AL: probability vector corresponding to the label predictions.
    Y: true "label" vector.
    Returns: the gradient of the cost with respect to AL.
    """
    m = Y.shape[1]
    epsilon = 1e-15
    AL = torch.clamp(AL, epsilon, 1 - epsilon)  # Avoid division by zero
    dAL = - (torch.div(Y, AL) - torch.div(1 - Y, 1 - AL))
    return dAL

def categorical_cross_entropy_loss(AL: torch.Tensor, Y: torch.Tensor, device='cpu'):
    """
    Compute the categorical cross-entropy loss.
    AL: probability vector corresponding to the label predictions, shape (num_classes, num_examples)
    Y: true "label" vector (one-hot encoded), shape (num_classes, num_examples)
    Returns: the loss
    """
    m = Y.shape[1]
    epsilon = 1e-15
    AL_clipped = torch.clamp(AL, min=epsilon, max=1-epsilon)
    AL_clipped = AL_clipped.to(device)
    loss = -1/m * torch.sum(Y * torch.log(AL_clipped))
    return loss.squeeze()

def categorical_cross_entropy_loss_backward(AL: torch.Tensor, Y: torch.Tensor, device='cpu'):
    """
    Compute the gradient of the cost with respect to AL for categorical cross-entropy loss.
    AL: probability vector corresponding to the label predictions, shape (num_classes, num_examples)
    Y: true "label" vector (one-hot encoded), shape (num_classes, num_examples)
    Returns: the gradient of the cost with respect to AL
    """
    m = Y.shape[1]
    epsilon = 1e-15
    AL_clipped = torch.clamp(AL, min=epsilon, max=1-epsilon)
    AL_clipped = AL_clipped.to(device)
    dAL = -1/m * Y / AL_clipped
    return dAL

def linear_backward(dZ: torch.Tensor, cache: dict):
    """
    Perform the linear part of backpropagation for a single layer.
    dZ: gradient of the cost with respect to the linear output of the current layer.
    cache: tuple of values (A_prev, W, b) from the forward propagation.
    Returns:
    - dA_prev: gradient of the cost with respect to the activation of the previous layer.
    - dW: gradient of the cost with respect to W.
    - db: gradient of the cost with respect to b.
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * torch.matmul(dZ, A_prev.T)
    db = 1/m * torch.sum(dZ, dim=1, keepdim=True)
    dA_prev = torch.matmul(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA: torch.Tensor, cache: dict, activation: str):
    """
    Linear -> Activation
    Perform the backpropagation for the linear and activation parts of a layer.
    dA: post-activation gradient for current layer l.
    cache: tuple of values (linear_cache, activation_cache).
    activation: the activation function to use.
    Returns:
    - dA_prev: gradient of the cost with respect to the activation of the previous layer.
    - dW: gradient of the cost with respect to W.
    - db: gradient of the cost with respect to b.
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def model_backward(AL: torch.Tensor, Y: torch.Tensor, caches, device='cpu'):
    """
    Backpropagation for the entire neural network.
    AL: probability vector, output of the forward propagation.
    Y: true "label" vector.
    caches: list of caches containing every cache of linear_activation_forward().
    Returns: a dictionary with the gradients.
    """
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.view(AL.shape) # make sure Y is the same shape as AL
    
    dAL = binary_cross_entropy_loss_backward(AL, Y)
    activation = "sigmoid"

    # Lth layer (softmax/sigmoid -> linear) gradients
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation)

    # Loop through all layers
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f"dA{l + 1}"], current_cache, "relu")
        grads[f"dA{l}"] = dA_prev_temp
        grads[f"dW{l + 1}"] = dW_temp
        grads[f"db{l + 1}"] = db_temp

    return grads

def update_parameters(params: dict, grads: dict, learning_rate: float):
    """
    Update the parameters using gradient descent.
    params: dictionary containing the weights and biases for each layer.
    grads: dictionary containing the gradients for each layer.
    learning_rate: the learning rate to use.
    """

    L = len(params) // 2 # number of layers in the neural network

    for l in range(L):
        params[f"W{l+1}"] = params[f"W{l+1}"] - learning_rate * grads[f"dW{l+1}"]
        params[f"b{l+1}"] = params[f"b{l+1}"] - learning_rate * grads[f"db{l+1}"]

    return params