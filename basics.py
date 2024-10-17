import torch
# using Apple Metal GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tensor = torch.tensor([1, 2, 3], device=device, dtype=torch.float32)

# print(tensor.device)

# matrix multiplication
x = torch.tensor([1, 2, 3], device=device, dtype=torch.float32)
y = torch.nn.Linear(3, 2).to(device)
print(y(x))
# print("weights ", y.weight)
# print("bias ", y.bias)
# under the hood, the following is happening
res = torch.matmul(x, y.weight.T) + y.bias
print(res)