from cs336_basics.train import SGD
import torch

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
for lr in [1e1, 1e2, 1e3]:
    opt = SGD([weights], lr=lr)
    print("#"*10, "lr =", lr, "#"*10)

    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.