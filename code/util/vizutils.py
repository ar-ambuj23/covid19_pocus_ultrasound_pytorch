import torch

def get_selector_matrix(y, num_batch, num_classes):
    N = num_batch
    C = num_classes
    sel_mat = torch.zeros((C,N))
    for i in range(0,N):
        sel_mat[y[i],i] = 1
    return sel_mat

# Assumes X is the batch of input images, A is the "selector" matrix
def img_grad(X, A, model):
    # do the forward pass
    S = model(X) # S holds the unnormalized scores for each image in the batch
    print(torch.min(S)) 
    t = torch.diag(torch.matmul(S, A))
    L = torch.sum(t)
    # do the backwards pass, and extract the gradient
    L.backward()
    w = X.grad.detach().clone()

    X.grad.zero_() # set gradients to zero after doing the backwards pass

    return w # think of this as dL/dX