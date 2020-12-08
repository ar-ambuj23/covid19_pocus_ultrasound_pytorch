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
    t = torch.diag(torch.matmul(S, A))
    L = torch.sum(t)
    # do the backwards pass, and extract the gradient
    L.backward()
    w = X.grad.detach().clone()

    X.grad.zero_() # set gradients to zero after doing the backwards pass

    return w # think of this as dL/dX

def compute_saliency_maps(X, y, num_classes, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    A = get_selector_matrix(y, X.shape[0], num_classes).cuda() 
    w = img_grad(X,A,model)
    # Take the maximum magnitude of w across all color channels, output the result
    max_values, max_idxs = torch.max(w,1)
    saliency = max_values
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return saliency