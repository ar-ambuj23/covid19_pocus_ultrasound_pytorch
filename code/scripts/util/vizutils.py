import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X

def get_selector_matrix(y, num_batch, num_classes):
    N = num_batch
    C = num_classes
    sel_mat = torch.zeros((C,N))
    for i in range(0,N):
        sel_mat[y[i],i] = 1
    return sel_mat

# Assumes X is the batch of input images, A is the "selector" matrix
def img_grad(X, A, model, reg_lambda=0):
    # do the forward pass
    S = model(X) # S holds the unnormalized scores for each image in the batch
    # print(S)
    t = torch.diag(torch.matmul(S, A))
    L = torch.sum(t) - reg_lambda*torch.pow(torch.norm(X,2),2) # note the NEGATIVE sign!
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

def class_visualization(target_y, model, device, num_classes, channel_means, channel_std_devs, deprocess_func, class_names_dict, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    - dtype: Torch datatype to use for computations

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    model.to(device)
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
    img = torch.randn(1, 3, 224, 224).mul_(1.0).to(device).requires_grad_()

    ########################################################################
    #                             START OF MY CODE                         #
    ########################################################################
    tgt_y = np.array([target_y])
    A = get_selector_matrix(tgt_y, img.shape[0], num_classes).cuda()
    ########################################################################
    #                             END OF MY CODE                           #
    ########################################################################

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))

        ########################################################################
        # TODO: Use the model to compute the gradient of the score for the     #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. Don't forget the #
        # L2 regularization term!                                              #
        # Be very careful about the signs of elements in your code.            #
        ########################################################################
    
        # Compute the gradient of S_c(I) - λ*norm(I)^2 . Be careful of the sign!
        g = img_grad(img,A,model,reg_lambda=l2_reg)
        with torch.no_grad():
            # compute gradient step (using a normalized gradient?)
            d_img = learning_rate * g / torch.norm(g,2)
            img += d_img # Do gradient ascent, NOT gradient descent! 
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    
    # Undo the random jitter
    img.data.copy_(jitter(img.data, -ox, -oy))

    # As regularizer, clamp and periodically blur the image
    for c in range(3):
        lo = float(-channel_means[c] / channel_std_devs[c])
        hi = float((1.0 - channel_means[c]) / channel_std_devs[c])
        img.data[:, c].clamp_(min=lo, max=hi)
    if t % blur_every == 0:
        blur_image(img.data, sigma=0.5)

    # Periodically show the image
    if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
        img_data = img.data.clone().cpu()
        plt.imshow(deprocess_func(img_data))
        class_name = class_names_dict[target_y]
        plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
        plt.gcf().set_size_inches(4, 4)
        plt.axis('off')
        plt.show()

    return deprocess_func(img.data.cpu())