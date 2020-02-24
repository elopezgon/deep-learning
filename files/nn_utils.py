import numpy as np

def loss_surface(model_parameters, target_weights, target_ix, model, y):
    """
    Function to return the cost surface between two chosen weights
    in a model and its L2 loss surface.
    
    Parameters
    ----------
    model_parameters: tuple containing the parameters to compute the model
    target_weights: tuple of lenght 2 containing the parameters compute the loss
    target_ix: typle of lenght 2 containing the indices for each parameter to
               compute the loss
    model: function with parameters "model_parameters" used to compute the output of
           the model
    y: vector with m model outputs
    
    Returns
    -------
    tuppple with grid containing the surface along the chosen parameters
    """
    # target weights
    tweight1, tweight2 = target_weights
    # target indices
    tix1, tix2 = target_ix
    
    # Grid surface for target weights to plot
    tw1, tw2 = np.mgrid[-5:5:0.1, -5:5:0.1]
    # Initialize cost surface
    J = np.zeros_like(tw1.ravel())
    
    for ix, (tw1i, tw2i) in enumerate(zip(tw1.ravel(), tw2.ravel())):
        tweight1[tix1] = tw1i
        tweight2[tix2] = tw2i
        yhat = model(*model_parameters)
        # Updating cost function
        try:
            J[ix] =  np.sum((yhat - y) ** 2) / 2
        except Exception:
            print(ix)
        
    J = J.reshape(*tw1.shape)
    return tw1, tw2, J
