"""
This is the loss for the Discrete Mask Diffustion Model (DMDM).
It takes in the ground truth token values (B, L, N) where B is batch size and L is sequence lenght and N is the number of tokens(these are one hot),
it also takes in the predicted token values which is also size (B, L, N) but this is contionus and is from the model,
lastly it also takes in the time t which is size of B which values varies from [0, 1].

Of course the output is a single value which is the mean of the losses of all the values
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class DMDMLoss(nn.Module):

    def __init__(self):
        super(DMDMLoss, self).__init__()
        

    def forward(self, GT, P, t):
        # Get the dimentions
        B, L, N = GT.shape

        # We need to compute -(1/(t+1))*/sum(log(P[i]*GT[i]).T) for all i and then take the mean
        return (-1*(1/(t+1)))@((torch.log((P*GT).sum(dim=2))).sum(dim=1)) / B
        # This works as P*GT is element wise multication so it keeps size (B, L, N)
        # Then doing the .sum(dim=2) sums across the last dimention making the size (B, L)
        # Then you log it to get the surprise keeping it size (B, L)
        # Then you sum across the last dimention getting size (B)
        # Now the -1*(1/(t+1)) outputs size (B)
        # Then the @ is the dot product and sums them and then we get size ()
        # Laslty we divide by B to get the average
        





