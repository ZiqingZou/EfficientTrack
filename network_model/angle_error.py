import torch


def angle_error(next_pos_predict, next_pos):
    """ Calculate angle error between predicted and target angles

    Args:
        next_pos_predict (torch.Tensor(batch_size, dim_pos)): Predicted angles, range (-pi, pi)
        next_pos (torch.Tensor(batch_size, dim_pos)): Target angles, range (-pi, pi)

    Returns:
        torch.Tensor: Angle error(torch.Tensor(batch_size, dim_pos)), range (-pi, pi)
    """

    error = next_pos_predict - next_pos
    # error_new = torch.atan2(torch.sin(error), torch.cos(error))

    return error
    
