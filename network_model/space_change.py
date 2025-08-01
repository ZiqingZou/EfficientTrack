import torch


def sincos_to_rad(state):
    """Converts the state from sine and cosine to radians.

    Args:
        state (torch.Tensor(batch_size, state_dim=16)): The state tensor with sine and cosine values.

    Returns:
        pos (torch.Tensor(batch_size, 4)): joints pos angle in (rad).
        vel (torch.Tensor(batch_size, 4)): joints angle velocity in (rad/s).
    """

    state_rad = torch.atan2(state[:, :8], state[:, 8:])
    pos = state_rad[:, :4]
    vel = state_rad[:, 4:]

    return pos, vel


def sincos_to_rad_for_pos(s):
    """Converts the state from sine and cosine to radians.

    Args:
        s (torch.Tensor(batch_size, state_dim=16/8)): The state/target tensor with sine and cosine values.

    Returns:
        pos (torch.Tensor(batch_size, 4)): joints pos angle in (rad).
    """
    if s.shape[1] == 16:
        state_rad = torch.atan2(s[:, :8], s[:, 8:])
        pos = state_rad[:, :4]
    elif s.shape[1] == 8:
        pos = torch.atan2(s[:, :4], s[:, 4:])
    else:
        raise ValueError("Invalid state shape")

    return pos


def rad_to_sincos(s):
    """Converts the state from radians to sine and cosine.

    Args:
        s (torch.Tensor): The state/target tensor with radians values.

    Returns:
        torch.Tensor: The state/target tensor with sine and cosine values.
    """

    return torch.cat([torch.sin(s), torch.cos(s)], dim=-1)


def sincos_to_rad_special_v(state, freq):
    """Converts the state from sine and cosine to radians.

    Args:
        state (torch.Tensor): The state tensor with sine and cosine values.
        freq (int): The frequency of the state.

    Returns:
        pos (torch.Tensor(batch_size, 4)): joints pos angle in (rad).
        vel (torch.Tensor(batch_size, 4)): joints angle velocity in (rad/s).
    """

    pos = torch.atan2(state[:, :4], state[:, 4:8])
    vel = (torch.atan2((state[:, 8:12] + state[:, :4]), (state[:, 12:] + state[:, 4:8])) - pos) * freq
        
    return pos, vel


def rad_to_sincos_special_v(state, freq):
    """Converts the state from radians to sine and cosine.

    Args:
        state (torch.Tensor): The state tensor with radians values.
        freq (int): The frequency of the state.

    Returns:
        torch.Tensor: The state tensor with sine and cosine values.
    """
    if state.dim() == 2:
        sin_pos = torch.sin(state[:, :4])
        cos_pos = torch.cos(state[:, :4])

        sin_vel = torch.sin(state[:, :4] + state[:, 4:] / freq) - sin_pos
        cos_vel = torch.cos(state[:, :4] + state[:, 4:] / freq) - cos_pos
    elif state.dim() == 3:
        sin_pos = torch.sin(state[:, :, :4])
        cos_pos = torch.cos(state[:, :, :4])

        sin_vel = torch.sin(state[:, :, :4] + state[:, :, 4:] / freq) - sin_pos
        cos_vel = torch.cos(state[:, :, :4] + state[:, :, 4:] / freq) - cos_pos
    else:
        raise ValueError("Invalid state tensor dimension")

    return torch.cat([sin_pos, cos_pos, sin_vel, cos_vel], dim=-1)


