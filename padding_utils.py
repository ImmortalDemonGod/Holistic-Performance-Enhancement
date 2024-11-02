
import torch

# Central function for padding tensors to a fixed shape
def pad_to_fixed_size(tensor, target_shape=(30, 30), pad_value=10):
    """
    Pads the tensor to a fixed target shape using the specified pad value.
    Args:
        tensor (torch.Tensor): The tensor to be padded.
        target_shape (tuple): The target shape (rows, columns) to pad to.
        pad_value (int): The value to use for padding.
    Returns:
        torch.Tensor: Padded tensor with the target shape.
    """
    current_shape = tensor.shape
    pad_height = max(target_shape[0] - current_shape[0], 0)
    pad_width = max(target_shape[1] - current_shape[1], 0)
    padding = (0, pad_width, 0, pad_height)
    padded_tensor = torch.nn.functional.pad(tensor, padding, value=pad_value)
    return padded_tensor[:, :target_shape[1]]
