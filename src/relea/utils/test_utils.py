import torch

def shape_test(tensor, reference_tensor):
    assert tensor.shape == reference_tensor.shape, \
        f"left tensor has shape {tensor.shape}, while right tensor has shape {reference_tensor.shape}"

def equality_test(tensor, reference_tensor):
    matches = torch.isclose(tensor, reference_tensor).sum()
    matches_percentage = matches / tensor.numel()

    print(f"{matches_percentage * 100}% of values matched.")
    
    assert torch.allclose(tensor, reference_tensor), \
        "The two tensors are not equal."