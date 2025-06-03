import torch

mask = torch.load("precomputed_mask.pt", map_location="cpu")

print("Type:", type(mask))
if torch.is_tensor(mask):
    print("Dtype:", mask.dtype)
    print("Shape:", mask.shape)
    print("Unique values:", torch.unique(mask))
