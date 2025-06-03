import torch

path = "./output/final_dino/point_cloud/iteration_10000/scale_gate.pt"
state_dict = torch.load(path, map_location="cpu")

# Print keys and their shapes
for k, v in state_dict.items():
    print(f"{k}: {v.shape}")
