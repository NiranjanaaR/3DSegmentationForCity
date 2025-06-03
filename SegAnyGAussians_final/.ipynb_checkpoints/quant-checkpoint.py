import pyvista as pv
from PIL import Image
import torch
import torchvision.transforms as T
from piq import niqe
import os

# ---------- Settings ----------
ply_folder = "segmented_plys"
output_folder = "rendered_pngs"
os.makedirs(output_folder, exist_ok=True)

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

results = []

for file in os.listdir(ply_folder):
    if file.endswith(".ply"):
        ply_path = os.path.join(ply_folder, file)
        output_png = os.path.join(output_folder, file.replace(".ply", ".png"))

        # Step 1: Load and render with PyVista
        plotter = pv.Plotter(off_screen=True)
        mesh = pv.read(ply_path)
        plotter.add_mesh(mesh, show_scalar_bar=False)
        plotter.camera_position = 'xy'  # Top-down or adjust as needed
        plotter.show(screenshot=output_png)
        plotter.close()

        # Step 2: Compute NIQE
        img = Image.open(output_png).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        score = niqe(img_tensor)

        results.append((file, round(score.item(), 4)))
        print(f"{file}: NIQE = {score.item():.4f}")

# ---------- Summary ----------
print("\nNIQE Results:")
for name, score in results:
    print(f"{name}: {score}")
