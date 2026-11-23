
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import numpy as np

root = tk.Tk()
root.withdraw()

selected_folder = filedialog.askdirectory(
    title="Select MRI dataset folder"
)

if not selected_folder:
    print("No folder selected.")
    raise SystemExit

MRI_ROOT = Path(selected_folder)

ALLOWED_SHAPES = {
    (204, 256, 48),
    (228, 256, 54)
}

EXPECTED_SPACING = np.array([0.9375, 0.9375, 3.0])
TOLERANCE = 0.01

total_compatible = 0

for class_name in ["Healthy", "Unhealthy"]:
    class_folder = MRI_ROOT / class_name
    compatible = []

    if not class_folder.exists():
        print(f"\n{class_name}: folder not found")
        continue

    for p in class_folder.rglob("*"):
        if not p.is_file() or not (
            p.name.lower().endswith(".nii")
            or p.name.lower().endswith(".nii.gz")
        ):
            continue

        try:
            img = nib.load(str(p))
            shape = tuple(img.shape[:3])
            spacing = np.array(img.header.get_zooms()[:3])

            if (
                shape in ALLOWED_SHAPES
                and np.allclose(
                    spacing,
                    EXPECTED_SPACING,
                    atol=TOLERANCE
                )
            ):
                compatible.append(p)

        except Exception as e:
            print("Error reading:", p.name, e)

    print(f"\n========== {class_name.upper()} ==========")
    print("Compatible files:", len(compatible))

    for p in compatible:
        print(p.name)

    total_compatible += len(compatible)

print("\nTOTAL COMPATIBLE MRI FILES:", total_compatible)

root.destroy()
