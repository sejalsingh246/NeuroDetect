
from pathlib import Path
import csv
import requests
import nibabel as nib
import numpy as np
from tkinter import Tk, filedialog

API_URL = "http://127.0.0.1:8000/predict/mri"

# Select the parent folder containing Healthy and Unhealthy
root = Tk()
root.withdraw()
dataset_root = filedialog.askdirectory(
    title="Select Parkinson Multi Model DATASET folder"
)
root.destroy()

if not dataset_root:
    raise SystemExit("No folder selected.")

dataset_root = Path(dataset_root)
results = []

accepted_shapes = {
    (204, 256, 48),
    (228, 256, 54)
}

for class_name in ["Healthy", "Unhealthy"]:
    class_folder = dataset_root / class_name

    if not class_folder.exists():
        print(f"Missing folder: {class_folder}")
        continue

    files = sorted(class_folder.rglob("*.nii"))

    for file_path in files:
        try:
            nii = nib.load(str(file_path))
            shape = nii.shape[:3]
            spacing = nii.header.get_zooms()[:3]

            if shape not in accepted_shapes:
                continue

            if not np.allclose(spacing, (0.9375, 0.9375, 3.0),
                               atol=0.002):
                continue

            with open(file_path, "rb") as f:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            file_path.name,
                            f,
                            "application/octet-stream"
                        )
                    },
                    timeout=180
                )

            if response.ok:
                data = response.json()
                status = "SUCCESS"
                prediction = data.get("class_name")
                probabilities = data.get("probabilities")
                print(
                    f"[SUCCESS] {class_name}/{file_path.name} "
                    f"-> {prediction}"
                )
            else:
                status = f"HTTP {response.status_code}"
                prediction = response.text
                probabilities = None
                print(f"[FAILED] {file_path.name}: {response.text}")

        except Exception as exc:
            status = "ERROR"
            prediction = str(exc)
            probabilities = None
            print(f"[ERROR] {file_path.name}: {exc}")

        results.append({
            "actual_folder": class_name,
            "filename": file_path.name,
            "status": status,
            "prediction": prediction,
            "probabilities": str(probabilities)
        })

output_file = dataset_root / "mri_batch_results.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "actual_folder", "filename", "status",
        "prediction", "probabilities"
    ])
    writer.writeheader()
    writer.writerows(results)

successful = sum(r["status"] == "SUCCESS" for r in results)

print("\n========== BATCH TEST SUMMARY ==========")
print(f"Files tested: {len(results)}")
print(f"Successful: {successful}")
print(f"Failed: {len(results) - successful}")
print(f"Results saved to: {output_file}")
