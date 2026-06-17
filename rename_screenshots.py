
import os

assets_dir = "assets"

# Map original filenames to descriptive names
rename_map = {
    "Screenshot 2026-06-17 112145.png": "homepage.png",
    "Screenshot 2026-06-17 112211.png": "diagnostic-workstation.png",
    "Screenshot 2026-06-17 112237.png": "segmentation-tab.png",
    "Screenshot 2026-06-17 112250.png": "vessels-tab.png",
    "Screenshot 2026-06-17 112311.png": "mathematical-proofs-tab.png"
}

for old_name, new_name in rename_map.items():
    old_path = os.path.join(assets_dir, old_name)
    new_path = os.path.join(assets_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")

print("\nDone!")
