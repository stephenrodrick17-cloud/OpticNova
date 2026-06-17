
import shutil
import os

screenshots = [
    "C:\\Users\\asus\\OneDrive\\画像\\Screenshots\\Screenshot 2026-06-17 112145.png",
    "C:\\Users\\asus\\OneDrive\\画像\\Screenshots\\Screenshot 2026-06-17 112211.png",
    "C:\\Users\\asus\\OneDrive\\画像\\Screenshots\\Screenshot 2026-06-17 112237.png",
    "C:\\Users\\asus\\OneDrive\\画像\\Screenshots\\Screenshot 2026-06-17 112250.png",
    "C:\\Users\\asus\\OneDrive\\画像\\Screenshots\\Screenshot 2026-06-17 112311.png"
]

assets_dir = "assets"

for src in screenshots:
    if os.path.exists(src):
        filename = os.path.basename(src)
        dest = os.path.join(assets_dir, filename)
        shutil.copy(src, dest)
        print(f"Copied {filename} successfully!")
    else:
        print(f"Warning: {src} not found!")

print("\nDone!")
