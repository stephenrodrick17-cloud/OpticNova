
import requests
from PIL import Image
import io

# We'll use a high-quality eye image from Unsplash (similar to the one provided)
logo_url = "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=512"
response = requests.get(logo_url)
img = Image.open(io.BytesIO(response.content))

# Save to assets folder
img.save("assets/opticnova-logo.png", "PNG")
print("Logo saved successfully to assets/opticnova-logo.png!")
