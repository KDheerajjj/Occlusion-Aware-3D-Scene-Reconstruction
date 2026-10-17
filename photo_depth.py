import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch

print("Loading Depth Anything model...")
# Using the Small model for efficiency; ensure you have 'transformers' and 'accelerate' installed
depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)
print("Model loaded!")

# ── Path configuration ──
# Ensure the path is correct. Using raw strings (r"") prevents escape character errors.
IMAGE_PATH = r"C:\Users\dheer\.antigravity\Occlusion-Aware 3D Scene Reconstruction\Online pfp.jpeg"

# Load images
image = Image.open(r"C:\Users\dheer\.antigravity\Occlusion-Aware 3D Scene Reconstruction\Online pfp.jpeg")
frame = cv2.imread(r"C:\Users\dheer\.antigravity\Occlusion-Aware 3D Scene Reconstruction\Online pfp.jpeg")

if frame is None:
    print(f"Error: Could not load image at {IMAGE_PATH}")
    exit()

print("Running depth estimation...")
result = depth_estimator(image)
depth_map = np.array(result["depth"])

# Normalize depth for visualization (0-255)
depth_display = cv2.normalize(
    depth_map, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

# ── NEW OCCLUSION LOGIC: SOBEL GRADIENT ──
# Instead of Canny edges, we calculate the gradient magnitude of the depth map
sobelx = cv2.Sobel(depth_display, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(depth_display, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

# Threshold the gradient to find significant depth "jumps"
# You can adjust '40' to be more or less sensitive to depth changes
_, binary_occlusion = cv2.threshold(gradient_magnitude, 40, 255, cv2.THRESH_BINARY)
binary_occlusion = binary_occlusion.astype(np.uint8)

# Morphological Operations to clean up and "solidify" the mask
kernel = np.ones((5, 5), np.uint8)
# Closing fills small holes within the mask
occlusion_mask = cv2.morphologyEx(binary_occlusion, cv2.MORPH_CLOSE, kernel)
# Dilation expands the mask to fully cover the boundary overlap
occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=2)

# Apply the occlusion highlight to the frame (Red Color)
frame_with_occlusion = frame.copy()
frame_with_occlusion[occlusion_mask > 0] = [0, 0, 255]

# ── SAVE AND DISPLAY ──
cv2.imwrite("depth_output.jpg", depth_colored)
cv2.imwrite("occlusion_mask.jpg", occlusion_mask)
cv2.imwrite("occlusion_highlighted.jpg", frame_with_occlusion)

# Set a reasonable display size (100x100 was too small to see details)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
depth_resized = cv2.resize(depth_colored, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
occlusion_resized = cv2.resize(frame_with_occlusion, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

cv2.imshow("Original", frame_resized)
cv2.imshow("Depth Map (Magma)", depth_resized)
cv2.imshow("Improved Occlusion Regions", occlusion_resized)

print("✅ Done! Saved improved output images.")
cv2.waitKey(0)
cv2.destroyAllWindows()