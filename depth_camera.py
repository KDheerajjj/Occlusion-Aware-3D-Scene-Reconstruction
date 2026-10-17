import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# ─────────────────────────────────────────
# STEP 1: Connect to your phone camera
# Replace with YOUR phone's IP address
# ─────────────────────────────────────────
PHONE_IP = "10.108.184.27"
PHONE_PORT = "4747"
stream_url = "http://10.108.184.27:4747/video"

print("Loading Depth Anything model... (first time takes 2-3 mins)")

# ─────────────────────────────────────────
# STEP 2: Load Depth Anything model
# Downloads automatically from HuggingFace
# ─────────────────────────────────────────
depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

print("Model loaded! Connecting to phone camera...")

# ─────────────────────────────────────────
# STEP 3: Connect to phone camera stream
# ─────────────────────────────────────────
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("ERROR: Cannot connect to phone camera")
    print(f"Make sure IP Webcam is running at {stream_url}")
    exit()

print("Camera connected! Press Q to quit.")

# ─────────────────────────────────────────
# STEP 4: Process each frame
# ─────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame. Check connection.")
        break

    # Convert frame (OpenCV BGR) to PIL Image (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Run Depth Anything on current frame
    depth_result = depth_estimator(pil_image)
    depth_map = np.array(depth_result["depth"])

    # Normalize depth map for display (0-255)
    depth_display = cv2.normalize(
        depth_map, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Apply color map for better visualization
    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

    # ─────────────────────────────────────
    # STEP 5: Detect occluded regions
    # Areas with sudden depth jumps = occlusions
    # ─────────────────────────────────────
    depth_edges = cv2.Canny(depth_display, 50, 150)
    occlusion_mask = cv2.dilate(depth_edges, None, iterations=3)

    # Highlight occluded areas in red on original frame
    frame_with_occlusion = frame.copy()
    frame_with_occlusion[occlusion_mask > 0] = [0, 0, 255]  # Red

    # ─────────────────────────────────────
    # STEP 6: Show all windows side by side
    # ─────────────────────────────────────
    cv2.imshow("📷 Phone Camera (Original)", frame)
    cv2.imshow("🎨 Depth Map", depth_colored)
    cv2.imshow("🔴 Occlusion Detection", frame_with_occlusion)

    # Save frame when you press S
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("captured_frame.jpg", frame)
        cv2.imwrite("depth_map.jpg", depth_colored)
        cv2.imwrite("occlusion_mask.jpg", occlusion_mask)
        print("✅ Images saved!")

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")