# Occlusion-Aware 3D Scene Reconstruction

A lightweight Python toolkit that captures depth information from a single RGB camera, detects occlusions, and visualises the resulting 3‑D scene.

## Repository Layout
```
├── depth_camera.py          # Captures RGB image → depth map
├── photo_depth.py           # Processes depth, builds occlusion mask, visualises results
├── .gitignore               # Standard Python ignore patterns
├── *.jpeg / *.jpg           # Sample images & output visualisations
└── README.md                # This file
```

## Quick Start
```bash
# Clone the repo (if not already cloned)
git clone https://github.com/KDheerajjj/Occlusion-Aware-3D-Scene-Reconstruction.git
cd Occlusion-Aware-3D-Scene-Reconstruction

# Install dependencies
pip install -r requirements.txt   # or manually: torch torchvision transformers opencv-python open3d pillow numpy matplotlib

# Run the demo scripts
python depth_camera.py
python photo_depth.py
```

## Dependencies
- **torch**, **torchvision**, **transformers** – for potential deep‑learning extensions.
- **opencv‑python** – image capture & processing.
- **open3d** – 3‑D point‑cloud handling (optional).
- **pillow**, **numpy**, **matplotlib** – basic image I/O and visualisation.

## License
MIT License – feel free to adapt and build upon this work.
