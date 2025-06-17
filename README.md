# Intelligent-Pedestrian-Flow-Monitoring-MOT-with-Re-ID
An object-oriented platform integrating YOLO, BotSort, and Re-ID for robust real-time pedestrian detection and tracking. It generates heatmaps, performance metrics, and trajectory analyses for comprehensive flow insights.


---

## 📑 Table of Contents

1. [Features](#features)  
2. [Tech Stack](#tech-stack)  
3. [Dataset](#dataset)  
4. [Prerequisites](#prerequisites)  
5. [Installation](#installation)  
6. [Configuration](#configuration)  
7. [Usage](#usage)  
8. [Outputs](#outputs)  
9. [Project Structure](#project-structure)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## 🚀 Features

- **Real-Time Pedestrian Detection**: YOLO-based inference optimized for pedestrian class.  
- **Multi-Object Tracking (MOT)**: BotSort integration for reliable identity assignment across frames.  
- **Re-Identification (Re-ID)**: Consistent ID management for reappearing pedestrians.  
- **Analytical Insights**:  
  - Trajectory history and speed-based status (active/passive)  
  - Heatmap generation to visualize movement density  
  - Frame-by-frame and rolling-average FPS metrics  
- **Export Capabilities**:  
  - Last processed frame snapshot  
  - FPS trend chart (PNG)  
  - Heatmap and heatmap overlay images  

---

## 🧰 Tech Stack

- **Python** 3.8+  
- **Ultralytics YOLO** for object detection  
- **boxmot** (BotSort) for tracking  
- **PyTorch** as the deep learning backend  
- **OpenCV** for video I/O and image processing  
- **Matplotlib** for plotting analytics  
- **NumPy**, **collections**, **Pathlib**, **typing** (standard library utilities)  

---

## 🎥 Dataset

Sample test videos are hosted on Google Drive:

> [Video Dataset (Google Drive)](https://drive.google.com/drive/folders/1HREd4u_iMUsA87WBDn9zHzfuMywZDokO)

Download your preferred files and set the `video_path` in the configuration accordingly.

---

## ⚙️ Prerequisites

- Python 3.8 or newer  
- Virtual environment (recommended)  

---

## 🛠️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<YourUsername>/intelligent-pedestrian-mot-reid.git
   cd intelligent-pedestrian-mot-reid

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   #macOS/Linux
   source venv/bin/activate
   #Windows
   venv\Scripts\activate

3. **Install dependencies**
    ```bash
   pip install -r requirments.txt
  
## ⚙️ Configuration
Modify runtime parameters in the CONFIG dictionary at the top of MOT_with_trajektori.pyy:
```python
CONFIG = {
"video_path": "path/to/video.mp4",
"output_dir": "results/",
"yolo_weights": "assets/YoloWeights/yolo11x.pt",
"reid_weights": "assets/resnet50_fc512_msmt17.pt",
"use_mask": False,
"mask_path": "",
"draw_tracks": True,
"conf_threshold": 0.35,
"iou_threshold": 0.45,
"inference_size": 1600,
"target_resolution": (1280, 720),
"speed_threshold": 15.0,
"frame_window": 150,
"heatmap_accumulation_rate": 30.0,
}
```
Adjust these values to match your environment and use case.

## ▶️ Usage
Run the application and follow interactive prompts:
```python
python MOT_with_trajektori.py
```
Select tak or nie when prompted for ROI masking and trajectory drawing.
Press q to exit the live preview and terminate the program.


## 📂 Outputs
Upon completion, the specified output_dir will contain:

- **last_frame_oop.jpg** — Annotated snapshot of the final frame.
- **fps_plot_oop.png** — Chart of FPS performance over time.
- **heatmap_oop.jpg** — Raw heatmap visualization.
- **heatmap_overlay_oop.jpg** — Heatmap overlay on the last frame.

## 🗂️ Project Structure

```python
intelligent-pedestrian-mot-reid/
├── MOT_with_trajektori.py # Entry point with CONFIG and processing logic
├── requirements.txt       # List of Python dependencies
├── README.md              # Project documentation (this file)
├── .gitignore             # Git ignore patterns
└── assets/                # External assets (excluded from Git)
    ├── YoloWeights/       # YOLO model weight files
    │   └── yolo11x.pt
    └── resnet50_fc512_msmt17.pt  # Re-ID model weights
```

## 🤝 Contributing 

Contributions are welcome! Please open issues and submit pull requests for improvements or bug fixes.









