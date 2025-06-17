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
   pip install -r requirements.txt

  ## ⚙️ Configuration






