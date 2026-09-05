# 🕳️ Pothole Detection using YOLOv8

A computer vision project that detects potholes in road videos using a custom-trained **YOLOv8** model. The script processes an input video frame-by-frame, draws bounding boxes around detected potholes in real time, and saves the fully annotated result as a new video file.

## 🎥 Demo

> Add a GIF or screenshot of `pothole_output.mp4` here once available.

## ✨ Features

- Real-time pothole detection using a custom-trained YOLOv8 model
- Live preview window while processing
- Automatically saves annotated output video (`pothole_output.mp4`)
- Simple, well-commented single-script implementation

## 🛠️ Tech Stack

- [Python](https://www.python.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)

## 📁 Project Structure

```
pothole-detection/
├── pothole_detection.py   # Main detection script
├── pothole_model.pt       # Custom-trained YOLOv8 weights
├── road_input.mp4         # Input video to process
└── pothole_output.mp4     # Generated output (created after running)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/your-username/pothole-detection.git
cd pothole-detection
pip install ultralytics opencv-python
```

### Usage

1. Place your trained model weights (`pothole_model.pt`) and input video (`road_input.mp4`) in the project root.
2. Run the script:

```bash
python pothole_detection.py
```

3. A live window will show detections in real time. Press **`q`** to stop early.
4. The annotated video will be saved as `pothole_output.mp4`.

## 🧠 Model

The detection model is a custom-trained YOLOv8 model (`pothole_model.pt`) trained on a pothole dataset. 

> Add details here on your training dataset, number of images, epochs, and mAP/accuracy results if available.

## 🗺️ Roadmap

- [ ] Add support for image input (not just video)
- [ ] Deploy as a web app (e.g. Streamlit/Flask)
- [ ] Add pothole severity/size estimation
- [ ] Integrate GPS logging for mapping detected potholes

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
