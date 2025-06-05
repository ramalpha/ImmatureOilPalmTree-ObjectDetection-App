# DeepForest Object Detection Model for Immature Oil Palm Tree
App by Ramadya Alif Satya

[![Launch Demo App](https://img.shields.io/badge/Demo_App-Open-brightgreen?style=for-the-badge&logo=streamlit&logoColor=white)](https://ramalpha-tbmdetection.streamlit.app/)

## About

This application allows for the multi-class detection of immature oil palm trees in images using a custom-trained DeepForest model. It provides a user-friendly interface to upload images and visualize the detection results.

tbm-1 (0-12 months)
tbm-2 (13-24 months)

## Features

- Detect objects (Immature Oil Palms) in user-uploaded images.
- Supports various image formats (JPG, JPEG, PNG, TIF, TIFF).
- Displays detection bounding boxes, class labels, and confidence scores.
- Allows download of the visualized image and a CSV summary of detections.
- Uses a demo image if no image is uploaded.

*(Please adjust this Features list if your app has more capabilities, like video or live detection, or other specific functionalities we haven't focused on recently).*

## Tech Stack

- Python
- Streamlit
- DeepForest
- PyTorch
- OpenCV
- Pillow
- RasterIO
- Pandas
- NumPy

## Run Locally

Clone the project

```bash
  git clone https://github.com/ramalpha/ImmatureOilPalmTree-ObjectDetection-App
```

Go to the project directory

```bash
  cd ImmatureOilPalmTree-ObjectDetection-App
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run tbm_od_app.py
```

## Author

- [Ramadya Alif Satya](https://github.com/ramalpha)