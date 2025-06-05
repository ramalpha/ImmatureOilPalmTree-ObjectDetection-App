# DeepForest Object Detection Model for Immature Oil Palm Tree

## About

This application allows for the multi-class detection of immature oil palm trees in images using a custom-trained model built with the [DeepForest](https://github.com/weecology/DeepForest) library. It provides a user-friendly interface to upload images and visualize the detection results.

The model is trained to identify the following classes:
- tbm-1 (0-12 months)
- tbm-2 (13-24 months)

[![Launch Demo App](https://img.shields.io/badge/Demo_App-Open-brightgreen?style=for-the-badge&logo=streamlit&logoColor=white)](https://ramalpha-tbmdetection.streamlit.app/)

## Features

- Detect objects (Immature Oil Palms) in user-uploaded images.
- Supports various image formats (JPG, JPEG, PNG, TIF, TIFF).
- Displays detection bounding boxes, class labels, and confidence scores.
- Allows download of the visualized image and a CSV summary of detections.
- Uses a demo image if no image is uploaded.

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