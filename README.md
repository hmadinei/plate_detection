# Persian Cars & Motorcycles Plate Detection
This repository contains the pipeline for detecting the plates of cars and motorcycles.
The project is divided into two primary stages: 
1. Plate Detection: Identifying the bounding box of the plate within the image.
2. Optical Character Recognition (OCR): Extracting the Persian text characters from the detected plate.  

## 🛠️ Build With 
- Framework: FastAPI (Python 3.12)
- OCR Detection Model: YOLOv8 - Used for both detection and character recognition
- Containerization: Docker & Docker Compose

## 🚀 Quick Start
This project is fully containerized using Docker. You don't need to install any local dependencies to get started.

Run the following command to build and start the containers in detached mode:

`docker-compose up -d --build`

## 📡 API Reference

The API is served via FastAPI. Once the Docker container is running, you can access the Swagger documentation at `http://localhost:8000/docs`.

### Detect Plate from Video
`POST /detect_video`

**Request:**
The endpoint accepts video of cars and motorcycles in crowded environments.

**Response (JSON):**
```json
{
    "label": "car",
    "id": 2,
    "most_common_prediction": "‫52ط72799‬",
    "unknown_num": 0,
    "confidence": 1,
    "start_time_sec": 0.1,
    "end_time_sec": 0.2
  }
}
```

**Note**: In addition to the JSON response, the processed video (with bounding boxes and labels) is automatically saved to your local environment. You can find the output file in the `./outputs` folder within your project directory.