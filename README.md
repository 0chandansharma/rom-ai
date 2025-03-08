# ROM Assessment Library

A comprehensive library for Range of Motion (ROM) assessment in physiotherapy applications, with a focus on lower back pain assessment and exercise monitoring.

## Features

- **Real-time ROM assessment** using computer vision and pose estimation
- **Accurate joint angle calculations** for various body parts
- **Multiple assessment protocols** for lower back: flexion, extension, lateral flexion, and rotation
- **WebSocket-based real-time feedback** for patient guidance
- **Visualization tools** for enhanced user experience
- **LLM integration** for analysis and personalized recommendations
- **Scalable architecture** designed to handle thousands of movement types
- **Clean API** for easy integration with chatbots and other platforms

## Installation

```bash
pip install rom-assessment
```

Or install from source:

```bash
git clone https://github.com/yourusername/rom-assessment.git
cd rom-assessment
pip install -e .
```

## Requirements

- Python 3.7 or higher
- OpenCV
- MediaPipe
- FastAPI
- NumPy
- HTTPX (for LLM integration)

## Quick Start

### 1. Running the Assessment Server

The ROM assessment library includes a built-in API server that can be used to perform assessments and provide real-time feedback:

```bash
python -m rom.api.main --host 0.0.0.0 --port 8000
```

### 2. Using the ExerciseHandler in Your Code

```python
import cv2
from rom.core.exercise_handler import ExerciseManager

# Initialize the exercise manager
manager = ExerciseManager()

# Create a test instance
test_type = "lower_back_flexion"
test = manager.create_test(test_type)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    processed_frame, rom_data = test.process_frame(frame)

    # Display the processed frame
    cv2.imshow("ROM Assessment", processed_frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

### 3. Integrating with a Chatbot

```python
from rom.core.exercise_handler import ExerciseManager
import cv2
import base64

async def perform_assessment(test_type, frame_data):
    """
    Process a single frame sent from a chatbot client.

    Args:
        test_type: Type of assessment to perform
        frame_data: Base64 encoded image data

    Returns:
        Assessment results and processed frame
    """
    # Create manager and test
    manager = ExerciseManager()

    # Decode base64 image
    image_data = base64.b64decode(frame_data)
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process frame
    processed_frame, rom_data = manager.process_frame(frame, test_type)

    # Encode processed frame
    _, buffer = cv2.imencode(".jpg", processed_frame)
    processed_b64 = base64.b64encode(buffer).decode("utf-8")

    return rom_data, processed_b64
```

## Available Assessment Types

### Lower Back Tests

- `lower_back_flexion`: Forward bending of the lower back
- `lower_back_extension`: Backward bending of the lower back
- `lower_back_lateral_flexion_left`: Side bending to the left
- `lower_back_lateral_flexion_right`: Side bending to the right
- `lower_back_rotation_left`: Rotation to the left
- `lower_back_rotation_right`: Rotation to the right

## Architecture

The ROM Assessment Library uses a modular, extensible architecture:

- **Core Module**: Base classes and interfaces
- **Utils Module**: Pose detection and visualization utilities
- **Tests Module**: Specific test implementations
- **API Module**: FastAPI and WebSocket interfaces
- **Data Module**: Storage and reporting utilities

This architecture allows easy extension with new assessment types while maintaining a consistent interface.

## WebSocket API

The ROM Assessment Library provides a WebSocket API for real-time assessment:

1. Connect to the WebSocket endpoint: `ws://server:port/api/assessment/{test_type}`
2. Send frames as base64-encoded JPEG images
3. Receive processed frames and ROM data in real-time

Example WebSocket response:

```json
{
  "image": "data:image/jpeg;base64,...",
  "rom_data": {
    "test_type": "lower_back_flexion",
    "joint_type": "lower_back",
    "status": "in_progress",
    "current_angle": 45.2,
    "min_angle": 10.5,
    "max_angle": 55.8,
```
