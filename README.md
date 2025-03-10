# Face Verification API

A Flask-based REST API that provides face verification capabilities using deep learning models. This system compares facial images against a database of known faces and determines if there's a match.

## Overview

This Face Verification API uses:
- **MTCNN** (Multi-task Cascaded Convolutional Networks) for face detection
- **FaceNet** for generating face embeddings
- **Cosine similarity** for comparing face embeddings

The system verifies whether a face in an image matches any face in the pre-trained database and returns verification results including confidence scores and identity information.

## Features

- Face detection and extraction from images
- Generation of facial embeddings using FaceNet
- Comparison against a database of known facial embeddings
- Simple REST API interface for integration with other systems
- Health check endpoint for monitoring system status

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-verification-api.git
   cd face-verification-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update the `DEFAULT_IMAGE_PATH` in `app.py` to point to your test image:
   ```python
   DEFAULT_IMAGE_PATH = 'path/to/your/test/image.jpg'
   ```

4. Ensure you have the face embeddings file (`faces_embeddings_done_4classes.npz`) in your project directory

## Running the API

### Local Development

```bash
python app.py
```

The API will be available at http://localhost:5000

### Production Deployment (Gunicorn)

```bash
gunicorn app:app
```

## API Endpoints

### Health Check

```
GET /health
```

Returns the status of the API.

**Response Example:**
```json
{
  "status": "ok",
  "message": "Face verification API is running"
}
```

### Face Verification

```
GET /verify
```

Verifies the face in the hardcoded image path against the database of known faces.

**Response Example (Successful Verification):**
```json
{
  "verification_successful": true,
  "confidence": 0.87,
  "identity": "person_name",
  "matching_score": "0.87"
}
```

**Response Example (Failed Verification):**
```json
{
  "verification_successful": false,
  "confidence": 0.32,
  "identity": null,
  "matching_score": "0.32"
}
```

**Response Example (Error):**
```json
{
  "verification_successful": false,
  "error": "No face detected in the image"
}
```

## Deployment to Render

This application is configured for easy deployment to Render.com.

### Steps to Deploy

1. Push your code to a Git repository (GitHub, GitLab, etc.)

2. Set up a Web Service on Render:
   - Sign in to [Render Dashboard](https://dashboard.render.com/)
   - Create a new Web Service and connect to your repository
   - Configure the service:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`
   - Add environment variables if needed

3. Render will automatically deploy your application

### Configuration Files

- **requirements.txt**: Lists all required Python packages
- **Procfile**: Specifies how to run the application on Render

## Customizing the System

### Adjusting the Similarity Threshold

The default similarity threshold is 0.5. You can adjust this in the `FaceVerificationSystem` class to make the system more or less sensitive:

```python
# More strict matching (fewer false positives)
self.similarity_threshold = 0.7

# More lenient matching (fewer false negatives)
self.similarity_threshold = 0.3
```

### Using Your Own Face Database

To use your own face database:
1. Generate embeddings for your face images
2. Save them in the npz format with embeddings in 'arr_0' and names in 'arr_1'
3. Update the `embeddings_path` parameter when initializing `FaceVerificationSystem`

## Troubleshooting

### CUDA Warnings
If you see CUDA-related warnings, these are normal when running on CPU-only environments like Render's standard instances. The system will still function correctly but may be slower than with GPU acceleration.

### No Face Detected
If you receive a "No face detected" error, ensure that:
- The image contains clearly visible faces
- The image file is not corrupted
- The image path is correct
