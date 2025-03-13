import os
import numpy as np
import cv2 as cv
from flask import Flask, jsonify
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from flask_cors import CORS
import requests
import json
from dotenv import load_dotenv
from supabase import create_client
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Load environment variables
load_dotenv()
# Get Supabase credentials from .env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)

## Define the bucket and folder path
bucket_name = "images"
folder_path = "public/"

# Fetch all files in the bucket
response = supabase.storage.from_(bucket_name).list()

# Fetch all files inside the "public" folder
response = supabase.storage.from_(bucket_name).list(path=folder_path)

if response and isinstance(response, list):
    # Sort by 'created_at' in descending order (newest first)
    latest_file = sorted(response, key=lambda x: x["created_at"], reverse=True)[0]

    # Get the public URL of the newest image
    latest_image_url = supabase.storage.from_(bucket_name).get_public_url(f"public/{latest_file['name']}")


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration - hardcoded image path
DEFAULT_IMAGE_PATH = latest_image_url  # Replace with your actual image path
# DEFAULT_IMAGE_PATH = str(image_url)  # Replace with your actual image path

# Load embeddings and detector/embedder models
class FaceVerificationSystem:
    def __init__(self, embeddings_path='faces_embeddings_done_4classes.npz'):
        # Load the face detector and embedder
        self.detector = MTCNN()
        self.embedder = FaceNet()
        
        # Load pre-trained embeddings
        data = np.load(embeddings_path)
        self.known_embeddings = data['arr_0']
        self.known_names = data['arr_1']
        
        # Threshold for face matching (can be adjusted based on requirements)
        self.similarity_threshold = 0.5
        print(f"Loaded {len(self.known_names)} faces for verification")
        
    def extract_face(self, image):
        """Extract face from an image"""
        # Convert to RGB for MTCNN
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.detector.detect_faces(rgb_image)
        
        if not results:
            raise Exception("No face detected in the image")
        
        # Extract the bounding box
        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)
        
        # Extract face region
        face = rgb_image[y:y+h, x:x+w]
        
        # Resize to required size
        face_resized = cv.resize(face, (200, 200))
        
        return face_resized
    
    def get_embedding(self, face_pixels):
        """Get embedding from face pixels"""
        face_pixels = face_pixels.astype('float32')
        face_pixels = np.expand_dims(face_pixels, axis=0)
        embedding = self.embedder.embeddings(face_pixels)
        return embedding[0]
    
    def verify_face(self, image):
        """Verify a face against the known embeddings"""
        try:
            # Extract face
            face = self.extract_face(image)
            
            # Get embedding
            embedding = self.get_embedding(face)
            
            # Calculate similarity with all known embeddings
            similarities = []
            for known_embedding in self.known_embeddings:
                similarity = 1 - cosine(embedding, known_embedding)
                similarities.append(similarity)
            
            # Find the best match
            best_match_index = np.argmax(similarities)
            best_match_score = similarities[best_match_index]
            
            # Check if the similarity is above threshold
            verification_successful = best_match_score > self.similarity_threshold
            
            # Ensure all values are JSON serializable
            result = {
                'verification_successful': bool(verification_successful),  # Explicitly convert to Python bool
                'confidence': float(best_match_score),  # Ensure it's a Python float
                'identity': str(self.known_names[best_match_index]) if verification_successful else None,
                'matching_score': str(round(best_match_score, 2))  # Convert to string explicitly
            }
            
            return result
            
        except Exception as e:
            return {
                'verification_successful': False,
                'error': str(e)
            }

# Initialize the verification system
verifier = FaceVerificationSystem()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Face verification API is running'
    })

@app.route('/verify', methods=['GET'])
def verify_face():
    """Face verification endpoint using hardcoded image path"""
    try:
        # Check if file exists
        if not os.path.exists(DEFAULT_IMAGE_PATH):
            return jsonify({
                'verification_successful': False,
                'error': f'Image file not found: {DEFAULT_IMAGE_PATH}'
            })
            
        # Read image from file
        image = cv.imread(DEFAULT_IMAGE_PATH)
        
        if image is None:
            return jsonify({
                'verification_successful': False,
                'error': f'Failed to read image from: {DEFAULT_IMAGE_PATH}'
            })
        
        # Verify face
        result = verifier.verify_face(image)
        
        # Ensure the entire result is JSON serializable
        for key in result:
            if isinstance(result[key], np.bool_):
                result[key] = bool(result[key])
            elif isinstance(result[key], np.floating):
                result[key] = float(result[key])
            elif isinstance(result[key], np.integer):
                result[key] = int(result[key])
                
        return jsonify(result)
    
    except Exception as e:
        error_msg = str(e)
        return jsonify({
            'verification_successful': False,
            'error': error_msg
        })

# At the end of your app.py file, change:
if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
   