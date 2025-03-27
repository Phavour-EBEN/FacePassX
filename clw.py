import os
import numpy as np
import cv2 as cv
from flask import Flask, jsonify
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration - hardcoded image path
DEFAULT_IMAGE_PATH = "D:\Telegram\photo_8_2025-03-27_02-07-40.jpg"  # Replace with your actual image path

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
        self.similarity_threshold = 0.8
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
            
            result = {
                'verification_successful': verification_successful,
                'confidence': float(best_match_score),
                'identity': str(self.known_names[best_match_index]) if verification_successful else None,
                'matching_score': f"{best_match_score:.2f}"
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
            }), 404
            
        # Read image from file
        image = cv.imread(DEFAULT_IMAGE_PATH)
        
        if image is None:
            return jsonify({
                'verification_successful': False,
                'error': f'Failed to read image from: {DEFAULT_IMAGE_PATH}'
            }), 400
            
        # Verify face
        result = verifier.verify_face(image)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'verification_successful': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)