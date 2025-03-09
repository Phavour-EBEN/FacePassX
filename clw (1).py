import cv2 as cv
import os
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_path = "images"

os.listdir(dataset_path)

img = cv.imread("images/Araba Turkson/WhatsApp Image 2025-03-04 at 5.43.25 PM.jpeg")
# opencv BGR channel format and plt reads images as RGB channel format

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img) # RGB

from mtcnn.mtcnn import MTCNN

detector = MTCNN()
results = detector.detect_faces(img)

results

x,y,w,h = results[0]['box']

img = cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 10)
plt.imshow(img)

my_face = img[y:y+h, x:x+w]
#Facenet takes as input 160x160
my_face = cv.resize(my_face, (200,200))
plt.imshow(my_face)

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (200,200)
        self.X = []
        self.Y = []
        self.detector = MTCNN()


    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr


    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')

faceloading = FACELOADING(dataset_path)
X, Y = faceloading.load_classes()

plt.figure(figsize=(16,12))
for num,image in enumerate(X):
    ncols = 3
    nrows = len(Y)//ncols + 1
    plt.subplot(nrows,ncols,num+1)
    plt.imshow(image)
    plt.axis('off')



from keras_facenet import FaceNet
embedder = FaceNet()

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    face_pixels = np.expand_dims(face_pixels, axis=0)
    yhat = embedder.embeddings(face_pixels)
    return yhat[0]

EMBEDDED_X = []

for img in X:
    EMBEDDED_X.append(get_embedding(img))

EMBEDDED_X = np.asarray(EMBEDDED_X)

np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)

Y

EMBEDDED_X

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your saved embeddings
data = np.load('faces_embeddings_done_4classes.npz')
EMBEDDED_X, Y = data['arr_0'], data['arr_1']

# Encode the labels
encoder = LabelEncoder()
encoder.fit(Y)
Y_encoded = encoder.transform(Y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(EMBEDDED_X, Y_encoded, test_size=0.2, random_state=42)

# Train a classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Test the classifier
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model and encoder
import pickle
with open('face_recognition_model.pkl', 'wb') as file:
    pickle.dump((model, encoder), file)

def verify_face(image_path, model, encoder, detector, embedder, threshold=0.7):
    # Read image
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Detect face
    try:
        results = detector.detect_faces(img)
        if not results:
            return "No face detected"

        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face = cv.resize(face, (200, 200))

        # Get embedding
        face_embedding = get_embedding(face)

        # Predict
        samples = np.expand_dims(face_embedding, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        # Get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_name = encoder.inverse_transform([class_index])[0]

        if class_probability < threshold * 100:
            return "Unknown"

        return predict_name, class_probability

    except Exception as e:
        return f"Error: {str(e)}"

def real_time_recognition(model, encoder, detector, embedder):
    cap = cv.VideoCapture(0)  # This would be replaced by the mobile app camera feed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB (FaceNet expects RGB)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Detect faces
        results = detector.detect_faces(rgb_frame)

        for result in results:
            x, y, w, h = result['box']
            x, y = abs(x), abs(y)

            # Draw rectangle
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract and resize face
            face = rgb_frame[y:y+h, x:x+w]
            face = cv.resize(face, (200, 200))

            # Get embedding
            face_embedding = get_embedding(face)

            # Predict
            samples = np.expand_dims(face_embedding, axis=0)
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)

            # Get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100
            predict_name = encoder.inverse_transform([class_index])[0]

            # Display name and probability
            text = f"{predict_name}: {class_probability:.2f}%"
            cv.putText(frame, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv.imshow('Face Recognition', frame)

        # Break loop on 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
