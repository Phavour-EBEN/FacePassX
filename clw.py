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