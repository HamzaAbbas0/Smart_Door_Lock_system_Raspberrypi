# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    print(i, imagePath)
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    try:
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    except Exception as e:
        print(f"[ERROR] Failed to process image: {imagePath}")
        print(f"Exception: {e}")

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

try:
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data, f)
    print("[INFO] Encodings serialized and saved successfully.")
except Exception as e:
    print(f"[ERROR] Failed to serialize encodings: {e}")
