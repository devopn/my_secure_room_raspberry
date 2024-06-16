from picamera2 import Picamera2
import cv2
import numpy as np
from libcamera import Transform
import dlib
import face_recognition
import numpy as np              


picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(transform=Transform(vflip=True))

picam2.start(preview_config)

name = input()
cache = np.array([])
while True:
    array = picam2.capture_array()
    image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    cv2.imshow(f"color", image)
    new_faces = face_recognition.face_encodings(image)

    # faces_locations = face_recognition.face_locations(image)
    # for i in faces_locations:
    #     start = (i[3], i[2])
    #     end = (i[1], i[0])
    #     image = cv2.rectangle(image, start, end, (55,255,55), 2)

    if len(new_faces) > 1:
        print("You must be alone")
        exit()

    if new_faces:
        cache = np.append(cache, new_faces[0])
    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

np.savetxt(f"models/{name}", cache)