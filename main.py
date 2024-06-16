from picamera2 import Picamera2
import cv2
import numpy as np
from libcamera import Transform
import face_recognition
import os
import requests
from datetime import datetime, timedelta
import base64
from collections import defaultdict

# Train models with live data. Not optimized. Sometmes get bad data.
isAfterTraining = False
# Creating new models on umrecognized faces
newModeling = True
url = "http://devopn.ru:8000"
sendTime = timedelta(seconds=10)



picam2 = Picamera2()
# picam2.sensor_resolution = (320, 240)
preview_config = picam2.create_preview_configuration(transform=Transform(vflip=True))
picam2.start(preview_config)


# Load models and name mappings from files
saved_faces = []
mapping = []
np.reshape
for i in os.listdir("models"):
    data = np.loadtxt(f"models/{i}")
    data = data.reshape((-1, 128))
    for j in data:
        saved_faces.append(j)
        mapping.append(i)


lastwrite = datetime.now()
lastPhoto = datetime.now()
lastSend = datetime.now()
counter = 1

while True:
    array = picam2.capture_array()
    image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    # 
    new_faces = face_recognition.face_encodings(image)

    '''
    Green rectangles
    reduce perfomance from 0.8s to 1.6s
    '''
    # faces_locations = face_recognition.face_locations(image)
    # for i in faces_locations:
    #     start = (i[3], i[2])
    #     end = (i[1], i[0])
    #     image = cv2.rectangle(image, start, end, (55,255,55), 2)

    all_names = defaultdict(int) # Names on the picture

    for face_encode in new_faces:
        result = face_recognition.compare_faces(saved_faces, face_encode)
        answer = set()
        for k in range(len(result)):
            if result[k]:
                filename = mapping[k]
                answer.add(filename)
                all_names[str(filename)] += 1

                # If we want to train model on the live data
                if isAfterTraining:

                    if os.path.getsize(f"models/{filename}") < 102400 * 3:
                        np.savetxt(".train_swap", face_encode)
                        file = open(f"models/{filename}", "a")
                        swap = open(".train_swap")
                        for i in swap:
                            file.write(i)
                        file.close()
                        swap.close()

                if filename.startswith("anon") and (datetime.now() - lastwrite >= timedelta(minutes=1)):
                    cv2.imwrite(f"guests/anon_{counter}_{datetime.now().strftime('%d.%m.%Y_%H:%M')}.jpg", image)
                            

        
        
        # If we dont recognize face we create new model with this face
        if not answer and newModeling:
            all_names[f"anon_{counter}"] += 1
        
            # Save photo of guest
            cv2.imwrite(f"guests/anon_{counter}_{datetime.now().strftime('%d.%m.%Y_%H:%M')}.jpg", image)
            np.savetxt(f"models/anon_{counter}", face_encode)
            saved_faces.append(face_encode)
            mapping.append(f"anon_{counter}")
            counter += 1

    print(dict(all_names), datetime.now() - lastPhoto, "sec")
    lastPhoto = datetime.now()

    if all_names and (datetime.now() - lastSend >= sendTime):
        lastSend = datetime.now()

        faces_locations = face_recognition.face_locations(image)
        for i in faces_locations:
            start = (i[3], i[2])
            end = (i[1], i[0])
            image = cv2.rectangle(image, start, end, (55,255,55), 2)
        try:
            photo64 = base64.b64encode(cv2.imencode(".jpg", image)[1]).decode()
            req = requests.post(url+ "/meet/new", json={"names": dict(all_names), "photo":photo64})
            if req.status_code != 200:
                requests.post(url+ "/meet/new", json={"names": dict(all_names), "photo":None})
        except Exception as er:
            print(er)

    
    # cv2.imshow(f"guard", image)

    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

# cv2.waitKey(0)