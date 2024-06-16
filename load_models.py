import requests
import json
import base64
import face_recognition
import cv2
import numpy as np

url = "http://devopn.ru:8000"
req = requests.get(f"{url}/model")
print(req.json())
for i in req.json():
    name = i['name']
    id = i['id']
    b64image = i['image_path']
    nparr = np.frombuffer(base64.b64decode(b64image), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # with open(f"samples/{name}_{id}", "wb") as file:
    #     file.write(base64.b64decode(b64image))
    encodings = face_recognition.face_encodings(image)
    if len(encodings) != 1:
        continue
    encodings = encodings[0]
    np.savetxt(".load_swap", encodings)
    with open(f"models/{name}", "a") as file:
        en = open(".load_swap", "r") 
        file.write(en.read())
        en.close()
        
