"""
Created on Sat Aug  4 09:13:50 2018

@author: vahid
"""


# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time

# load the image
video_path='G:/car-color-recognition/main_car_color_detection_code/206-2.mp4'
data=[]
capt=cv2.VideoCapture(video_path)
# binarizer
print("[INFO] loading network...")
model = load_model('G:/car-color-recognition/car-color-recognition/mult_label_main/results/weight')
mlb = pickle.loads(open('G:/car-color-recognition/car-color-recognition/mult_label_main/results/lb', "rb").read())
   

while True:
    #start = time.time()
    #print(start)
    ret,frame=capt.read()  
    if not ret:
        break
    else:   
        # Start time
        fps = capt.get(cv2.CAP_PROP_FPS)  
        #print(fps)    
        output = imutils.resize(frame, width=400)
        #end=time.time()
        #sub=end-start
        #print(sub)
        frame=cv2.resize(frame,(96,96))
        frame = frame.astype("float") / 255.0
        #data.append(frame)
        input_data=img_to_array(frame) 
        input_data = np.expand_dims(input_data, axis=0)
        print(input_data.shape)
        
        # labels 
        print("[INFO] classifying image...")
        proba = model.predict(input_data)[0]
        idxs = np.argsort(proba)[::-1][:2]

        # loop over 
        for (i, j) in enumerate(idxs):
            # build the label and draw the label on the image
            label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
            cv2.putText(output, label, (10, (i * 30) + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # show 
        for (label, p) in zip(mlb.classes_, proba):
            print("{}: {:.2f}%".format(label, p * 100))
            # show the output image
        cv2.imshow("Output", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break      
     
capt.release()        
cv2.destroyAllWindows()  



















