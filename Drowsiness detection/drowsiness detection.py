import cv2
import os
from keras.models import load_model
from twilio.rest import Client
import numpy as np
from pygame import mixer
import time
import sys


mixer.init()
sound = mixer.Sound('alarm.wav')
car = mixer.Sound('car.wav')
root_path = os.getcwd()
face = cv2.CascadeClassifier(os.path.join(root_path,'haar cascade files', 'haarcascade_frontalface_alt.xml'))
leye = cv2.CascadeClassifier(os.path.join(root_path, 'haar cascade files', 'haarcascade_lefteye_2splits.xml'))
reye = cv2.CascadeClassifier(os.path.join(root_path, 'haar cascade files', 'haarcascade_righteye_2splits.xml'))

#client = Client('ACc6fc575fedbb1b800b0833ce15ba456e', 'ae60557a84e5243ebd9cdeede49691bf')

lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
asleep_prev_loop = False
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
no_eyes = 0

sys.stdout.flush()
print("Welcome. Push to start car.")
input()
car.play()

sound_played = False
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    """
    if (right_eye == ()):
        print("There does not appear to be a right eye in this frame. ")
    else:
        print("There appears to be a right eye in this frame. ")

    if (left_eye == ()):
        print("There does not appear to be a left eye in this frame. ")
    else:
        print("There appears to be a left eye in this frame. ")
    """

    if (left_eye == () and right_eye == ()):
        cv2.putText(frame, 'I DONT SEE ANY EYES: ' + str(no_eyes), (50, 50), font, 1, (0, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        no_eyes += 1
        if (no_eyes > 45 and no_eyes < 90):
            score = 20
        elif(no_eyes >= 90):
            score = 35

    else:
        no_eyes = 0

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye), axis = 1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0 and (left_eye != () or right_eye != ())):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
    if(score<0):
        score=0
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score < 15 and sound_played):
        sound_played = False
    if(score>=15):
        try:
            if (not sound_played):
                sound.play()
                sound_played = True
        except:
            pass

        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    if (score > 30):
        if (not asleep_prev_loop):
            asleep_prev_loop = True
            #client.messages.create(to='+17347907349', from_='+18782177085', body='Your son Frankie has fallen asleep.\n\nSafe landing the vehicle.')
        cv2.putText(frame,'INITIATING SAFE LAND NOW. TEXTING YOUR MOM',(10,height-100), font, 1,(0,0,0),1,cv2.LINE_AA)
    if (score < 15 and asleep_prev_loop):
        asleep_prev_loop = False

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
