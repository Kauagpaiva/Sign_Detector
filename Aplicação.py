import pickle 
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd


with open('signs.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

column_names = list()
for val in range(1, 22):
    column_names += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Right hand
        if cv2.waitKey(10) & 0xFF == 27: #esc
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )
        
        try:
            # Extract right hand landmarks
            rightHandLandmarks = results.right_hand_landmarks.landmark
            row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in rightHandLandmarks]).flatten())
            
            # Make Detections
            X = pd.DataFrame([row], columns=column_names)
            signs_class = model.predict(X)[0]
            signs_prob = model.predict_proba(X)[0]

            if signs_prob[np.argmax(signs_prob)].item() <= float(0.7): 
                signs_class = "Unknown"

            # Grab right wrist coords
            coords = tuple(np.multiply(np.array(
                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x, 
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y)), 
                 [640, 480]).astype(int))

            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(signs_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, signs_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, signs_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(signs_prob[np.argmax(signs_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
                        
        cv2.imshow('Sign Detector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()