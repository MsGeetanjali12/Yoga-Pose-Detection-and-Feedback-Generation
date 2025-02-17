
import cv2
import numpy as np
import csv
import mediapipe as mp
cap=cv2.VideoCapture(0)
mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()
class_name = 'cobra'
while True:
    sucess,img = cap.read()
    results = pose.process(img)

    try:
        lmd = results.pose_landmarks.landmark

        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        num_coords = len(results.pose_landmarks.landmark)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (0, 200, 25), cv2.FILLED)
        pose_row = list(np.array([[float(landmark.x), float(landmark.y), float(landmark.z), float(landmark.visibility)] for landmark in lmd]).flatten())
        row = pose_row
        print(row)
        row.insert(0, class_name)
        with open('coords.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)
            print("written")
    except:
        pass

    cv2.imshow("image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
