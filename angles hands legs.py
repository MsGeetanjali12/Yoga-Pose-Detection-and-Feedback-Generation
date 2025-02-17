import cv2
import numpy as np
import mediapipe as mp
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

angles = []

# Read the image from file
img = cv2.imread(r"C:\Users\surya\PycharmProjects\yfg\.venv\Tree-pose-Vrksasana-Ekhart-Yoga.jpg")

# Convert the image to RGB format
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process the pose landmarks in the image
results = pose.process(imgRGB)
try:
    lmd = results.pose_landmarks.landmark
    Rshoulder = [lmd[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x, lmd[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
    Relbow = [lmd[mpPose.PoseLandmark.RIGHT_ELBOW.value].x, lmd[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
    Rwrist = [lmd[mpPose.PoseLandmark.RIGHT_WRIST.value].x, lmd[mpPose.PoseLandmark.RIGHT_WRIST.value].y]
    Rangle = calculate_angle(Rshoulder, Relbow, Rwrist)
    Lshoulder = [lmd[mpPose.PoseLandmark.LEFT_SHOULDER.value].x, lmd[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
    Lelbow = [lmd[mpPose.PoseLandmark.LEFT_ELBOW.value].x, lmd[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
    Lwrist = [lmd[mpPose.PoseLandmark.LEFT_WRIST.value].x, lmd[mpPose.PoseLandmark.LEFT_WRIST.value].y]
    Langle = calculate_angle(Lshoulder, Lelbow, Lwrist)
    RHip = [lmd[mpPose.PoseLandmark.RIGHT_HIP.value].x, lmd[mpPose.PoseLandmark.RIGHT_HIP.value].y]
    RKnee = [lmd[mpPose.PoseLandmark.RIGHT_KNEE.value].x, lmd[mpPose.PoseLandmark.RIGHT_KNEE.value].y]
    RAnkle = [lmd[mpPose.PoseLandmark.RIGHT_ANKLE.value].x, lmd[mpPose.PoseLandmark.RIGHT_ANKLE.value].y]
    Rangle_leg = calculate_angle(RHip, RKnee, RAnkle)
    LHip = [lmd[mpPose.PoseLandmark.LEFT_HIP.value].x, lmd[mpPose.PoseLandmark.LEFT_HIP.value].y]
    LKnee = [lmd[mpPose.PoseLandmark.LEFT_KNEE.value].x, lmd[mpPose.PoseLandmark.LEFT_KNEE.value].y]
    LAnkle = [lmd[mpPose.PoseLandmark.LEFT_ANKLE.value].x,lmd[mpPose.PoseLandmark.LEFT_ANKLE.value].y]
    Langle_leg=calculate_angle(LHip, LKnee, LAnkle)
except:
    pass

cv2.imshow("image",img)
cv2.waitKey(20000) # show the image for 20 seconds
cv2.destroyAllWindows()
lis=[Rangle,Langle,Rangle_leg,Langle_leg]
Tree_numpy=np.array(lis)
np.save('tree_angles',Tree_numpy)
x=np.load('tree_angles.npy')
print(x)
l=[10,20,40,50]
y=np.array(l)
print(x-y)