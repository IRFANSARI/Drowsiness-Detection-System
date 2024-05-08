import cv2
import dlib
from imutils import face_utils
from imp_functions import *

# Initialize variables
sleep = 0
yawn = 0
active = 0
status = ""
eyeColor = (255, 255, 255)
mouthColor = (255, 255, 255)

# Initialize camera, detector and predictor
print("[INFO] Loading Camera....")
camera = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
print("[INFO] Loading Predictor....")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Main loop
while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        left_blink = isBlinked(landmarks[36], landmarks[37], 
                             landmarks[38], landmarks[41], 
                             landmarks[40], landmarks[39])
        right_blink = isBlinked(landmarks[42], landmarks[43], 
                              landmarks[44], landmarks[47], 
                              landmarks[46], landmarks[45])
        mouth = isYawned(landmarks[60], landmarks[61],
                        landmarks[62], landmarks[63],
                        landmarks[64], landmarks[65],
                        landmarks[66], landmarks[67])

        # Analyze eye blinks
        if mouth == True:
            sleep = 0
            yawn += 1
            active = 0
            if yawn > 4:
                mouthColor = (0, 0, 255)
                eyeColor = (0, 0, 255)
                yawn = 0
                sleep = 0
                active = 0
                status = "Yawning"

        elif left_blink == True or right_blink == True:
            sleep += 1
            yawn = 0
            active = 0
            if sleep > 4:
                mouthColor = (255, 255, 255)
                eyeColor = (0, 0, 255)
                yawn = 0
                sleep = 0
                active = 0
                status = "Sleeping"
        else:
            yawn = 0
            sleep = 0
            active += 1
            if active > 4:
                mouthColor = (255, 255, 255)
                eyeColor = (255, 255, 255)
                yawn = 0
                sleep = 0
                active = 0
                status = "Active"
                
                
        cv2.putText(frame, "Status: " + status, (200, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.85, eyeColor, 2)


        #To show mouth landmarks
        i = 60
        while i < 68:
            (x, y) = landmarks[i]
            cv2.circle(frame, (x, y), 1, mouthColor, -1)
            i += 1


        # To show eyes' landmarks
        i = 36
        while i < 48:
            (x, y) = landmarks[i]
            cv2.circle(frame, (x, y), 1, eyeColor, -1)
            i += 1


        # To display all the landmarks on the face
        # for n in range(0, 68):
        #     (x, y) = landmarks[n]
        #     cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Drowsiness Detection System", frame)
    # cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
