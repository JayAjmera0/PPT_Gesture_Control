import cv2
import mediapipe as mp
import os

# getting input from webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# setting up mediapipe hand detector
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# test images
backgroundIslands = []
for img in os.listdir("images"):
    backgroundIslands.append(cv2.imread(f"images/{img}"))

imgIndex = 0
right = False
left = False
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip webcam video input
    background = cv2.resize(backgroundIslands[imgIndex], (640, 480))
    cv2.imshow("Background", background)
    cv2.imshow("Video", img)
    cv2.waitKey(1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        maxIndex = len(backgroundIslands) - 1

        # Adjusted gesture detection thresholds
        if not left and handPoints[8][0] < 200:  # Adjust this threshold for left gesture
            left = True
            if imgIndex > 0:
                imgIndex -= 1
            else:
                imgIndex = maxIndex
            print("Gesture Detected: Move Left")

        if not right and handPoints[8][0] > 440:  # Adjust this threshold for right gesture
            right = True
            if imgIndex < maxIndex:
                imgIndex += 1
            else:
                imgIndex = 0
            print("Gesture Detected: Move Right")

        if 200 <= handPoints[8][0] <= 440:  # Reset gesture detection
            left = False
            right = False

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
