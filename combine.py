import cv2
import mediapipe as mp
import pyautogui  # Import pyautogui for controlling the slides

# Setup MediaPipe hands detector
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize webcam input
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set webcam width
cap.set(4, 480)  # Set webcam height

# Flags to control slide movement
right = False
left = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image to make it a mirror view

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # Draw the hand landmarks
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        # Check if the index finger tip is in the left or right region of the screen
        if not left and handPoints[8][0] < 200:  # Adjust this threshold for left gesture
            left = True
            pyautogui.press('left')  # Simulate left arrow key press to move to the previous slide
            print("Gesture Detected: Move Left")

        if not right and handPoints[8][0] > 440:  # Adjust this threshold for right gesture
            right = True
            pyautogui.press('right')  # Simulate right arrow key press to move to the next slide
            print("Gesture Detected: Move Right")

        # Reset gesture detection if the finger is in the middle region of the screen
        if 200 <= handPoints[8][0] <= 440:
            left = False
            right = False

    cv2.imshow("Video", img)  # Show the webcam image with landmarks drawn

    # Break the loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
