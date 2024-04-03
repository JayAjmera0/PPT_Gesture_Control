import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

def is_fist_detected(handPoints):
    """
    Determines if the hand configuration represents a fist.
    
    Parameters:
    handPoints (list of tuples): Coordinates of hand landmarks.
    
    Returns:
    bool: True if a fist is detected, False otherwise.
    """
    # Getting base points of the fingers and the palm
    palm_base = handPoints[0]
    finger_bases = [handPoints[i] for i in [5, 9, 13, 17]]  # Base of each finger
    finger_tips = [handPoints[i] for i in [4, 8, 12, 16, 20]]  # Tips of the fingers
    
    # Check if fingertips are close to the base of any finger, indicating a closed hand
    for tip in finger_tips:
        if all((tip[0] - base[0])**2 + (tip[1] - base[1])**2 > 30**2 for base in finger_bases):
            return False  # A fingertip is too far from the bases, not a fist
    
    # Additional checks can be added here based on the specific requirements
    return True


# Flags
writing_mode = False
left = False
right = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks

    handPoints = []
    if multiLandMarks:
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        if not writing_mode:
            # Fist gesture to activate writing mode
            if is_fist_detected(handPoints):
                pyautogui.hotkey('ctrl', 'p')
                writing_mode = True
                print("Writing mode on")
                pyautogui.sleep(1)  # Prevent immediate re-toggle

            # Left and right functionality
            else:
            
                if not left and handPoints[8][0] < 200:  # Adjust this threshold for left gesture
                    left = True
                    pyautogui.press('left')  # Simulate left arrow key press to move to the previous slide
                    print("Gesture Detected: Move Left")

                if not right and handPoints[8][0] > 440:  # Adjust this threshold for right gesture
                    right = True
                    pyautogui.press('right')  # Simulate right arrow key press to move to the next slide
                    print("Gesture Detected: Move Right")

                if 200 <= handPoints[8][0] <= 440:
                    right = False
                    left = False


        else:
            # Check for fist to deactivate writing mode
            if is_fist_detected(handPoints) and writing_mode:
                pyautogui.press('esc')
                writing_mode = False
                print("Writing mode off")
                pyautogui.sleep(1)  # Prevent immediate re-toggle
                continue  # Skip to next frame to avoid accidental actions

            # Check for index finger up for mouse click
            if handPoints[8][1] < handPoints[6][1]:  # Index finger is up
                pyautogui.mouseDown()
                print("Mouse Down")
            else:
                pyautogui.mouseUp()
                print("Mouse Up")

            # Move mouse pointer following the index finger
            pyautogui.moveTo(handPoints[8][0], handPoints[8][1])

    cv2.imshow("Video", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
