import cv2
import mediapipe as mp
import pyautogui

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

right = False
left = False
writing_mode = False  # Flag to indicate writing mode

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

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

        if not left and handPoints[8][0] < 200:
            left = True
            pyautogui.press('left')
            print("Gesture Detected: Move Left")

        if not right and handPoints[8][0] > 440:
            right = True
            pyautogui.press('right')
            print("Gesture Detected: Move Right")

        if 200 <= handPoints[8][0] <= 440:
            left = False
            right = False

        # Check if all five fingers are up
        if len(handPoints) == 21:
            all_fingers_up = True
            for i in range(1, 5):
                if handPoints[i][1] > handPoints[i + 5][1]:
                    all_fingers_up = False
                    break

            # If all fingers are up, toggle writing mode with Escape key
            if all_fingers_up:
                if writing_mode:
                    pyautogui.press('esc')
                    writing_mode = False
                    print("Writing mode off")
            else:
                # Turn on writing mode and simulate holding click with index finger
                writing_mode = True
                pyautogui.hotkey('ctrl', 'p')
                print("Writing mode on")
                if handPoints[8][1] < handPoints[6][1] and handPoints[12][1] < handPoints[10][1]:
                    pyautogui.mouseDown(handPoints[8][0], handPoints[8][1])
                    print("Index finger up: Simulate Click")
                else:
                    pyautogui.mouseUp(handPoints[8][0], handPoints[8][1])

        # Release pen when not all fingers are up
        elif len(handPoints) != 21:
            pyautogui.hotkey('ctrl', 'p')
            print("Release Pen")
            if writing_mode:
                pyautogui.press('esc')
                writing_mode = False
                print("Writing mode off")

    cv2.imshow("Video", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
