import cv2

for i in range(5):  # 0-ден 4-ке дейін сынап көрейік
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Камера {i} жұмыс істейді")
        cap.release()
    else:
        print(f"Камера {i} табылмады")