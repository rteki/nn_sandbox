import cv2
from pynput.mouse import Listener
from pynput.mouse import Button

import sys

# cam = cv2.VideoCapture(0)

# cv2.namedWindow("test")

# img_counter = 0

# while True:
#     ret, frame = cam.read()
#     cv2.imshow("test", frame)
#     if not ret:
#         break
#     k = cv2.waitKey(1)

#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         # SPACE pressed
#         img_name = "opencv_frame_{}.png".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         img_counter += 1

# cam.release()

# cv2.destroyAllWindows()

def on_clicked(x, y, button, pressed):    
    print(x, y)
    print(pressed)



with Listener(
        on_click=on_clicked,
        on_scroll=sys.exit
        ) as listener:
    listener.join()