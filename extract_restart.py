import cv2
import numpy as np

# Load the debug image
img = cv2.imread('debug_view.png')
if img is None:
    print("Error: debug_view.png not found!")
    exit()

# From the image we can see the restart button is in the center-bottom area
# Coordinates from debug_view.png
# Let's try to detect it automatically or crop it
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# The button has a specific shape
# Let's just crop it based on the image size
# debug_view.png is 606x102 (from the code's WIDTH/HEIGHT)
# No, debug_view.png is 2729 bytes... wait.
# I'll use the findContours method to find the button
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # The restart button is roughly square-ish, around 30-50 pixels
    if 30 < w < 60 and 30 < h < 60:
        restart_btn = gray[y:y+h, x:x+w]
        cv2.imwrite('restart_button.png', restart_btn)
        print(f"Restart button saved from {x, y, w, h}")
        break
else:
    # Fallback: crop center manually if contour fails
    h, w = gray.shape
    # Center crop
    center_x, center_y = w // 2, h // 2
    restart_btn = gray[center_y-20:center_y+20, center_x-20:center_x+20]
    cv2.imwrite('restart_button.png', restart_btn)
    print("Restart button saved via center crop.")
