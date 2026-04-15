import pyautogui
import time
import os

print("---------------------------------------------------------")
print(" LIVE MOUSE COORDINATES (Press Ctrl+C to stop)")
print("---------------------------------------------------------")
print("1. Hover over the TOP-LEFT corner of the Dino game.")
print("2. Note the X and Y.")
print("3. Hover over the BOTTOM-RIGHT corner.")
print("4. Note the X and Y.")
print("---------------------------------------------------------")

try:
    while True:
        # Get current mouse position
        x, y = pyautogui.position()
        
        # Print coordinates on the same line (clears previous line)
        # ljust ensures the string stays a fixed length so numbers don't jiggle
        print(f"X: {x} | Y: {y}".ljust(30), end="\r")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")