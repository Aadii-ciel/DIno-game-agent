import pyautogui
import keyboard
import time
import os

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

print("=== DINO GAME COORDINATE FINDER ===")
print("1. Hover mouse over the TOP-LEFT corner of the game area.")
print("2. Press the '1' key on your keyboard.")
print("3. Hover mouse over the BOTTOM-RIGHT corner.")
print("4. Press the '2' key.")
print("5. Press 'q' to quit at any time.")
print("=====================================")

top_left = None
bottom_right = None

while True:
    try:
        x, y = pyautogui.position()
        
        # Display current mouse position nicely
        print(f"\rCurrent Mouse: X={x}, Y={y}   (Press '1' for Top-Left, '2' for Bottom-Right, 'q' to Quit)", end="")
        
        if keyboard.is_pressed('1'):
            top_left = (x, y)
            print(f"\n[CAPTURED] Top-Left: {top_left}")
            time.sleep(0.5) # Debounce to prevent double press
            
        if keyboard.is_pressed('2'):
            bottom_right = (x, y)
            print(f"\n[CAPTURED] Bottom-Right: {bottom_right}")
            time.sleep(0.5)

        if top_left and bottom_right:
            # Calculate Width and Height
            w = bottom_right[0] - top_left[0]
            h = bottom_right[1] - top_left[1]
            
            print("\n\n=== SUCCESSS! COPY THIS INTO YOUR BOT CODE: ===")
            print(f"TOP_Y   = {top_left[1]}")
            print(f"LEFT_X  = {top_left[0]}")
            print(f"WIDTH   = {w}")
            print(f"HEIGHT  = {h}")
            print("===============================================")
            
            # Reset so you can try again if you missed
            top_left = None
            bottom_right = None
            print("\nResetting... You can measure again or press 'q' to quit.\n")

        if keyboard.is_pressed('q'):
            print("\nQuitting...")
            break
        
        time.sleep(0.05)
        
    except KeyboardInterrupt:
        break