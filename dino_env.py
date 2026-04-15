import cv2
import numpy as np
import mss
import pydirectinput
import time
from collections import deque

class DinoEnv:
    def __init__(self, top=200, left=977, width=606, height=102):
        self.sct = mss.mss()
        self.game_area = {'top': top, 'left': left, 'width': width, 'height': height}
        self.restart_template = None
        self.load_template()
        
        # Disable pydirectinput delay
        pydirectinput.PAUSE = 0
        
        # Frame stacking: 4 frames
        self.frame_stack = deque(maxlen=4)
        
    def load_template(self):
        try:
            self.restart_template = cv2.imread('restart_button.png', 0)
        except:
            print("Warning: restart_button.png not found.")

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (84, 84))
        # Normalize to 0-1
        return resized / 255.0

    def get_screen(self):
        screen = np.array(self.sct.grab(self.game_area))
        return screen

    def is_dead(self, screen):
        if self.restart_template is not None:
            gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
            res = cv2.matchTemplate(gray, self.restart_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return max_val > 0.8
        return False

    def reset(self):
        pydirectinput.press('space')
        time.sleep(1.5) # Wait for animation
        
        screen = self.get_screen()
        processed = self.preprocess(screen)
        for _ in range(4):
            self.frame_stack.append(processed)
            
        return np.stack(self.frame_stack, axis=0)

    def step(self, action):
        if action == 1:
            pydirectinput.press('space')
        elif action == 2:
            pydirectinput.keyDown('down')
            time.sleep(0.1)
            pydirectinput.keyUp('down')
            
        screen = self.get_screen()
        done = self.is_dead(screen)
        processed = self.preprocess(screen)
        self.frame_stack.append(processed)
        
        state = np.stack(self.frame_stack, axis=0)
        reward = 1.0 if not done else -100.0
        
        return state, reward, done, {}

if __name__ == "__main__":
    # Test capture speed
    env = DinoEnv()
    start = time.time()
    for i in range(100):
        env.get_screen()
    end = time.time()
    print(f"FPS: {100 / (end - start)}")
