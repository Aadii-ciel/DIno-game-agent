import cv2
import numpy as np
import mss
import pyautogui
import time
import pickle
import os
import keyboard

# ======================================================================================
# CONFIGURATION
# ======================================================================================

SCALE = 1.0 

# YOUR VERIFIED COORDINATES:
TOP_Y   = 200    
LEFT_X  = 977    
WIDTH   = 606    
HEIGHT  = 102    

# The area to watch
GAME_AREA = {
    'top': int(TOP_Y * SCALE), 
    'left': int(LEFT_X * SCALE), 
    'width': int(WIDTH * SCALE), 
    'height': int(HEIGHT * SCALE)
}

# Restart Button Position
RESTART_BTN_X = int(LEFT_X + (WIDTH / 2))
RESTART_BTN_Y = int(TOP_Y + (HEIGHT / 2))

BRAIN_FILE = "dino_brain.pkl"

# AI Parameters
ALPHA = 0.1       
GAMMA = 0.99      
EPSILON_START = 1.0 
EPSILON_DECAY = 0.999 # Slower decay so it explores more
EPSILON_MIN = 0.01

# ======================================================================================
# THE AI CLASS
# ======================================================================================

class DinoBot:
    def __init__(self):
        self.q_table = {} 
        self.epsilon = EPSILON_START
        self.episode = 0
        self.load_brain()
        self.sct = mss.mss()

    def get_state(self, image):
        # Focus on the right side where obstacles come from
        roi = image[:, 50:] 
        contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        distance = 999
        width = 0
        
        if contours:
            obstacles = [c for c in contours if cv2.contourArea(c) > 50]
            if obstacles:
                x, y, w, h = cv2.boundingRect(obstacles[0])
                distance = x
                width = w

        # Return state as tuple (Distance, Width)
        return (int(distance / 20), int(width / 10))

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1, 2])
        
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0] 
            
        return np.argmax(self.q_table[state])

    def perform_action(self, action):
        if action == 1:
            pyautogui.press('space') # Jump
        elif action == 2:
            pyautogui.keyDown('down') # Duck
            time.sleep(0.05)          # Quick duck
            pyautogui.keyUp('down')
        else:
            pass # Run

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]

        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        
        # Q-Learning Formula
        new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)
        self.q_table[state][action] = new_q

    def save_brain(self):
        try:
            with open(BRAIN_FILE, 'wb') as f:
                pickle.dump((self.q_table, self.epsilon, self.episode), f)
            print(f"--> Saved! Episode: {self.episode}, High Score State count: {len(self.q_table)}")
        except Exception as e:
            print(f"Error saving: {e}")

    def load_brain(self):
        if os.path.exists(BRAIN_FILE):
            try:
                with open(BRAIN_FILE, 'rb') as f:
                    self.q_table, self.epsilon, self.episode = pickle.load(f)
                print(f"--> Brain Loaded! Resuming from Episode {self.episode}")
            except Exception as e:
                print("Starting fresh.")

# ======================================================================================
# MAIN LOOP
# ======================================================================================

def main():
    bot = DinoBot()
    print("Starting in 3 seconds. SWITCH TO CHROME NOW!")
    time.sleep(3)
    
    # Initial Focus Click
    pyautogui.moveTo(RESTART_BTN_X, RESTART_BTN_Y)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.press('space')

    # Window Setup
    window_name = "Dino AI Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 150)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while True:
        # --------------------------------------------------------
        # 1. CAPTURE CURRENT FRAME
        # --------------------------------------------------------
        screen_raw = np.array(bot.sct.grab(GAME_AREA))
        screen_gray = cv2.cvtColor(screen_raw, cv2.COLOR_BGRA2GRAY)
        _, processed_screen = cv2.threshold(screen_gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        current_state = bot.get_state(processed_screen)
        
        # --------------------------------------------------------
        # 2. ACT
        # --------------------------------------------------------
        action = bot.act(current_state)
        action_name = ["RUN", "JUMP", "DUCK"][action]
        bot.perform_action(action)
        
        # --------------------------------------------------------
        # 3. CAPTURE NEXT FRAME (Check Outcome)
        # --------------------------------------------------------
        # Small delay to let the action happen on screen
        time.sleep(0.05) 
        
        next_screen_raw = np.array(bot.sct.grab(GAME_AREA))
        next_screen_gray = cv2.cvtColor(next_screen_raw, cv2.COLOR_BGRA2GRAY)
        _, next_processed = cv2.threshold(next_screen_gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        next_state = bot.get_state(next_processed)

        # --------------------------------------------------------
        # 4. CHECK FOR CRASH (CRITICAL FIX)
        # --------------------------------------------------------
        # We compare the two frames. If the pixels didn't change enough, the game is frozen/over.
        frame_diff = cv2.absdiff(processed_screen, next_processed)
        movement_score = np.sum(frame_diff)
        
        # If movement is very low, the ground has stopped scrolling -> DEAD
        is_dead = movement_score < 500 # Threshold (adjust if needed)

        if is_dead:
            reward = -100 # BIG PUNISHMENT
            print(f"CRASH DETECTED! (Ep {bot.episode}) - Punishment applied.")
            bot.learn(current_state, action, reward, next_state)
            
            # Restart Sequence
            bot.episode += 1
            if bot.episode % 20 == 0:
                bot.save_brain()
            
            # Decay Epsilon
            bot.epsilon = max(EPSILON_MIN, bot.epsilon * EPSILON_DECAY)
            
            # Click Restart
            pyautogui.press('space')
            time.sleep(1.0) # Wait for restart animation
            
            # Reset loop to start of new game
            continue 
            
        else:
            reward = 1 # ALIVE REWARD
            bot.learn(current_state, action, reward, next_state)

        # --------------------------------------------------------
        # 5. VISUALIZATION
        # --------------------------------------------------------
        display_img = processed_screen.copy()
        cv2.putText(display_img, f"{action_name} | Ep: {bot.episode}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, display_img)
        
        if keyboard.is_pressed('q') or (cv2.waitKey(1) & 0xFF == ord('q')):
            bot.save_brain()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()