import cv2
import numpy as np
import torch
from dino_env import DinoEnv
import time

def test_vision():
    # Use verified coordinates
    env = DinoEnv(top=200, left=977, width=606, height=102)
    print("Resetting env...")
    state = env.reset() # state is (4, 84, 84)
    
    print("Showing vision. Press 'q' to quit.")
    while True:
        # Step with action 0 (Run)
        state, reward, done, _ = env.step(0)
        
        # Visualize the 4 stacked frames
        # We can tile them 2x2
        top_row = np.hstack((state[0], state[1]))
        bottom_row = np.hstack((state[2], state[3]))
        tiled = np.vstack((top_row, bottom_row))
        
        cv2.imshow("Agent Vision (4 Stacks)", tiled)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        if done:
            print("Crash detected!")
            env.reset()
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_vision()
