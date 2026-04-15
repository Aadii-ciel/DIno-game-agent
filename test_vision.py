# test_vision_fixed.py
import cv2
import time
from dino_env import DinoEnv

def main():
    print("1. Initializing Environment...")
    env = DinoEnv()
    
    print("2. Resetting Game...")
    obs, _ = env.reset()
    
    print("3. Starting Vision Loop. Press 'q' to quit.")
    while True:
        # Step the environment (Action 0 = Do Nothing)
        obs, _, done, _, _ = env.step(0)
        
        # Squeeze removes the extra dimension: (84, 84, 1) -> (84, 84)
        img = obs.squeeze()
        
        # Show the image
        cv2.imshow("Agent Vision", img)
        
        # CRITICAL: This 1ms delay allows the window to draw itself
        if cv2.waitKey(1) == ord('q'):
            break
            
        # If game over, reset immediately so we keep seeing something
        if done:
            env.reset()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()