import torch
from dino_env import DinoEnv
from dqn_agent import DQNAgent
import time

# Configuration
N_EPISODES = 1000
SAVE_EVERY = 50
TARGET_UPDATE_EVERY = 10
ACTIONS = [0, 1, 2] # Run, Jump, Duck

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use verified coordinates from agent_agent_cuda.py
    # TOP_Y = 200, LEFT_X = 977, WIDTH = 606, HEIGHT = 102
    env = DinoEnv(top=200, left=977, width=606, height=102)
    agent = DQNAgent(len(ACTIONS), device)
    
    try:
        agent.load("dino_dqn.pth")
        print("Resuming from checkpoint...")
    except:
        print("Starting training fresh.")

    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
            # Use small sleep to prevent CPU hogging if needed
            # time.sleep(0.01)
            
        print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()
            
        if episode % SAVE_EVERY == 0:
            agent.save("dino_dqn.pth")
            print("Checkpoint saved.")

if __name__ == "__main__":
    print("Switch to the Dino game now! Starting in 5 seconds...")
    time.sleep(5)
    train()
