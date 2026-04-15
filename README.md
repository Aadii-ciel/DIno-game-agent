# Dino Game Agent (DQN)

A robust Deep Reinforcement Learning agent trained to play the offline browser Dino game using **Deep Q-Networks (DQN)** with **Convolutional Neural Networks (CNN)**.

This project is optimized for high-performance training on **NVIDIA GPUs (RTX 4060)** using **PyTorch** and **CUDA**.

## 🚀 Features

- **Deep Q-Learning**: Uses a 3-layer CNN to process raw pixel input from the game.
- **Experience Replay**: Stores and samples past experiences to break temporal correlations and stabilize training.
- **Target Network**: Employs a separate network for stable Q-value target calculation.
- **Frame Stacking**: Stacks 4 consecutive grayscale frames into a single state (4x84x84). This allows the agent to perceive **speed and direction** of obstacles.
- **Reliable Death Detection**: Uses **OpenCV Template Matching** for 100% accurate game-over detection.
- **Low Latency Control**: Uses `pydirectinput` for near-instant reaction times.
- **GPU Accelerated**: Fully compatible with **CUDA** for lightning-fast training.

## 🛠️ Architecture

- **`dino_env.py`**: A custom environment class that handles screen capture, preprocessing, action execution, and death detection.
- **`model.py`**: Defines the Convolutional Neural Network (CNN) architecture.
- **`dqn_agent.py`**: The core DQN logic, including the replay buffer and epsilon-greedy exploration strategy.
- **`train.py`**: The main entry point to start training the agent.

## 📦 Installation

### Prerequisites
- Python 3.10 - 3.14
- NVIDIA GPU with CUDA support
- Chrome Browser (to run the game)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Aadii-ciel/DIno-game-agent.git
   cd DIno-game-agent
   ```

2. Install dependencies (specifically PyTorch with CUDA):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install opencv-python mss numpy keyboard pydirectinput
   ```

## 🎮 Usage

1. **Open Chrome** and navigate to `chrome://dino`.
2. **Set the Window Position**: Ensure the Dino game is visible on your screen.
3. **Verify Vision**: Run the vision test to ensure the agent "sees" the game correctly:
   ```bash
   python test_env_vision.py
   ```
4. **Start Training**:
   ```bash
   python train.py
   ```

## ⚙️ Configuration

The default coordinates are:
- `TOP_Y = 200`
- `LEFT_X = 977`
- `WIDTH = 606`
- `HEIGHT = 102`

If your browser window is different, use `Coordinates Finder.py` to find your specific coordinates and update them in `train.py`.

## 📄 License
MIT
