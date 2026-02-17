import os
import pickle
import time
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, RecordVideo
import neat
import numpy as np

# Add the directory where videos will be saved
VIDEO_BASE_DIR = r""
MAX_STEPS = 5000

def create_env(video_dir=None):
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, [["right","B"], ["right","A","B"]])
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (32, 32))
    if video_dir:
        env = RecordVideo(env, video_dir, episode_trigger=lambda e: True)
    return env

def load_winner(winner_file, config_file):
    with open(winner_file, 'rb') as f:
        genome = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    return genome, config

def record_winner(winner_file, config_file):
    # Add timestamp to video folder to prevent overwriting
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_dir = os.path.join(VIDEO_BASE_DIR, f"run_{timestamp}")
    os.makedirs(video_dir, exist_ok=True)

    genome, config = load_winner(winner_file, config_file)
    env = create_env(video_dir)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    state = env.reset()
    total_reward = 0
    max_x = 0

    for step in range(MAX_STEPS):
        env.render()
        state_flat = np.array(state).flatten() / 255.0
        output = net.activate(state_flat)
        action = int(np.argmax(output))
        state, reward, done, info = env.step(action)

        total_reward += reward
        max_x = max(max_x, info.get("x_pos", 0))

        if done:
            break

    env.close()
    print(f"Video recorded in: {video_dir}")
    print(f"Total Reward: {total_reward}, Max X Position: {max_x}")

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    winner_path = os.path.join(local_dir, 'winner.pkl')

    if not os.path.exists(winner_path):
        print("Winner file not found!")
    elif not os.path.exists(config_path):
        print("Config file not found!")
    else:
        record_winner(winner_path, config_path)
