import gymnasium as gym
import numpy as np
import pybullet as p
import time
from envs.maze_env import MazeEnv

def main():
    env = MazeEnv(render_mode='human')
    obs, _ = env.reset()
    
    print("Manual Control: Use Arrow Keys to move.")
    print("UP/DOWN: Linear Velocity")
    print("LEFT/RIGHT: Angular Velocity")
    print("Q: Quit")
    
    linear_vel = 0
    angular_vel = 0
    
    while True:
        keys = p.getKeyboardEvents()
        
        # Reset velocities
        linear_vel = 0
        angular_vel = 0
        
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            linear_vel = 1.0
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            linear_vel = -1.0
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            angular_vel = 1.0
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            angular_vel = -1.0
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            break
            
        action = np.array([linear_vel, angular_vel])
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print("Goal Reached!")
            obs, _ = env.reset()
            
    env.close()

if __name__ == "__main__":
    main()
