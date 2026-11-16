from gymnasium.envs.registration import register

# No max_episode_steps: the env truncates itself (budget depends on maze size).
register(id="MazeNav-v0", entry_point="envs.maze_env:MazeEnv")
