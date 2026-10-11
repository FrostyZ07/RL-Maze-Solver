import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import math

class MazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(MazeEnv, self).__init__()
        self.render_mode = render_mode
        
        # Connect to PyBullet
        if render_mode == 'human':
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Action space: Discrete(3)
        # 0: Forward
        # 1: Turn Left
        # 2: Turn Right
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 
        # [dist_to_target, sin(heading_error), cos(heading_error), lidar_1, ..., lidar_N]
        # 3 + 32 = 35
        self.num_lidar_rays = 32
        self.lidar_range = 10.0
        obs_dim = 3 + self.num_lidar_rays
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.agent_id = None
        self.goal_id = None
        self.plane_id = None
        self.walls = []
        
        self.max_steps = 500
        self.agent_radius = 0.15

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create walls (Grid Maze)
        self.walls = []
        wall_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1], rgbaColor=[0.7, 0.7, 0.7, 1])
        wall_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1])
        
        # 1 = Wall, 0 = Empty
        # Complex Maze (Zig-Zag)
        self.layout = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
        
        offset_x = -4
        offset_y = -4
        
        for row_idx, row in enumerate(self.layout):
            for col_idx, cell in enumerate(row):
                if cell == 1:
                    pos_x = col_idx + offset_x
                    pos_y = row_idx + offset_y
                    self.walls.append(p.createMultiBody(baseMass=0, 
                                                      baseCollisionShapeIndex=wall_collision_shape, 
                                                      baseVisualShapeIndex=wall_visual_shape, 
                                                      basePosition=[pos_x, pos_y, 1]))

        # No Dynamic Obstacle
        
        # Create Agent
        agent_start_pos = [-3, -3, 0.5] 
        agent_radius = 0.15
        agent_height = 0.5
        
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=agent_radius, length=agent_height, rgbaColor=[0, 0, 1, 1])
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=agent_radius, height=agent_height)
        
        self.agent_id = p.createMultiBody(baseMass=1, 
                                        baseCollisionShapeIndex=collision_shape, 
                                        baseVisualShapeIndex=visual_shape, 
                                        basePosition=agent_start_pos)
                                        
        # Physics tuning
        p.changeDynamics(self.agent_id, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
        p.changeDynamics(self.agent_id, -1, lateralFriction=0.0, spinningFriction=0.0)
        for wall_id in self.walls:
            p.changeDynamics(wall_id, -1, lateralFriction=0.0, spinningFriction=0.0)
        
        # Remove fixed constraint which was locking the agent
        # p.createConstraint(self.agent_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        
        # Keep agent upright (only allow Z rotation)
        # p.changeDynamics(self.agent_id, -1, angularFactor=[0, 0, 1])
        
        # Goal
        goal_pos = [3, 3, 0.5]
        goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 1])
        self.goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_visual, basePosition=goal_pos)
        self.goal_pos = np.array(goal_pos[:2])
        
        # Reset State
        self.target_pos = self.goal_pos

        if self.render_mode == 'human':
            p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0])

        self.step_count = 0
        self.visited = {self._cell(agent_start_pos[:2])}
        return self._get_obs(), {}

    @staticmethod
    def _cell(xy):
        return (int(round(xy[0])), int(round(xy[1])))

    def step(self, action):
        self.step_count += 1
        
        # No Dynamic Obstacle Movement
        
        # Discrete Action Logic
        # 0: Forward, 1: Left, 2: Right
        forward_speed = 0.0
        turn_speed = 0.0
        
        if action == 0: # Forward
            forward_speed = 5.0
        elif action == 1: # Left
            turn_speed = 5.0 
            forward_speed = 0.0 # Pure rotation
        elif action == 2: # Right
            turn_speed = -5.0 
            forward_speed = 0.0 # Pure rotation
            
        pos, orn = p.getBasePositionAndOrientation(self.agent_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        vx = forward_speed * np.cos(yaw)
        vy = forward_speed * np.sin(yaw)
        
        p.resetBaseVelocity(self.agent_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, turn_speed])
        
        p.stepSimulation()
        if self.render_mode == 'human':
            time.sleep(1./30.)
            
        obs = self._get_obs()
        
        # Event-based reward only: no distance/heading shaping, no path hints.
        agent_pos = np.array(pos[:2])
        reward = -0.01  # step penalty

        # First-visit novelty bonus (exploration signal, needs no map knowledge)
        cell = self._cell(agent_pos)
        if cell not in self.visited:
            self.visited.add(cell)
            reward += 0.05

        terminated = False
        truncated = self.step_count >= self.max_steps

        if np.linalg.norm(agent_pos - self.goal_pos) < 0.5:
            reward += 100
            terminated = True

        # Wall collision penalty
        for contact in p.getContactPoints(self.agent_id):
            if contact[2] in self.walls:
                reward -= 0.1
                break
        
        # Debug Draw
        if self.render_mode == 'human':
             p.addUserDebugLine([agent_pos[0], agent_pos[1], 0.5], [self.target_pos[0], self.target_pos[1], 0.5], [0, 1, 1], lifeTime=0.1)
                
        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.agent_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        agent_pos = np.array(pos[:2])
        target_pos = self.target_pos
        
        # Relative State
        dist_to_target = np.linalg.norm(agent_pos - target_pos)
        
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        angle_to_target = math.atan2(dy, dx)
        heading_error = angle_to_target - yaw
        
        sin_heading = np.sin(heading_error)
        cos_heading = np.cos(heading_error)
        
        # Lidar
        lidar_readings = []
        for i in range(self.num_lidar_rays):
            angle = yaw + (i - self.num_lidar_rays/2) * (np.pi / self.num_lidar_rays) * 2 
            c, s = np.cos(angle), np.sin(angle)
            r0 = self.agent_radius + 0.01  # start outside own body
            ray_from = [pos[0] + r0 * c, pos[1] + r0 * s, pos[2]]
            ray_to = [pos[0] + self.lidar_range * c, pos[1] + self.lidar_range * s, pos[2]]
            hit_fraction = p.rayTest(ray_from, ray_to)[0][2]
            lidar_readings.append(r0 + hit_fraction * (self.lidar_range - r0))
            
        obs = np.concatenate([[dist_to_target, sin_heading, cos_heading], lidar_readings])
        return obs.astype(np.float32)

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass
