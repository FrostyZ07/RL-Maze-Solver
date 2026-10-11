import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from collections import deque

class MazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    # Difficulty knobs (set NUM_OBSTACLES = 0 etc. to switch one off while debugging)
    MAZE_SIZE = 15          # odd; 15 -> 13x13 playable cells
    NUM_OBSTACLES = 2       # patrolling blocks
    TIME_SLACK = 1.5        # episode budget = slack * estimated actions for the shortest route
    WALL_PENALTY = 0.05
    OBSTACLE_PENALTY = 1.0
    LIDAR_NOISE = 0.05      # std (m) of Gaussian noise on each lidar reading
    ACTION_NOISE = 0.1      # std of relative error on forward/turn speed (wheel slip)

    def __init__(self, render_mode=None, fixed=False):
        super(MazeEnv, self).__init__()
        self.render_mode = render_mode
        self.fixed = fixed  # True: the seed-0 test maze, goal bottom-right; False: new maze + goal every episode

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
        # [time_left, goal_visible, dist_to_target, sin(heading_error), cos(heading_error), lidar_1, ..., lidar_N]
        # time_left = fraction of the step budget remaining (1 -> 0)
        # the three goal values are 0 unless the goal is in line of sight; 5 + 8 = 13
        self.num_lidar_rays = 8   # every 45 degrees
        self.lidar_range = 10.0
        obs_dim = 5 + self.num_lidar_rays

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.agent_id = None
        self.goal_id = None
        self.plane_id = None
        self.walls = []
        self.obstacles = []

        self.max_steps = 500  # overwritten per episode in reset()
        self.agent_radius = 0.15

        # Seed-0 layout (the test maze when fixed=True; otherwise reset() regenerates it)
        self.size = self.MAZE_SIZE
        self.offset = -(self.size - 1) // 2   # tile (r, c) -> world (c + offset, r + offset)
        self.layout = self._make_layout(self.size)
        self.segments = self._find_segments()

    def _make_layout(self, size, seed=0, loops=6):
        """DFS maze (long corridors + dead ends), then open a few walls to make loops."""
        rng = np.random.default_rng(seed)
        grid = [[1] * size for _ in range(size)]
        grid[1][1] = 0
        stack = [(1, 1)]
        while stack:
            r, c = stack[-1]
            nbrs = [(r + dr, c + dc, dr, dc) for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))
                    if 0 < r + dr < size - 1 and 0 < c + dc < size - 1 and grid[r + dr][c + dc] == 1]
            if not nbrs:
                stack.pop()
                continue
            nr, nc, dr, dc = nbrs[rng.integers(len(nbrs))]
            grid[r + dr // 2][c + dc // 2] = 0
            grid[nr][nc] = 0
            stack.append((nr, nc))
        # wall tiles between two open cells -> candidates for loops
        cands = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)
                 if grid[r][c] == 1 and
                 ((r % 2 == 1 and c % 2 == 0 and grid[r][c - 1] == 0 and grid[r][c + 1] == 0) or
                  (r % 2 == 0 and c % 2 == 1 and grid[r - 1][c] == 0 and grid[r + 1][c] == 0))]
        for i in rng.permutation(len(cands))[:loops]:
            r, c = cands[i]
            grid[r][c] = 0
        return grid

    def _find_segments(self, min_len=5):
        """Straight corridor runs (>= min_len open tiles) where obstacles can patrol: ((r0,c0), (r1,c1))."""
        segs = []
        n = self.size
        for r in range(n):
            c = 0
            while c < n:
                if self.layout[r][c] == 0:
                    e = c
                    while e + 1 < n and self.layout[r][e + 1] == 0:
                        e += 1
                    if e - c + 1 >= min_len:
                        segs.append(((r, c), (r, e)))
                    c = e + 1
                else:
                    c += 1
        for c in range(n):
            r = 0
            while r < n:
                if self.layout[r][c] == 0:
                    e = r
                    while e + 1 < n and self.layout[e + 1][c] == 0:
                        e += 1
                    if e - r + 1 >= min_len:
                        segs.append(((r, c), (e, c)))
                    r = e + 1
                else:
                    r += 1
        return segs

    def _route_budget(self, start, goal):
        """Episode step limit from the shortest route: ~6 actions per tile, ~9 per 90-degree turn."""
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (cur[0] + dr, cur[1] + dc)
                if self.layout[n[0]][n[1]] == 0 and n not in prev:
                    prev[n] = cur
                    q.append(n)
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        turns = sum(1 for a, b, c in zip(path, path[1:], path[2:])
                    if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]))
        return int(self.TIME_SLACK * (6 * (len(path) - 1) + 9 * turns))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.fixed:
            self.layout = self._make_layout(self.size, seed=int(self.np_random.integers(2**31)))
            self.segments = self._find_segments()

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self.plane_id = p.loadURDF("plane.urdf")

        # Create walls (Grid Maze, 1 = Wall, 0 = Empty)
        self.walls = []
        wall_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1], rgbaColor=[0.7, 0.7, 0.7, 1])
        wall_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1])

        off = self.offset
        for row_idx, row in enumerate(self.layout):
            for col_idx, cell in enumerate(row):
                if cell == 1:
                    self.walls.append(p.createMultiBody(baseMass=0,
                                                      baseCollisionShapeIndex=wall_collision_shape,
                                                      baseVisualShapeIndex=wall_visual_shape,
                                                      basePosition=[col_idx + off, row_idx + off, 1]))

        open_cells = [(r, c) for r, row in enumerate(self.layout) for c, v in enumerate(row) if v == 0]
        far = self.size // 2
        if self.fixed:
            # goal bottom-right; start = random open tile far enough from it
            gr, gc = self.size - 2, self.size - 2
            starts = [(r, c) for r, c in open_cells if abs(r - gr) + abs(c - gc) >= far]
            r, c = starts[self.np_random.integers(len(starts))]
        else:
            r, c = open_cells[self.np_random.integers(len(open_cells))]
            goals = [(a, b) for a, b in open_cells if abs(a - r) + abs(b - c) >= far]
            gr, gc = goals[self.np_random.integers(len(goals))]
        self.goal_cell = (gr, gc)
        agent_start_pos = [c + off, r + off, 0.5]
        self.max_steps = self._route_budget((r, c), (gr, gc))

        # Create Agent
        agent_radius = self.agent_radius
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

        # Moving obstacles: small blocks patrolling back and forth along corridor runs
        self.obstacles = []
        if self.NUM_OBSTACLES and self.segments:
            obs_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.5], rgbaColor=[1, 0, 0, 1])
            obs_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.5])
            for i in self.np_random.choice(len(self.segments), size=min(self.NUM_OBSTACLES, len(self.segments)), replace=False):
                (r0, c0), (r1, c1) = self.segments[i]
                a = np.array([c0 + off, r0 + off], dtype=float)
                b = np.array([c1 + off, r1 + off], dtype=float)
                length = np.linalg.norm(b - a)
                body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obs_collision,
                                         baseVisualShapeIndex=obs_visual, basePosition=[a[0], a[1], 0.5])
                self.obstacles.append({'id': body, 'a': a, 'dir': (b - a) / length, 'len': length,
                                       't': float(self.np_random.uniform(0, 2 * length))})
            self._move_obstacles(0.0)

        # Goal
        gr, gc = self.goal_cell
        goal_pos = [gc + off, gr + off, 0.5]
        goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 1])
        self.goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_visual, basePosition=goal_pos)
        self.goal_pos = np.array(goal_pos[:2])

        # Reset State
        self.target_pos = self.goal_pos

        if self.render_mode == 'human':
            p.resetDebugVisualizerCamera(cameraDistance=1.3 * self.size, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0])

        self.step_count = 0
        self.visited = {self._cell(agent_start_pos[:2])}
        return self._get_obs(), {}

    @staticmethod
    def _cell(xy):
        return (int(round(xy[0])), int(round(xy[1])))

    def _move_obstacles(self, speed=0.1):
        for o in self.obstacles:
            o['t'] += speed
            s = o['t'] % (2 * o['len'])
            u = s if s <= o['len'] else 2 * o['len'] - s   # triangle wave: out and back
            x, y = o['a'] + o['dir'] * u
            p.resetBasePositionAndOrientation(o['id'], [x, y, 0.5], [0, 0, 0, 1])

    def step(self, action):
        self.step_count += 1

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

        # Movement noise: the commanded speed is never exactly what you get
        forward_speed *= 1 + self.ACTION_NOISE * self.np_random.standard_normal()
        turn_speed *= 1 + self.ACTION_NOISE * self.np_random.standard_normal()

        vx = forward_speed * np.cos(yaw)
        vy = forward_speed * np.sin(yaw)

        p.resetBaseVelocity(self.agent_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, turn_speed])

        self._move_obstacles()
        for _ in range(8):
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
            reward += 0.5

        terminated = False
        truncated = self.step_count >= self.max_steps

        if np.linalg.norm(agent_pos - self.goal_pos) < 0.5:
            reward += 100
            terminated = True

        # Collision penalties
        obstacle_ids = {o['id'] for o in self.obstacles}
        hit_wall = hit_obstacle = False
        for contact in p.getContactPoints(self.agent_id):
            if contact[2] in self.walls:
                hit_wall = True
            elif contact[2] in obstacle_ids:
                hit_obstacle = True
        if hit_wall:
            reward -= self.WALL_PENALTY
        if hit_obstacle:
            reward -= self.OBSTACLE_PENALTY

        return obs, reward, terminated, truncated, {"goal": terminated}

    def _goal_visible(self, pos):
        """Line of sight to the goal: the goal sphere has no collision shape, so a clear ray means no wall/obstacle between."""
        d = self.goal_pos - np.array(pos[:2])
        dist = np.linalg.norm(d)
        r0 = self.agent_radius + 0.01  # start outside own body
        if dist <= r0:
            return True
        u = d / dist
        ray_from = [pos[0] + r0 * u[0], pos[1] + r0 * u[1], 0.5]
        return p.rayTest(ray_from, [self.goal_pos[0], self.goal_pos[1], 0.5])[0][0] == -1

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.agent_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        agent_pos = np.array(pos[:2])
        target_pos = self.target_pos

        # Relative goal state, only while the goal is in line of sight (else zeros)
        goal_obs = [0.0, 0.0, 0.0, 0.0]
        if self._goal_visible(pos):
            dx = target_pos[0] - agent_pos[0]
            dy = target_pos[1] - agent_pos[1]
            heading_error = math.atan2(dy, dx) - yaw
            goal_obs = [1.0, np.hypot(dx, dy), np.sin(heading_error), np.cos(heading_error)]

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
        # Sensor noise
        lidar_readings = np.clip(np.array(lidar_readings)
                                 + self.LIDAR_NOISE * self.np_random.standard_normal(self.num_lidar_rays),
                                 0.0, self.lidar_range)

        time_left = 1.0 - self.step_count / self.max_steps
        obs = np.concatenate([[time_left], goal_obs, lidar_readings])
        return obs.astype(np.float32)

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass
