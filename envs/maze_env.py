import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from collections import deque

# Maze layouts come from a seed. Training draws layout seeds >= EVAL_SEED_RANGE; evaluation uses
# seeds below it, so test mazes are never seen in training.
EVAL_SEED_RANGE = 2**20

class MazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    # Difficulty knobs (set NUM_OBSTACLES = 0 etc. to switch one off while debugging)
    MAZE_SIZE = 15          # odd; 15 -> 13x13 playable cells
    NUM_OBSTACLES = 2       # patrolling blocks
    LOOPS = 6               # extra walls opened after the DFS maze, to create loops
    STEPS_PER_TILE = 15     # episode budget = this * open tiles; depends on maze size only, so time_left carries no route-length hint
    NOVELTY_BONUS = 0.1     # first visit to a tile; keep small so the +100 goal dominates
    REVISIT_PENALTY = 0.05  # re-entering an already visited tile (discourages loops)
    WALL_PENALTY = 0.05
    OBSTACLE_PENALTY = 1.0
    LIDAR_NOISE = 0.05      # std (m) of Gaussian noise on each lidar reading
    ACTION_NOISE = 0.1      # std of relative error on forward/turn speed (wheel slip)
    CROP = 7                # side (tiles) of the egocentric map window in the observation
    CARDINALS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))  # world axes; extra scan rays used only to map

    def __init__(self, render_mode=None, fixed=False, size=None):
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
        # [time_left, goal_visible, dist_to_target, sin(heading_error), cos(heading_error),
        #  visits ahead/left/behind/right, lidar_1, ..., lidar_N]
        # time_left = fraction of the step budget remaining (1 -> 0)
        # the three goal values are 0 unless the goal is in line of sight
        # visits = how often the neighbouring tile was entered (odometry memory), 0..1; 5 + 4 + 8 = 17
        # then, from the agent's own map (built from lidar + odometry, never from the true layout):
        #  [has_target, path_len/size, sin, cos] to the next tile on the BFS route to the remembered goal
        #  (or, before the goal was seen, to the nearest unexplored tile), then a 7x7 egocentric crop of the map
        #  (+1 open, 0 unknown, -1 wall/outside), axis-aligned to the nearest heading; 17 + 4 + 49 = 70
        self.num_lidar_rays = 8   # every 45 degrees
        self.lidar_range = 10.0
        obs_dim = 5 + 4 + self.num_lidar_rays + 4 + self.CROP * self.CROP

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.agent_id = None
        self.goal_id = None
        self.plane_id = None
        self.walls = []
        self.obstacles = []

        self.max_steps = 500  # overwritten per episode in reset()
        self.agent_radius = 0.15

        # Seed-0 layout (the test maze when fixed=True; otherwise reset() regenerates it)
        self.size = size or self.MAZE_SIZE
        self.next_size = self.size
        self.offset = -(self.size - 1) // 2   # tile (r, c) -> world (c + offset, r + offset)
        self.layout_seed = 0
        self.layout = self._make_layout(self.size, seed=0)
        self.segments = self._find_segments()
        self.start_cell = self.goal_cell = None

    def set_size(self, size):
        """Curriculum hook: takes effect on the next reset(). The running episode keeps its maze, offset and
        map, which are all sized for it; `next_size` is what the next episode will use."""
        self.next_size = size

    def _make_layout(self, size, seed=0, loops=None):
        loops = self.LOOPS if loops is None else loops
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

    @staticmethod
    def _bfs(start, passable, is_goal):
        """BFS over 4-connected tiles: list of (r, c) from start to the nearest tile with is_goal, or None."""
        prev = {start: None}
        q = deque([start])
        found = None
        while q:
            cur = q.popleft()
            if is_goal(cur):
                found = cur
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (cur[0] + dr, cur[1] + dc)
                if n not in prev and passable(n):
                    prev[n] = cur
                    q.append(n)
        if found is None:
            return None
        path = []
        while found is not None:
            path.append(found)
            found = prev[found]
        path.reverse()
        return path

    def shortest_path(self, start, goal):
        """BFS over open tiles of the true layout: list of (r, c) from start to goal inclusive, or None."""
        return self._bfs(start, lambda n: self.layout[n[0]][n[1]] == 0, lambda n: n == goal)

    def _time_budget(self, size):
        """Episode step limit from the maze size alone (open tiles of a DFS maze plus loops),
        never from the start-goal route, so time_left cannot reveal how far the hidden goal is."""
        k = (size - 1) // 2
        spanning = k * k - 1                      # passages of the DFS spanning tree
        possible = 2 * k * (k - 1)                # all wall tiles between two cells
        open_tiles = k * k + spanning + min(self.LOOPS, possible - spanning)
        return self.STEPS_PER_TILE * open_tiles

    def get_pose(self):
        """(xy position, yaw) of the agent."""
        pos, orn = p.getBasePositionAndOrientation(self.agent_id)
        return np.array(pos[:2]), p.getEulerFromQuaternion(orn)[2]

    def reset(self, seed=None, options=None):
        """options (all optional): {"size": maze size, "layout_seed": int}. Evaluation uses them to pin an
        episode; layout seeds below EVAL_SEED_RANGE are reserved for evaluation."""
        super().reset(seed=seed)
        options = options or {}
        if "size" in options:
            self.set_size(int(options["size"]))
        self.size = self.next_size
        self.offset = -(self.size - 1) // 2

        if self.fixed:
            self.layout_seed = 0
        elif "layout_seed" in options:
            self.layout_seed = int(options["layout_seed"])
        else:
            self.layout_seed = int(self.np_random.integers(EVAL_SEED_RANGE, 2**31))
        self.layout = self._make_layout(self.size, seed=self.layout_seed)
        self.segments = self._find_segments()

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self.plane_id = p.loadURDF("plane.urdf")

        # Create walls (Grid Maze, 1 = Wall, 0 = Empty)
        self.walls = []
        self.wall_tile = {}   # body id -> (r, c); lets a lidar hit say which tile is a wall
        wall_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1], rgbaColor=[0.7, 0.7, 0.7, 1])
        wall_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1])

        off = self.offset
        for row_idx, row in enumerate(self.layout):
            for col_idx, cell in enumerate(row):
                if cell == 1:
                    wid = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=wall_collision_shape,
                                            baseVisualShapeIndex=wall_visual_shape,
                                            basePosition=[col_idx + off, row_idx + off, 1])
                    self.walls.append(wid)
                    self.wall_tile[wid] = (row_idx, col_idx)

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
        self.start_cell = (r, c)
        self.goal_cell = (gr, gc)
        agent_start_pos = [c + off, r + off, 0.5]
        self.max_steps = self._time_budget(self.size)

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
        self.last_cell = self._cell(agent_start_pos[:2])
        self.visits = {self.last_cell: 1}  # tile -> times entered
        self.known = np.full((self.size, self.size), -1, dtype=np.int8)  # own map: -1 unknown, 0 open, 1 wall
        self.searched = np.zeros((self.size, self.size), dtype=bool)  # tiles the goal is known not to be on
        self.goal_seen_cell = None
        return self._get_obs(), {}

    @staticmethod
    def _cell(xy):
        return (int(round(xy[0])), int(round(xy[1])))

    def _tile(self, xy):
        """World (x, y) -> map tile (r, c)."""
        return (int(round(xy[1])) - self.offset, int(round(xy[0])) - self.offset)

    def _in_map(self, t):
        return 0 <= t[0] < self.size and 0 <= t[1] < self.size

    def _mark_ray(self, origin, u, reading, hit_id, scan=False):
        """Fold one range reading into the map: tiles along the free part of the ray are open; a wall body
        hit marks that wall's tile. Tiles already known are never overwritten.
        scan=True (world-axis rays): tiles whose centre lies on the clear stretch of the agent's own row/column
        are also ruled out as goal locations, since the goal would have been in line of sight from there."""
        d = 0.25
        while d <= reading - 0.3:   # stay clear of the hit point, which sits on a tile edge
            t = self._tile(origin + d * u)
            if self._in_map(t) and self.known[t] == -1:
                self.known[t] = 0
            d += 0.25
        if scan:
            axis = 0 if u[0] else 1
            sign = u[axis]
            if abs(origin[1 - axis] - round(origin[1 - axis])) < 0.35:   # ray stays inside the row/column strip
                k = 0
                while True:
                    centre = round(origin[axis]) + sign * k
                    if sign * (centre - origin[axis]) > reading - 0.3:
                        break
                    if sign * (centre - origin[axis]) >= -0.35:
                        xy = origin.copy()
                        xy[axis] = centre
                        xy[1 - axis] = round(origin[1 - axis])
                        t = self._tile(xy)
                        if self._in_map(t):
                            self.searched[t] = True
                    k += 1
        t = self.wall_tile.get(hit_id)
        if t is not None and self.known[t] == -1:
            self.known[t] = 1

    def _update_map(self, pos, yaw, readings, hit_ids):
        """Own map from the noisy lidar (+ 4 world-axis scan rays) and odometry; no access to the layout."""
        origin = np.array(pos[:2])
        t = self._tile(origin)
        if not self._in_map(t):   # knocked out of the maze by the physics: nothing sensible to map
            return
        self.known[t] = 0   # where we stand is open (and the goal is not here, or we would have won)
        self.searched[t] = True
        n = self.num_lidar_rays
        for i in range(n):
            a = yaw + (i - n / 2) * (2 * np.pi / n)
            self._mark_ray(origin, np.array([np.cos(a), np.sin(a)]), readings[i], hit_ids[i])
        r0 = self.agent_radius + 0.01
        starts = [[pos[0] + r0 * c, pos[1] + r0 * s, pos[2]] for c, s in self.CARDINALS]
        ends = [[pos[0] + self.lidar_range * c, pos[1] + self.lidar_range * s, pos[2]] for c, s in self.CARDINALS]
        hits = p.rayTestBatch(starts, ends)
        noise = self.LIDAR_NOISE * self.np_random.standard_normal(len(hits))
        for (c, s), h, e in zip(self.CARDINALS, hits, noise):
            reading = float(np.clip(r0 + h[2] * (self.lidar_range - r0) + e, 0.0, self.lidar_range))
            self._mark_ray(origin, np.array([c, s]), reading, h[0], scan=True)
        if self._goal_visible(pos):
            gt = self._tile(self.goal_pos)
            self.goal_seen_cell = gt
            self.known[gt] = 0
            d = 0.25
            gd = np.linalg.norm(self.goal_pos - origin)
            while d < gd:   # the line of sight is free space
                tt = self._tile(origin + d * (self.goal_pos - origin) / gd)
                if self.known[tt] == -1:
                    self.known[tt] = 0
                d += 0.25

    def _route(self):
        """Path (tiles, from the current one) over the known map to the remembered goal, else to the nearest
        frontier (open tile next to an unknown one, or one the goal has not been ruled out from).
        None if nothing is left to explore."""
        pos, _ = self.get_pose()
        start = self._tile(pos)
        known, n = self.known, self.size
        if not self._in_map(start):
            return None

        def passable(t):
            return 0 <= t[0] < n and 0 <= t[1] < n and known[t] == 0

        def unknown_nbrs(t):
            return [(t[0] + dr, t[1] + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= t[0] + dr < n and 0 <= t[1] + dc < n and known[t[0] + dr, t[1] + dc] == -1]

        if self.goal_seen_cell is not None:
            path = self._bfs(start, passable, lambda t: t == self.goal_seen_cell)
            if path is not None:
                return path
        path = self._bfs(start, passable, lambda t: not self.searched[t] or bool(unknown_nbrs(t)))
        if path is not None and len(path) == 1:   # standing on the frontier: step into the unknown tile
            nb = unknown_nbrs(start)
            if nb:
                path.append(nb[0])
        return path

    def waypoint(self):
        """World (x, y) centre of the next tile on the route from `_route`, or None."""
        path = self._route()
        if path is None or len(path) < 2:
            return None
        return np.array([path[1][1] + self.offset, path[1][0] + self.offset], dtype=float)

    def _crop(self, pos, yaw):
        """CROP x CROP window of the map in the agent's frame (heading snapped to a world axis):
        rows run far -> near ahead, columns left -> right. +1 open, 0 unknown, -1 wall / outside."""
        k = int(round(yaw / (np.pi / 2))) % 4
        fx, fy = self.CARDINALS[k]          # forward in world (x, y) = (col, row) steps
        lx, ly = -fy, fx                    # left
        r, c = self._tile(pos)
        h = self.CROP // 2
        out = np.zeros((self.CROP, self.CROP), dtype=np.float32)
        for i in range(self.CROP):
            ahead = h - i
            for j in range(self.CROP):
                left = h - j
                t = (int(r + ahead * fy + left * ly), int(c + ahead * fx + left * lx))
                if not self._in_map(t):
                    out[i, j] = -1.0
                else:
                    v = self.known[t]
                    out[i, j] = 0.0 if v == -1 else (1.0 if v == 0 else -1.0)
        return out.ravel()

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

        # Event-based reward only: no distance/heading shaping, no path hints.
        pos = p.getBasePositionAndOrientation(self.agent_id)[0]
        agent_pos = np.array(pos[:2])
        reward = -0.01  # step penalty

        # Entering a tile: bonus the first time, small penalty when coming back (needs no map knowledge)
        cell = self._cell(agent_pos)
        if cell != self.last_cell:
            self.last_cell = cell
            n = self.visits.get(cell, 0)
            self.visits[cell] = n + 1
            reward += self.NOVELTY_BONUS if n == 0 else -self.REVISIT_PENALTY

        obs = self._get_obs()

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

        return obs, reward, terminated, truncated, {"goal": terminated, "hit_obstacle": hit_obstacle}

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
        hit_ids = []
        for i in range(self.num_lidar_rays):
            angle = yaw + (i - self.num_lidar_rays/2) * (np.pi / self.num_lidar_rays) * 2
            c, s = np.cos(angle), np.sin(angle)
            r0 = self.agent_radius + 0.01  # start outside own body
            ray_from = [pos[0] + r0 * c, pos[1] + r0 * s, pos[2]]
            ray_to = [pos[0] + self.lidar_range * c, pos[1] + self.lidar_range * s, pos[2]]
            hit = p.rayTest(ray_from, ray_to)[0]
            hit_ids.append(hit[0])
            lidar_readings.append(r0 + hit[2] * (self.lidar_range - r0))
        # Sensor noise
        lidar_readings = np.clip(np.array(lidar_readings)
                                 + self.LIDAR_NOISE * self.np_random.standard_normal(self.num_lidar_rays),
                                 0.0, self.lidar_range)

        # Visit counts of the 4 neighbouring tiles relative to heading: ahead, left, behind, right
        cx, cy = self._cell(agent_pos)
        visits = []
        for k in range(4):
            a = yaw + k * np.pi / 2
            n = self.visits.get((cx + int(round(np.cos(a))), cy + int(round(np.sin(a)))), 0)
            visits.append(min(n, 3) / 3)

        # Own map + route cue
        self._update_map(pos, yaw, lidar_readings, hit_ids)
        path = self._route()
        cue = [0.0, 0.0, 0.0, 0.0]
        if path is not None and len(path) >= 2:
            nx, ny = path[1][1] + self.offset, path[1][0] + self.offset
            err = math.atan2(ny - agent_pos[1], nx - agent_pos[0]) - yaw
            cue = [1.0, (len(path) - 1) / self.size, np.sin(err), np.cos(err)]

        time_left = 1.0 - self.step_count / self.max_steps
        obs = np.concatenate([[time_left], goal_obs, visits, lidar_readings, cue, self._crop(agent_pos, yaw)])
        return obs.astype(np.float32)

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass
