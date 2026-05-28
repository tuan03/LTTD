from __future__ import annotations

from collections import deque
from pathlib import Path
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # Submission still works as a pure rule agent if torch is unavailable.
    torch = None
    nn = None


GRASS = 0
WALL = 1
BOX = 2
ITEM_RADIUS = 3
ITEM_CAPACITY = 4
BOMB_TIMER = 7
MODEL_NAME = "lgl_policy.pth"


class PolicyNet(nn.Module if nn is not None else object):
    def __init__(self, map_shape=(13, 13), aux_dim=16, num_actions=6):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        conv_dim = 64 * map_shape[0] * map_shape[1]
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(conv_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, map_x, aux_x):
        spatial = self.map_encoder(map_x).reshape(map_x.shape[0], -1)
        aux = self.aux_encoder(aux_x)
        return self.head(torch.cat([spatial, aux], dim=1))


def _as_bombs_array(bombs):
    arr = np.asarray(bombs)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int16)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.zeros((arr.shape[0], 4), dtype=np.int16)
    cols = min(arr.shape[1], 4)
    out[:, :cols] = arr[:, :cols]
    if cols < 3:
        out[:, 2] = BOMB_TIMER
    return out


def encode_obs(obs, agent_id):
    grid = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    bombs = _as_bombs_array(obs.get("bombs", []))
    h, w = grid.shape
    me = players[int(agent_id)]
    my_x, my_y = int(me[0]), int(me[1])

    channels = [
        (grid == GRASS).astype(np.float32),
        (grid == WALL).astype(np.float32),
        (grid == BOX).astype(np.float32),
        (grid == ITEM_RADIUS).astype(np.float32),
        (grid == ITEM_CAPACITY).astype(np.float32),
    ]

    my_pos = np.zeros((h, w), dtype=np.float32)
    enemy_pos = np.zeros((h, w), dtype=np.float32)
    if int(me[2]) == 1:
        my_pos[my_x, my_y] = 1.0
    for pid, p in enumerate(players):
        if pid != int(agent_id) and int(p[2]) == 1:
            enemy_pos[int(p[0]), int(p[1])] = 1.0

    bomb_timer = np.zeros((h, w), dtype=np.float32)
    own_bomb = np.zeros((h, w), dtype=np.float32)
    enemy_bomb = np.zeros((h, w), dtype=np.float32)
    danger = np.zeros((h, w), dtype=np.float32)
    urgent = np.zeros((h, w), dtype=np.float32)
    for bx, by, timer, owner in bombs:
        bx, by, timer, owner = int(bx), int(by), int(timer), int(owner)
        if 0 <= bx < h and 0 <= by < w:
            bomb_timer[bx, by] = max(bomb_timer[bx, by], timer / float(BOMB_TIMER))
            if owner == int(agent_id):
                own_bomb[bx, by] = 1.0
            else:
                enemy_bomb[bx, by] = 1.0
            radius = 1 + int(players[owner][4]) if 0 <= owner < len(players) else 2
            for tx, ty in _blast_tiles(grid, bx, by, radius):
                danger[tx, ty] = max(danger[tx, ty], (BOMB_TIMER + 1 - timer) / BOMB_TIMER)
                if timer <= 2:
                    urgent[tx, ty] = 1.0

    map_x = np.stack(
        [*channels, my_pos, enemy_pos, bomb_timer, own_bomb, enemy_bomb, danger, urgent],
        axis=0,
    ).astype(np.float32)

    alive_enemies = [p for i, p in enumerate(players) if i != int(agent_id) and int(p[2]) == 1]
    enemy_d = min(
        [abs(my_x - int(p[0])) + abs(my_y - int(p[1])) for p in alive_enemies],
        default=24,
    )
    aux = np.array(
        [
            my_x / 12.0,
            my_y / 12.0,
            float(me[3]) / 5.0,
            float(me[4]) / 4.0,
            len(alive_enemies) / 3.0,
            enemy_d / 24.0,
            float(np.sum(grid == BOX)) / float(h * w),
            float(np.sum(grid == ITEM_RADIUS)) / 10.0,
            float(np.sum(grid == ITEM_CAPACITY)) / 10.0,
            min(len(bombs), 10) / 10.0,
            1.0 if int(me[2]) == 1 else 0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    return map_x, aux


def _blast_tiles(grid, bx, by, radius):
    h, w = grid.shape
    tiles = {(int(bx), int(by))}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, int(radius) + 1):
            x, y = int(bx) + dx * r, int(by) + dy * r
            if not (0 <= x < h and 0 <= y < w):
                break
            cell = int(grid[x, y])
            if cell == WALL:
                break
            tiles.add((x, y))
            if cell == BOX:
                break
    return tiles


class Agent:
    team_id = "lgl_agent"

    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self._rng = random.Random(20260527 + self.agent_id)
        self.device = torch.device("cpu") if torch is not None else None
        self.model = None
        self.model_path = Path(__file__).parent / MODEL_NAME
        self._load_model_if_present()

    def _load_model_if_present(self):
        if torch is None or not self.model_path.exists():
            return
        try:
            checkpoint = torch.load(str(self.model_path), map_location="cpu")
            self.model = PolicyNet()
            state = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state)
            self.model.eval()
        except Exception:
            self.model = None

    def act(self, obs: dict) -> int:
        players = obs.get("players")
        if players is None or self.agent_id >= len(players):
            return 0
        if int(players[self.agent_id][2]) != 1:
            return 0

        try:
            context = self._build_context(obs)
            safe_actions = self._safe_actions(context)
            rule_action = self._rule_action(context, safe_actions)
            if self.model is None:
                return int(rule_action)

            action = self._model_action(obs, safe_actions)
            if action is None:
                return int(rule_action)
            return int(action)
        except Exception:
            return 0

    def _model_action(self, obs, safe_actions):
        if not safe_actions:
            return None
        map_x, aux = encode_obs(obs, self.agent_id)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(map_x).unsqueeze(0),
                torch.from_numpy(aux).unsqueeze(0),
            )[0].cpu().numpy()
        masked = np.full(6, -1e9, dtype=np.float32)
        for a in safe_actions:
            masked[int(a)] = logits[int(a)]
        return int(masked.argmax())

    def _build_context(self, obs):
        grid = np.asarray(obs["map"])
        players = np.asarray(obs["players"])
        bombs = _as_bombs_array(obs.get("bombs", []))
        me = players[self.agent_id]
        my_pos = (int(me[0]), int(me[1]))
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and int(p[2]) == 1
        ]
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        blocked = set(bomb_positions)
        blocked.discard(my_pos)
        danger_time = self._danger_time_map(grid, players, bombs)
        return {
            "grid": grid,
            "players": players,
            "bombs": bombs,
            "me": me,
            "my_pos": my_pos,
            "enemies": enemies,
            "bomb_positions": bomb_positions,
            "blocked": blocked,
            "danger_time": danger_time,
            "radius": 1 + int(me[4]),
            "bombs_left": int(me[3]),
        }

    def _rule_action(self, c, safe_actions):
        my_pos = c["my_pos"]
        danger_timer = int(c["danger_time"][my_pos[0], my_pos[1]])
        danger_now = self._danger_at(c["danger_time"], my_pos, 0)

        if danger_now or danger_timer < 99:
            escape = self._best_escape_action(c, safe_actions)
            return escape if escape is not None else 0

        if c["bombs_left"] > 0 and 5 in safe_actions:
            boxes = self._count_boxes_in_blast(c["grid"], my_pos, c["radius"])
            hit_enemy = self._can_hit_enemy(c["grid"], my_pos, c["enemies"], c["radius"])
            if hit_enemy or boxes >= 2:
                return 5

        item_move = self._move_to_targets(c, self._item_targets(c), safe_actions)
        if item_move is not None:
            return item_move

        if c["bombs_left"] > 0 and 5 in safe_actions:
            boxes = self._count_boxes_in_blast(c["grid"], my_pos, c["radius"])
            if boxes >= 1 and self._nearest_enemy_distance(c) > 3:
                return 5

        box_move = self._move_to_targets(c, self._box_bomb_spots(c), safe_actions)
        if box_move is not None:
            return box_move

        enemy_move = self._move_to_targets(c, set(c["enemies"]), safe_actions)
        if enemy_move is not None:
            return enemy_move

        non_stop = [a for a in safe_actions if a != 0]
        return int(self._rng.choice(non_stop or safe_actions or [0]))

    def _safe_actions(self, c):
        valid = self._valid_actions(c)
        safe = []
        for action in valid:
            npos = self._next_pos(c["my_pos"], action)
            if self._danger_at(c["danger_time"], npos, 1):
                continue
            if action == 5:
                if not self._can_escape_after_bomb(c):
                    continue
            elif not self._has_future_escape(c, npos, start_time=1, require_clear_now=False):
                continue
            safe.append(action)
        if safe:
            return safe
        non_blast = [a for a in valid if not self._danger_at(c["danger_time"], self._next_pos(c["my_pos"], a), 0)]
        return non_blast or [0]

    def _valid_actions(self, c):
        actions = [0]
        grid = c["grid"]
        for action in (1, 2, 3, 4):
            x, y = self._next_pos(c["my_pos"], action)
            if self._passable(grid, x, y) and (x, y) not in c["blocked"]:
                actions.append(action)
        if c["bombs_left"] > 0 and c["my_pos"] not in c["bomb_positions"]:
            actions.append(5)
        return actions

    def _next_pos(self, pos, action):
        if action == 5:
            return pos
        dx, dy = self.MOVES.get(int(action), (0, 0))
        return int(pos[0]) + dx, int(pos[1]) + dy

    def _passable(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and int(grid[x, y]) in (
            GRASS,
            ITEM_RADIUS,
            ITEM_CAPACITY,
        )

    def _danger_time_map(self, grid, players, bombs):
        h, w = grid.shape
        danger = np.full((h, w), 99, dtype=np.int16)
        bomb_infos = []
        for bx, by, timer, owner in bombs:
            bx, by, timer, owner = int(bx), int(by), max(0, int(timer)), int(owner)
            radius = 1 + int(players[owner][4]) if 0 <= owner < len(players) else 2
            bomb_infos.append(
                {
                    "pos": (bx, by),
                    "time": timer,
                    "tiles": _blast_tiles(grid, bx, by, radius),
                }
            )

        changed = True
        while changed:
            changed = False
            for src in bomb_infos:
                for dst in bomb_infos:
                    if src is dst:
                        continue
                    if dst["pos"] in src["tiles"] and src["time"] < dst["time"]:
                        dst["time"] = src["time"]
                        changed = True

        for bomb in bomb_infos:
            for tx, ty in bomb["tiles"]:
                danger[tx, ty] = min(danger[tx, ty], bomb["time"])
        return danger

    def _danger_at(self, danger_time, pos, arrive_time):
        t = int(danger_time[int(pos[0]), int(pos[1])])
        return t <= int(arrive_time) + 1

    def _has_future_escape(self, c, start, start_time=0, max_depth=8, require_clear_now=True):
        q = deque([(start, int(start_time))])
        seen = {(start, int(start_time))}
        while q:
            pos, t = q.popleft()
            if (
                t > start_time
                and not self._danger_at(c["danger_time"], pos, t)
                and (not require_clear_now or int(c["danger_time"][pos[0], pos[1]]) >= 99)
            ):
                return True
            if t - start_time >= max_depth:
                continue
            for action in (0, 1, 2, 3, 4):
                npos = self._next_pos(pos, action)
                if action != 0:
                    if not self._passable(c["grid"], npos[0], npos[1]) or npos in c["blocked"]:
                        continue
                nt = t + 1
                key = (npos, nt)
                if key in seen or self._danger_at(c["danger_time"], npos, nt):
                    continue
                seen.add(key)
                q.append((npos, nt))
        return False

    def _best_escape_action(self, c, safe_actions):
        best = None
        best_score = -10**9
        for action in safe_actions:
            if action == 5:
                continue
            npos = self._next_pos(c["my_pos"], action)
            score = int(c["danger_time"][npos[0], npos[1]]) * 3
            score += self._open_neighbor_count(c, npos)
            score -= self._nearest_bomb_distance(c, npos)
            if action == 0:
                score -= 5
            if score > best_score:
                best_score = score
                best = action
        return best

    def _can_escape_after_bomb(self, c):
        my_pos = c["my_pos"]
        my_blast = _blast_tiles(c["grid"], my_pos[0], my_pos[1], c["radius"])
        simulated = dict(c)
        danger = np.array(c["danger_time"], copy=True)
        for tx, ty in my_blast:
            danger[tx, ty] = min(danger[tx, ty], BOMB_TIMER)
        simulated["danger_time"] = danger
        simulated["blocked"] = set(c["blocked"]) | {my_pos}
        return self._escape_after_new_bomb(simulated, my_pos, my_blast)

    def _escape_after_new_bomb(self, c, start, new_blast):
        q = deque([(start, 0, None)])
        seen = {(start, 0)}
        while q:
            pos, t, first = q.popleft()
            if t > 0 and pos not in new_blast and not self._danger_at(c["danger_time"], pos, t):
                return True
            if t >= BOMB_TIMER - 1:
                continue
            for action in (1, 2, 3, 4, 0):
                if t == 0 and action == 0:
                    continue
                npos = self._next_pos(pos, action)
                if action != 0:
                    if not self._passable(c["grid"], npos[0], npos[1]):
                        continue
                    if npos in c["blocked"]:
                        continue
                nt = t + 1
                if self._danger_at(c["danger_time"], npos, nt):
                    continue
                key = (npos, nt)
                if key in seen:
                    continue
                seen.add(key)
                q.append((npos, nt, action if first is None else first))
        return False

    def _move_to_targets(self, c, targets, safe_actions):
        if not targets:
            return None
        safe_first = set(a for a in safe_actions if a in (1, 2, 3, 4))
        q = deque([(c["my_pos"], None, 0)])
        seen = {c["my_pos"]}
        while q:
            pos, first, dist = q.popleft()
            if pos in targets and first is not None:
                return first
            if dist >= 12:
                continue
            for action in (1, 2, 3, 4):
                if first is None and action not in safe_first:
                    continue
                npos = self._next_pos(pos, action)
                if npos in seen:
                    continue
                if not self._passable(c["grid"], npos[0], npos[1]) or npos in c["blocked"]:
                    continue
                if self._danger_at(c["danger_time"], npos, dist + 1):
                    continue
                seen.add(npos)
                q.append((npos, action if first is None else first, dist + 1))
        return None

    def _item_targets(self, c):
        grid = c["grid"]
        preferred = []
        values = [ITEM_RADIUS, ITEM_CAPACITY]
        if c["bombs_left"] <= 1:
            values = [ITEM_CAPACITY, ITEM_RADIUS]
        for val in values:
            tiles = {
                (x, y)
                for x in range(grid.shape[0])
                for y in range(grid.shape[1])
                if int(grid[x, y]) == val and not self._danger_at(c["danger_time"], (x, y), 0)
            }
            if tiles:
                preferred.extend(tiles)
        return set(preferred)

    def _box_bomb_spots(self, c):
        spots = set()
        grid = c["grid"]
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if int(grid[x, y]) != BOX:
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in c["blocked"]:
                        spots.add((nx, ny))
        return spots

    def _can_hit_enemy(self, grid, my_pos, enemies, radius):
        mx, my = my_pos
        for ex, ey in enemies:
            if mx == ex and abs(ey - my) <= radius and self._line_clear(grid, my_pos, (ex, ey)):
                return True
            if my == ey and abs(ex - mx) <= radius and self._line_clear(grid, my_pos, (ex, ey)):
                return True
        return False

    def _line_clear(self, grid, src, dst):
        sx, sy = src
        dx, dy = dst
        if sx == dx:
            step = 1 if dy > sy else -1
            for y in range(sy + step, dy, step):
                if int(grid[sx, y]) in (WALL, BOX):
                    return False
            return True
        if sy == dy:
            step = 1 if dx > sx else -1
            for x in range(sx + step, dx, step):
                if int(grid[x, sy]) in (WALL, BOX):
                    return False
            return True
        return False

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for x, y in _blast_tiles(grid, pos[0], pos[1], radius) if int(grid[x, y]) == BOX)

    def _nearest_enemy_distance(self, c):
        x, y = c["my_pos"]
        return min((abs(x - ex) + abs(y - ey) for ex, ey in c["enemies"]), default=99)

    def _nearest_bomb_distance(self, c, pos):
        x, y = pos
        return min((abs(x - int(b[0])) + abs(y - int(b[1])) for b in c["bombs"]), default=8)

    def _open_neighbor_count(self, c, pos):
        count = 0
        for action in (1, 2, 3, 4):
            npos = self._next_pos(pos, action)
            if self._passable(c["grid"], npos[0], npos[1]) and npos not in c["blocked"]:
                count += 1
        return count
