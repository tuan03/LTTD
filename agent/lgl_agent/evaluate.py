from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.lgl_agent.agent import Agent
from agent import BoxFarmerAgent, GeniusRuleAgent, RandomAgent, SmarterRuleAgent, TacticalRuleAgent
from engine import BomberEnv


def make_agent(agent_id, name):
    if name == "lgl":
        return Agent(agent_id)
    if name == "random":
        return RandomAgent(agent_id)
    if name == "smarter":
        return SmarterRuleAgent(agent_id)
    if name == "genius":
        return GeniusRuleAgent(agent_id)
    if name == "box_farmer":
        return BoxFarmerAgent(agent_id)
    return TacticalRuleAgent(agent_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument(
        "--opponents",
        nargs=3,
        default=["tactical", "genius", "smarter"],
        choices=["random", "smarter", "genius", "box_farmer", "tactical"],
    )
    args = parser.parse_args()

    random.seed(args.seed)
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    wins = 0
    survived = 0
    steps = []

    progress = tqdm(range(args.episodes), desc="evaluate", unit="episode")
    for ep in progress:
        obs = env.reset(seed=args.seed + ep)
        agents = [None] * 4
        agents[args.agent_id] = make_agent(args.agent_id, "lgl")
        opp_iter = iter(args.opponents)
        for pid in range(4):
            if pid != args.agent_id:
                agents[pid] = make_agent(pid, next(opp_iter))

        step = 0
        while step < args.max_steps:
            actions = []
            for agent in agents:
                try:
                    actions.append(int(agent.act(obs)))
                except Exception:
                    actions.append(0)
            obs, terminated, truncated = env.step(actions)
            step += 1
            if terminated or truncated:
                break

        alive = [int(p[2]) for p in obs["players"]]
        if alive[args.agent_id]:
            survived += 1
        if alive[args.agent_id] and sum(alive) == 1:
            wins += 1
        steps.append(step)
        completed = ep + 1
        progress.set_postfix(
            win=f"{wins / completed:.3f}",
            survived=f"{survived / completed:.3f}",
            avg_steps=f"{float(np.mean(steps)):.1f}",
        )

    print("=== LGL evaluation ===")
    print(f"episodes: {args.episodes}")
    print(f"opponents: {args.opponents}")
    print(f"wins: {wins} ({wins / args.episodes:.3f})")
    print(f"survived: {survived} ({survived / args.episodes:.3f})")
    print(f"avg_steps: {float(np.mean(steps)):.1f}")


if __name__ == "__main__":
    main()
