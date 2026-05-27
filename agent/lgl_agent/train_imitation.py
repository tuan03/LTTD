from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.lgl_agent.agent import Agent, PolicyNet, encode_obs
from agent import BoxFarmerAgent, GeniusRuleAgent, RandomAgent, SmarterRuleAgent, TacticalRuleAgent
from engine import BomberEnv


def make_opponent(agent_id: int, name: str):
    if name == "random":
        return RandomAgent(agent_id)
    if name == "smarter":
        return SmarterRuleAgent(agent_id)
    if name == "genius":
        return GeniusRuleAgent(agent_id)
    if name == "box_farmer":
        return BoxFarmerAgent(agent_id)
    return TacticalRuleAgent(agent_id)


def collect_dataset(args):
    rng = random.Random(args.seed)
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    teacher = Agent(args.agent_id)
    maps = []
    auxes = []
    actions_out = []
    opponent_names = ["tactical", "genius", "smarter", "box_farmer", "random"]

    for ep in tqdm(range(args.episodes), desc="collect"):
        obs = env.reset(seed=args.seed + ep)
        opponents = []
        for pid in range(4):
            if pid == args.agent_id:
                opponents.append(teacher)
            else:
                opponents.append(make_opponent(pid, rng.choice(opponent_names)))

        for _ in range(args.max_steps):
            action = int(teacher.act(obs))
            map_x, aux = encode_obs(obs, args.agent_id)
            maps.append(map_x)
            auxes.append(aux)
            actions_out.append(action)

            step_actions = []
            for pid, agent in enumerate(opponents):
                try:
                    step_actions.append(int(agent.act(obs)))
                except Exception:
                    step_actions.append(0)
            obs, terminated, truncated = env.step(step_actions)
            if terminated or truncated or int(obs["players"][args.agent_id][2]) != 1:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        map=np.asarray(maps, dtype=np.float32),
        aux=np.asarray(auxes, dtype=np.float32),
        action=np.asarray(actions_out, dtype=np.int64),
    )
    print(f"saved {len(actions_out)} samples to {output}")


def train_bc(args):
    data = np.load(args.dataset)
    map_x = torch.from_numpy(data["map"]).float()
    aux_x = torch.from_numpy(data["aux"]).float()
    y = torch.from_numpy(data["action"]).long()

    n = len(y)
    order = torch.randperm(n)
    split = int(n * 0.9)
    train_idx = order[:split]
    val_idx = order[split:]

    train_ds = TensorDataset(map_x[train_idx], aux_x[train_idx], y[train_idx])
    val_ds = TensorDataset(map_x[val_idx], aux_x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = PolicyNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for mb_map, mb_aux, mb_y in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            mb_map = mb_map.to(device)
            mb_aux = mb_aux.to(device)
            mb_y = mb_y.to(device)
            logits = model(mb_map, mb_aux)
            loss = loss_fn(logits, mb_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(mb_y)
            correct += int((logits.argmax(1) == mb_y).sum().item())
            seen += len(mb_y)

        val_acc = evaluate_accuracy(model, val_loader, device)
        print(
            f"epoch={epoch + 1} loss={total_loss / max(1, seen):.4f} "
            f"train_acc={correct / max(1, seen):.3f} val_acc={val_acc:.3f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict()}, output)
    print(f"saved policy to {output}")


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for mb_map, mb_aux, mb_y in loader:
            mb_map = mb_map.to(device)
            mb_aux = mb_aux.to(device)
            mb_y = mb_y.to(device)
            pred = model(mb_map, mb_aux).argmax(1)
            correct += int((pred == mb_y).sum().item())
            seen += len(mb_y)
    return correct / max(1, seen)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--episodes", type=int, default=200)
    collect.add_argument("--max_steps", type=int, default=500)
    collect.add_argument("--seed", type=int, default=86)
    collect.add_argument("--agent_id", type=int, default=0)
    collect.add_argument("--output", type=str, default="agent/lgl_agent/data/imitation_dataset.npz")

    train = sub.add_parser("train")
    train.add_argument("--dataset", type=str, default="agent/lgl_agent/data/imitation_dataset.npz")
    train.add_argument("--output", type=str, default="agent/lgl_agent/lgl_policy.pth")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    if args.cmd == "collect":
        collect_dataset(args)
    else:
        train_bc(args)


if __name__ == "__main__":
    main()
