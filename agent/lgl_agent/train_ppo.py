from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.lgl_agent.agent import Agent, PolicyNet, encode_obs
from agent import BoxFarmerAgent, GeniusRuleAgent, RandomAgent, SmarterRuleAgent, TacticalRuleAgent
from engine import BomberEnv


class ValueNet(nn.Module):
    def __init__(self, map_shape=(13, 13), aux_dim=16):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.aux_encoder = nn.Sequential(nn.Linear(aux_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(64 * map_shape[0] * map_shape[1] + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, map_x, aux_x):
        spatial = self.map_encoder(map_x).reshape(map_x.shape[0], -1)
        aux = self.aux_encoder(aux_x)
        return self.head(torch.cat([spatial, aux], dim=1)).squeeze(-1)


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


def masked_distribution(policy, map_x, aux_x, mask, device):
    logits = policy(
        torch.from_numpy(map_x).unsqueeze(0).to(device),
        torch.from_numpy(aux_x).unsqueeze(0).to(device),
    )[0]
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
    logits = logits.masked_fill(~mask_t, -1e9)
    return torch.distributions.Categorical(logits=logits)


def safe_mask(agent: Agent, obs):
    c = agent._build_context(obs)
    actions = agent._safe_actions(c)
    mask = np.zeros(6, dtype=np.bool_)
    for action in actions:
        mask[int(action)] = True
    if not mask.any():
        mask[0] = True
    return mask


def shaped_reward(prev_obs, obs, agent_id):
    if prev_obs is None:
        return 0.0
    prev_p = prev_obs["players"]
    curr_p = obs["players"]
    reward = -0.003
    if int(prev_p[agent_id][2]) == 1 and int(curr_p[agent_id][2]) == 0:
        return -2.0
    prev_enemy = sum(1 for i, p in enumerate(prev_p) if i != agent_id and int(p[2]) == 1)
    curr_enemy = sum(1 for i, p in enumerate(curr_p) if i != agent_id and int(p[2]) == 1)
    if curr_enemy < prev_enemy:
        reward += 1.0 * (prev_enemy - curr_enemy)
    if curr_enemy == 0 and prev_enemy > 0:
        reward += 2.0
    if int(curr_p[agent_id][3]) > int(prev_p[agent_id][3]) or int(curr_p[agent_id][4]) > int(prev_p[agent_id][4]):
        reward += 0.12
    if int(prev_p[agent_id][0]) == int(curr_p[agent_id][0]) and int(prev_p[agent_id][1]) == int(curr_p[agent_id][1]):
        reward -= 0.01
    return float(reward)


def discounted_returns(rewards, dones, gamma):
    out = []
    ret = 0.0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        ret = float(reward) + gamma * ret * (1.0 - float(done))
        out.append(ret)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--load_model", type=str, default="agent/lgl_agent/lgl_policy.pth")
    parser.add_argument("--output", type=str, default="agent/lgl_agent/lgl_policy.pth")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--update_every", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    policy = PolicyNet().to(device)
    load_path = Path(args.load_model)
    if load_path.exists():
        ckpt = torch.load(load_path, map_location=device)
        policy.load_state_dict(ckpt.get("model_state_dict", ckpt))
    value = ValueNet().to(device)
    opt = torch.optim.AdamW(list(policy.parameters()) + list(value.parameters()), lr=args.lr, weight_decay=1e-5)
    selector = Agent(args.agent_id)
    opponent_names = ["tactical", "genius", "smarter", "box_farmer", "random"]

    buffer = []
    wins = []
    for ep in tqdm(range(args.episodes), desc="ppo"):
        obs = env.reset(seed=args.seed + ep)
        opponents = []
        for pid in range(4):
            if pid == args.agent_id:
                opponents.append(None)
            else:
                opponents.append(make_opponent(pid, random.choice(opponent_names)))

        ep_reward = 0.0
        for _ in range(args.max_steps):
            map_x, aux = encode_obs(obs, args.agent_id)
            mask = safe_mask(selector, obs)
            dist = masked_distribution(policy, map_x, aux, mask, device)
            action_t = dist.sample()
            action = int(action_t.item())
            logp = float(dist.log_prob(action_t).detach().cpu().item())
            val = float(
                value(
                    torch.from_numpy(map_x).unsqueeze(0).to(device),
                    torch.from_numpy(aux).unsqueeze(0).to(device),
                )[0].detach().cpu().item()
            )

            actions = []
            for pid, opponent in enumerate(opponents):
                if pid == args.agent_id:
                    actions.append(action)
                else:
                    try:
                        actions.append(int(opponent.act(obs)))
                    except Exception:
                        actions.append(0)

            next_obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated or int(next_obs["players"][args.agent_id][2]) != 1)
            reward = shaped_reward(obs, next_obs, args.agent_id)
            ep_reward += reward
            buffer.append((map_x, aux, mask, action, logp, val, reward, done))
            obs = next_obs
            if done:
                break

        wins.append(1 if int(obs["players"][args.agent_id][2]) == 1 and sum(int(p[2]) for p in obs["players"]) <= 1 else 0)
        if (ep + 1) % args.update_every == 0 and buffer:
            update(policy, value, opt, buffer, args, device)
            buffer.clear()
            print(f"episode={ep + 1} reward={ep_reward:.2f} recent_win={np.mean(wins[-50:]):.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": policy.cpu().state_dict()}, output)
    print(f"saved PPO-tuned policy to {output}")


def update(policy, value, opt, buffer, args, device):
    maps = torch.tensor(np.asarray([b[0] for b in buffer], dtype=np.float32), device=device)
    auxes = torch.tensor(np.asarray([b[1] for b in buffer], dtype=np.float32), device=device)
    masks = torch.tensor(np.asarray([b[2] for b in buffer], dtype=np.bool_), device=device)
    actions = torch.tensor([b[3] for b in buffer], dtype=torch.long, device=device)
    old_logp = torch.tensor([b[4] for b in buffer], dtype=torch.float32, device=device)
    old_values = torch.tensor([b[5] for b in buffer], dtype=torch.float32, device=device)
    rewards = [b[6] for b in buffer]
    dones = [b[7] for b in buffer]
    returns = discounted_returns(rewards, dones, args.gamma).to(device)
    adv = returns - old_values
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    for _ in range(args.ppo_epochs):
        logits = policy(maps, auxes).masked_fill(~masks, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ratio = torch.exp(logp - old_logp)
        policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
        values = value(maps, auxes)
        value_loss = nn.functional.mse_loss(values, returns)
        entropy = dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), 0.5)
        opt.step()


if __name__ == "__main__":
    main()
