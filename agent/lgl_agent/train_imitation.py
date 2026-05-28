from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import random
import shutil
import sys
import time

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


def save_dataset(output_path, maps, auxes, actions_out, episodes_out=None, compressed=False):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    payload = {
        "map": np.asarray(maps, dtype=np.float32),
        "aux": np.asarray(auxes, dtype=np.float32),
        "action": np.asarray(actions_out, dtype=np.int64),
    }
    if episodes_out is not None:
        payload["episode"] = np.asarray(episodes_out, dtype=np.int64)
    saver = np.savez_compressed if compressed else np.savez
    with tmp_output.open("wb") as f:
        saver(f, **payload)
    tmp_output.replace(output)


def collect_samples(args, seed, episodes, max_samples, flush_path=None, worker_id=0, progress_queue=None):
    rng = random.Random(seed)
    env = BomberEnv(max_steps=args.max_steps, seed=seed)
    teacher = Agent(args.agent_id)
    if args.rule_only_teacher:
        teacher.model = None
    maps = []
    auxes = []
    actions_out = []
    episodes_out = []
    episodes_out = []
    opponent_names = ["tactical", "genius", "smarter", "box_farmer", "random"]
    next_flush = args.flush_every if args.flush_every > 0 and flush_path is not None else None
    next_episode_flush = args.flush_episodes if args.flush_episodes > 0 and flush_path is not None else None
    next_progress = args.progress_every if args.progress_every > 0 and progress_queue is not None else None
    last_progress = 0

    for ep in range(episodes):
        episode_id = seed + ep
        obs = env.reset(seed=seed + ep)
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
            episodes_out.append(episode_id)

            if next_flush is not None and len(actions_out) >= next_flush:
                save_dataset(flush_path, maps, auxes, actions_out, episodes_out, compressed=args.compressed)
                next_flush += args.flush_every
            if next_progress is not None and len(actions_out) >= next_progress:
                progress_queue.put(("sample", worker_id, len(actions_out) - last_progress))
                last_progress = len(actions_out)
                next_progress += args.progress_every

            step_actions = []
            for pid, agent in enumerate(opponents):
                try:
                    step_actions.append(int(agent.act(obs)))
                except Exception:
                    step_actions.append(0)
            obs, terminated, truncated = env.step(step_actions)
            if terminated or truncated or int(obs["players"][args.agent_id][2]) != 1:
                break
            if max_samples and len(actions_out) >= max_samples:
                break
        if progress_queue is not None:
            progress_queue.put(("episode", worker_id, 1))
        if next_episode_flush is not None and ep + 1 >= next_episode_flush:
            save_dataset(flush_path, maps, auxes, actions_out, episodes_out, compressed=args.compressed)
            next_episode_flush += args.flush_episodes
        if max_samples and len(actions_out) >= max_samples:
            break

    if progress_queue is not None and len(actions_out) > last_progress:
        progress_queue.put(("sample", worker_id, len(actions_out) - last_progress))
    return maps, auxes, actions_out, episodes_out


def collect_worker(payload):
    args, worker_id, episodes, max_samples, shard_path, progress_queue = payload
    seed = int(args.seed) + worker_id * 1000003
    maps, auxes, actions_out, episodes_out = collect_samples(
        args,
        seed,
        episodes,
        max_samples,
        flush_path=shard_path,
        worker_id=worker_id,
        progress_queue=progress_queue,
    )
    save_dataset(shard_path, maps, auxes, actions_out, episodes_out, compressed=args.compressed)
    return str(shard_path), len(actions_out)


def merge_shards(shard_paths, output_path, max_samples=0, compressed=False):
    map_parts = []
    aux_parts = []
    action_parts = []
    episode_parts = []
    total = 0
    for shard_path in shard_paths:
        data = np.load(shard_path)
        remaining = max_samples - total if max_samples else len(data["action"])
        take = min(len(data["action"]), remaining)
        if take <= 0:
            break
        map_parts.append(data["map"][:take])
        aux_parts.append(data["aux"][:take])
        action_parts.append(data["action"][:take])
        if "episode" in data.files:
            episode_parts.append(data["episode"][:take])
        total += take
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    saver = np.savez_compressed if compressed else np.savez
    with tmp_output.open("wb") as f:
        saver(
            f,
            map=np.concatenate(map_parts, axis=0) if map_parts else np.empty((0, 12, 13, 13), dtype=np.float32),
            aux=np.concatenate(aux_parts, axis=0) if aux_parts else np.empty((0, 16), dtype=np.float32),
            action=np.concatenate(action_parts, axis=0) if action_parts else np.empty((0,), dtype=np.int64),
            episode=np.concatenate(episode_parts, axis=0) if episode_parts else np.empty((0,), dtype=np.int64),
        )
    tmp_output.replace(output)
    return total


def choose_torch_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        print("using device: cpu")
        return torch.device("cpu")
    try:
        device = torch.device("cuda")
        test = torch.ones((1,), device=device)
        _ = (test + 1).cpu().item()
        name = torch.cuda.get_device_name(0)
        print(f"using device: cuda ({name})")
        return device
    except Exception as exc:
        print(f"CUDA is visible but unusable, falling back to CPU: {exc}")
        return torch.device("cpu")


def collect_dataset_parallel(args):
    from concurrent.futures import ProcessPoolExecutor

    worker_count = max(1, int(args.workers))
    samples_per_worker = 0
    base_episodes = int(args.episodes) // worker_count
    extra_episodes = int(args.episodes) % worker_count
    worker_episodes = [base_episodes + (1 if worker_id < extra_episodes else 0) for worker_id in range(worker_count)]
    if args.max_samples:
        samples_per_worker = math.ceil(args.max_samples / worker_count)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / f"{output.stem}_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    with mp.Manager() as manager:
        progress_queue = manager.Queue()
        payloads = []
        for worker_id in range(worker_count):
            shard_path = shard_dir / f"shard_{worker_id:03d}.npz"
            payloads.append((args, worker_id, worker_episodes[worker_id], samples_per_worker, shard_path, progress_queue))

        progress = tqdm(total=int(args.episodes), desc="collect episodes", unit="episode")
        worker_samples = [0] * worker_count
        worker_done_episodes = [0] * worker_count

        shard_paths = []
        collected = 0
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(collect_worker, payload) for payload in payloads]
                pending = set(futures)
                while pending:
                    while True:
                        try:
                            event_type, worker_id, delta = progress_queue.get_nowait()
                        except Empty:
                            break
                        worker_id = int(worker_id)
                        delta = int(delta)
                        if event_type == "episode":
                            worker_done_episodes[worker_id] += delta
                            progress.update(delta)
                        else:
                            worker_samples[worker_id] += delta
                        progress.set_postfix(
                            samples=sum(worker_samples),
                            target=args.max_samples or "all",
                            eps="/".join(str(v) for v in worker_done_episodes),
                            sample_workers="/".join(str(v) for v in worker_samples),
                        )

                    done = [future for future in list(pending) if future.done()]
                    for future in done:
                        pending.remove(future)
                        shard_path, count = future.result()
                        shard_paths.append(shard_path)
                        collected += count
                        worker_id = int(Path(shard_path).stem.split("_")[-1])
                        if worker_samples[worker_id] < count:
                            delta = count - worker_samples[worker_id]
                            worker_samples[worker_id] = count
                        progress.set_postfix(
                            samples=sum(worker_samples),
                            target=args.max_samples or "all",
                            eps="/".join(str(v) for v in worker_done_episodes),
                            sample_workers="/".join(str(v) for v in worker_samples),
                            done=f"{worker_count - len(pending)}/{worker_count}",
                        )
                    if pending:
                        time.sleep(0.1)

                while True:
                    try:
                        event_type, worker_id, delta = progress_queue.get_nowait()
                    except Empty:
                        break
                    worker_id = int(worker_id)
                    delta = int(delta)
                    if event_type == "episode":
                        worker_done_episodes[worker_id] += delta
                        progress.update(delta)
                    else:
                        worker_samples[worker_id] += delta
                    progress.set_postfix(
                        samples=sum(worker_samples),
                        target=args.max_samples or "all",
                        eps="/".join(str(v) for v in worker_done_episodes),
                        sample_workers="/".join(str(v) for v in worker_samples),
                    )
        finally:
            progress.close()

    total = merge_shards(sorted(shard_paths), args.output, max_samples=args.max_samples, compressed=args.compressed)
    print(f"saved {total} samples to {args.output} from {worker_count} workers")
    print(f"shards kept in {shard_dir}")


def collect_dataset(args):
    if int(args.workers) > 1:
        collect_dataset_parallel(args)
        return

    rng = random.Random(args.seed)
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    teacher = Agent(args.agent_id)
    if args.rule_only_teacher:
        teacher.model = None
    maps = []
    auxes = []
    actions_out = []
    opponent_names = ["tactical", "genius", "smarter", "box_farmer", "random"]
    next_flush = args.flush_every if args.flush_every > 0 else None
    next_episode_flush = args.flush_episodes if args.flush_episodes > 0 else None

    progress = tqdm(range(args.episodes), desc="collect episodes", unit="episode")
    for ep in progress:
        episode_id = args.seed + ep
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
            episodes_out.append(episode_id)

            if next_flush is not None and len(actions_out) >= next_flush:
                save_dataset(args.output, maps, auxes, actions_out, episodes_out, compressed=args.compressed)
                print(f"checkpoint saved {len(actions_out)} samples to {args.output}")
                next_flush += args.flush_every

            step_actions = []
            for pid, agent in enumerate(opponents):
                try:
                    step_actions.append(int(agent.act(obs)))
                except Exception:
                    step_actions.append(0)
            obs, terminated, truncated = env.step(step_actions)
            if terminated or truncated or int(obs["players"][args.agent_id][2]) != 1:
                break
            if args.max_samples and len(actions_out) >= args.max_samples:
                break
        progress.set_postfix(samples=len(actions_out), target=args.max_samples or "all")
        if next_episode_flush is not None and ep + 1 >= next_episode_flush:
            save_dataset(args.output, maps, auxes, actions_out, episodes_out, compressed=args.compressed)
            print(f"episode checkpoint saved {len(actions_out)} samples after {ep + 1} episodes to {args.output}")
            next_episode_flush += args.flush_episodes
        if args.max_samples and len(actions_out) >= args.max_samples:
            break

    save_dataset(args.output, maps, auxes, actions_out, episodes_out, compressed=args.compressed)
    print(f"saved {len(actions_out)} samples to {args.output}")


def train_bc(args):
    data = np.load(args.dataset)
    map_x = torch.from_numpy(data["map"]).float()
    aux_x = torch.from_numpy(data["aux"]).float()
    y = torch.from_numpy(data["action"]).long()

    n = len(y)
    train_idx, val_idx = make_train_val_split(data, n, args.val_ratio, args.seed)
    print(f"split: train_samples={len(train_idx)} val_samples={len(val_idx)}")

    train_ds = TensorDataset(map_x[train_idx], aux_x[train_idx], y[train_idx])
    val_ds = TensorDataset(map_x[val_idx], aux_x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = choose_torch_device(args.cpu)
    model = PolicyNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0

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

        val_loss, val_acc = evaluate_validation(model, val_loader, loss_fn, device)
        improved = val_acc > best_val_acc + args.min_delta
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch + 1} loss={total_loss / max(1, seen):.4f} "
            f"train_acc={correct / max(1, seen):.3f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} best_val_acc={best_val_acc:.3f} stale={stale_epochs}"
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"early stopping after {epoch + 1} epochs")
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None and args.save_best:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        },
        output,
    )
    print(f"saved policy to {output}")


def evaluate_validation(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for mb_map, mb_aux, mb_y in loader:
            mb_map = mb_map.to(device)
            mb_aux = mb_aux.to(device)
            mb_y = mb_y.to(device)
            logits = model(mb_map, mb_aux)
            total_loss += float(loss_fn(logits, mb_y).item()) * len(mb_y)
            pred = logits.argmax(1)
            correct += int((pred == mb_y).sum().item())
            seen += len(mb_y)
    return total_loss / max(1, seen), correct / max(1, seen)


def make_train_val_split(data, n, val_ratio, seed):
    if "episode" not in data.files or len(data["episode"]) != n:
        print("warning: dataset has no episode ids; falling back to random sample split")
        order = torch.randperm(n)
        split = int(n * (1.0 - val_ratio))
        return order[:split], order[split:]

    episodes = np.asarray(data["episode"], dtype=np.int64)
    unique_episodes = np.unique(episodes)
    if len(unique_episodes) < 2:
        print("warning: dataset has fewer than 2 episodes; falling back to random sample split")
        order = torch.randperm(n)
        split = int(n * (1.0 - val_ratio))
        return order[:split], order[split:]
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_episodes)
    val_count = max(1, int(round(len(unique_episodes) * val_ratio)))
    if val_count >= len(unique_episodes):
        val_count = max(1, len(unique_episodes) - 1)
    val_episodes = set(int(ep) for ep in unique_episodes[:val_count])
    val_mask = np.array([int(ep) in val_episodes for ep in episodes], dtype=np.bool_)
    val_idx_np = np.flatnonzero(val_mask)
    train_idx_np = np.flatnonzero(~val_mask)
    print(
        f"episode split: train_episodes={len(unique_episodes) - val_count} "
        f"val_episodes={val_count}"
    )
    return torch.from_numpy(train_idx_np).long(), torch.from_numpy(val_idx_np).long()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--episodes", type=int, default=200)
    collect.add_argument("--max_steps", type=int, default=500)
    collect.add_argument("--seed", type=int, default=86)
    collect.add_argument("--agent_id", type=int, default=0)
    collect.add_argument("--output", type=str, default="agent/lgl_agent/data/imitation_dataset.npz")
    collect.add_argument("--max_samples", type=int, default=0)
    collect.add_argument("--flush_every", type=int, default=1000)
    collect.add_argument("--flush_episodes", type=int, default=10)
    collect.add_argument("--compressed", action="store_true")
    collect.add_argument("--rule_only_teacher", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--workers", type=int, default=1)
    collect.add_argument("--progress_every", type=int, default=100)

    train = sub.add_parser("train")
    train.add_argument("--dataset", type=str, default="agent/lgl_agent/data/imitation_dataset.npz")
    train.add_argument("--output", type=str, default="agent/lgl_agent/lgl_policy.pth")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--min_delta", type=float, default=0.001)
    train.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--val_ratio", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=86)

    args = parser.parse_args()
    if args.cmd == "collect":
        collect_dataset(args)
    else:
        train_bc(args)


if __name__ == "__main__":
    main()
