# LGL Agent

Hybrid submission agent for Bomberland:

1. Rule/BFS safety core.
2. Optional imitation policy checkpoint: `lgl_policy.pth`.
3. Optional PPO fine-tune of the same policy.

`agent.py` is submission-safe. It defines `class Agent`, loads `lgl_policy.pth` from the same folder if present, and falls back to pure rule/BFS if the checkpoint is missing.

## Quick Test

From repository root:

```powershell
python -m scripts.participant.run_local_match --agent_paths agent/lgl_agent/ TacticalRuleAgent GeniusRuleAgent SmarterRuleAgent --num_episodes 1 --max_steps 500
```

Visualize:

```powershell
python -m scripts.participant.visualizer --agent_paths agent/lgl_agent/ TacticalRuleAgent GeniusRuleAgent SmarterRuleAgent --num_episodes 1 --max_steps 500
```

Evaluate rough local stats:

```powershell
python -m agent.lgl_agent.evaluate --episodes 100 --opponents tactical genius smarter
```

## Train Imitation Policy

Collect behavior-cloning data from the built-in LGL rule/BFS teacher:

```powershell
python -m agent.lgl_agent.train_imitation collect --episodes 1000 --max_steps 500 --seed 86 --output agent/lgl_agent/data/imitation_dataset.npz
```

Train the CNN policy:

```powershell
python -m agent.lgl_agent.train_imitation train --dataset agent/lgl_agent/data/imitation_dataset.npz --epochs 10 --output agent/lgl_agent/lgl_policy.pth
```

After this, `agent.py` automatically loads `agent/lgl_agent/lgl_policy.pth`.

## PPO Fine-Tune

Start from the imitation checkpoint:

```powershell
python -m agent.lgl_agent.train_ppo --episodes 1000 --max_steps 500 --seed 186 --load_model agent/lgl_agent/lgl_policy.pth --output agent/lgl_agent/lgl_policy.pth
```

PPO still uses the rule/BFS safety mask. The model can only sample actions considered safe by `agent.py`.

## Submission

For a model submission, zip only:

```text
agent.py
lgl_policy.pth
```

For rule-only submission, zip only:

```text
agent.py
```

Do not include training datasets in the final zip.
