# Actor-Critic Policy For Task 2

This is the second learning-based policy framework for Task 2.
It implements a DDPG-style actor-critic method.

## Why Actor-Critic?

DQN is Q-learning based and uses discrete actions.
The real motor voltage is continuous:

```text
u in [-3, 3] V
```

Actor-critic is a natural fit because:

- the actor learns the continuous policy `u = pi(theta, omega)`;
- the critic learns `Q(theta, omega, u)`;
- the action does not need to be discretized.

This gives a clear comparison with your group member's DQN:

| Method | Action | Learned Object |
| --- | --- | --- |
| DQN | discrete voltage choices | `Q(state, action)` |
| Actor-critic | continuous voltage | actor `pi(state)` and critic `Q(state, action)` |

## Files

- `actor_critic_common.py`
  Actor network, critic network, replay buffer, soft target update, policy loader.

- `train_actor_critic.py`
  Trains the actor-critic policy in simulation.

- `evaluate_actor_critic.py`
  Loads a trained actor-critic policy and evaluates it in simulation.

## Short Smoke Test

This only checks that the code runs. It is not expected to learn a good policy.

```bash
cd /Users/zijiandong/Documents/GitHub/5SC28_MLSC_DesignAssignment
python3 project_framework/train_actor_critic.py --episodes 10 --warmup-steps 100 --out-dir project_framework/results/actor_critic_smoke
python3 project_framework/evaluate_actor_critic.py --model project_framework/results/actor_critic_smoke/actor_critic_policy.pt --out-dir project_framework/results/actor_critic_smoke --steps 300
```

## Train A Candidate Policy

```bash
python3 project_framework/train_actor_critic.py --episodes 800 --out-dir project_framework/results/actor_critic
python3 project_framework/evaluate_actor_critic.py --model project_framework/results/actor_critic/actor_critic_policy.pt --out-dir project_framework/results/actor_critic --steps 1200
```

## Render The Policy

```bash
python3 project_framework/evaluate_actor_critic.py --model project_framework/results/actor_critic/actor_critic_policy.pt --render
```

## What To Understand

You should be able to explain:

1. Actor: maps state features to continuous voltage.
2. Critic: estimates the value of a state-action pair.
3. Replay buffer: reuses past transitions for stable learning.
4. Target networks: reduce moving-target instability.
5. Soft update: slowly updates target networks using `tau`.
6. Exploration noise: helps the actor try different voltages during training.

This method satisfies the assignment's "actor-critic or model internalization based" requirement.
