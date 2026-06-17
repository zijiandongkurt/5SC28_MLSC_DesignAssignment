# Task 2 Actor-Critic

This folder contains your actor-critic method for Task 2.

## Main Scripts

- `train_actor_critic.py`: trains the actor-critic policy.
- `evaluate_actor_critic.py`: evaluates a trained policy and generates plots.
- `evaluate_actor_critic_safe.py`: evaluates the policy with voltage clipping, safety checks, shutdown handling, and optional real-setup mode.
- `actor_critic_common.py`: actor network, critic network, replay buffer, target-network update, policy loader.

## Main Results

- `actor_critic_policy.pt`: trained PyTorch policy.
- `actor_critic_training_curve.png`: training return and top-error curve.
- `actor_critic_eval.png`: simulation evaluation result.
- `actor_critic_training_log.npz`: raw training data.
- `actor_critic_eval_logs.npz`: raw evaluation data.

## Run

From the project root:

```bash
python3 Task2/actor_critic/train_actor_critic.py --episodes 800 --out-dir Task2/actor_critic
python3 Task2/actor_critic/evaluate_actor_critic.py --model Task2/actor_critic/actor_critic_policy.pt --out-dir Task2/actor_critic --steps 1200
```

Render:

```bash
python3 Task2/actor_critic/evaluate_actor_critic.py --model Task2/actor_critic/actor_critic_policy.pt --render --steps 1200 --sleep 0.025
```

Safe simulator test before real setup:

```bash
python3 Task2/actor_critic/evaluate_actor_critic_safe.py --env sim --model Task2/actor_critic/actor_critic_policy.pt --steps 300 --out-dir Task2/actor_critic/safe_test
```

Real setup dry run, only when the hardware is ready:

```bash
python3 Task2/actor_critic/evaluate_actor_critic_safe.py --env real --model Task2/actor_critic/actor_critic_policy.pt --steps 200 --out-dir Task2/actor_critic/real_test
```

For real setup, the script asks for confirmation. Type `RUN_REAL` only when the setup is connected and supervised.

## Report Description

The actor learns a continuous voltage policy `u = pi(theta, omega)`, while the critic estimates `Q(theta, omega, u)`.
Compared to DQN, this method does not discretize the voltage input.
