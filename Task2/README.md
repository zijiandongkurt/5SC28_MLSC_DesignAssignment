# Task 2: Policy Learning

Task 2 is about learning a controller/policy.

It answers:

```text
Given the current angle and angular velocity, what voltage should be applied?
```

This is different from Task 1. Task 1 learns dynamics; Task 2 learns control.

## Current Folder Structure

```text
Task2/
  shared/
    policy_interface.py
    dqn_common.py
  actor_critic/
    actor_critic_common.py
    train_actor_critic.py
    evaluate_actor_critic.py
    actor_critic_policy.pt
    actor_critic_training_curve.png
    actor_critic_eval.png
```

## Your Method

Your implemented method is actor-critic, specifically a DDPG-style actor-critic:

- Actor: maps state to continuous voltage.
- Critic: estimates the value of a state-action pair.

This satisfies the assignment's actor-critic/model-internalization requirement.

## Recommended Run Commands

The original runnable scripts are still in `project_framework/`, which keeps imports simple.

Train:

```bash
cd /Users/zijiandong/Documents/GitHub/5SC28_MLSC_DesignAssignment
python3 project_framework/train_actor_critic.py --episodes 800 --out-dir Task2/actor_critic
```

Evaluate:

```bash
python3 project_framework/evaluate_actor_critic.py --model Task2/actor_critic/actor_critic_policy.pt --out-dir Task2/actor_critic --steps 1200
```

Render with the course simulator:

```bash
python3 project_framework/evaluate_actor_critic.py --model Task2/actor_critic/actor_critic_policy.pt --render --steps 1200 --sleep 0.025
```

## Main Results To Use In Report

- `Task2/actor_critic/actor_critic_training_curve.png`
- `Task2/actor_critic/actor_critic_eval.png`
- terminal evaluation metrics: max voltage, mean reward, final angle/velocity, final top error.
