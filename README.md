
# Solving a Markov Decision Process Using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment.

The objective is to find the optimal state-value function and optimal policy by repeatedly performing policy evaluation and policy improvement until the policy becomes stable.

## Problem Statement

In this experiment, the FrozenLake-v1 environment is solved using the Policy Iteration algorithm.

The agent starts from an initial position and must reach the goal without falling into holes. The environment is represented as a finite Markov Decision Process (MDP).

Policy Iteration is used to evaluate the current policy and improve it repeatedly until the optimal policy is obtained.

The main objectives are:

1. Calculate the optimal state-value function V*(s).
2. Determine the optimal policy pi*(s).
3. Find the best action for each state to maximize the expected discounted reward.

## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

**Requirements:**
- Python 3.x
- Gymnasium
- NumPy
- Jupyter Notebook or any Python IDE

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment in which an agent moves across frozen tiles to reach a goal while avoiding holes.

The default environment contains a 4 x 4 grid.

| Component | Description |
|---|---|
| Environment | FrozenLake-v1 |
| Map size | 4 x 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Action 0 | Left |
| Action 1 | Down |
| Action 2 | Right |
| Action 3 | Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and holes |

**Environment layout:**

```text
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

Where:

- S = Starting position
- F = Frozen surface
- H = Hole
- G = Goal

## Theory

Policy Iteration is a Dynamic Programming algorithm used to find the optimal policy of a Markov Decision Process.

It consists of two main steps:

1. Policy Evaluation
2. Policy Improvement

These two steps are repeated until the policy becomes stable.

### Policy Evaluation

Policy evaluation calculates the value of each state under the current policy.

The Bellman expectation equation is:

```text
V(s) = SUM over actions [ pi(a|s) *
       SUM over next states [ P(s'|s,a) *
       (R(s,a,s') + gamma * V(s')) ] ]
```

Where:

| Symbol | Meaning |
|---|---|
| s | Current state |
| a | Selected action |
| s' | Next state |
| pi(a\|s) | Probability of selecting an action |
| P(s'\|s,a) | Transition probability |
| R(s,a,s') | Immediate reward |
| gamma | Discount factor |
| V(s) | Value of state under the current policy |

The algorithm repeatedly updates the state values until the maximum change between consecutive iterations becomes smaller than a predefined threshold.

```text
Delta = MAX over all states |New V(s) - Old V(s)|
```

When:

```text
Delta < theta
```

the policy evaluation process is considered converged.

### Policy Improvement

Policy improvement updates the current policy by selecting the action that provides the highest expected return according to the calculated state values.

The improved policy is:

```text
pi_new(s) = ARGMAX over actions [
    SUM over next states [
        P(s'|s,a) * (R(s,a,s') + gamma * V(s'))
    ]
]
```

The algorithm compares the old policy with the improved policy.

If both policies are identical, the policy is considered stable and the algorithm terminates.

Otherwise, the improved policy is evaluated again.

## Algorithm

1. Create the Gymnasium FrozenLake-v1 environment.
2. Initialize the state-value function with zeros.
3. Initialize a random policy by assigning an action to every state.
4. Set the discount factor and convergence threshold.
5. Perform policy evaluation:
   - Calculate the expected value of every state using the Bellman expectation equation.
   - Repeat the updates until the value function converges.
6. Perform policy improvement:
   - Calculate the expected return for every available action.
   - Select the action with the highest expected return.
   - Update the policy.
7. Compare the old policy with the new policy.
8. If the policy has changed, repeat policy evaluation and policy improvement.
9. Stop when the policy becomes stable.
10. Display the optimal state-value function and optimal policy.

## Python Program

```python
import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)
P = env.unwrapped.P

# Parameters
gamma = 0.99
theta = 1e-8

n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize value function and random policy
V = np.zeros(n_states)
policy = np.random.randint(n_actions, size=n_states)

# Identify terminal states
desc = env.unwrapped.desc.flatten()
terminal = [
    s for s in range(n_states)
    if desc[s] in (b'H', b'G')
]

# Policy Iteration
iterations = 0

while True:

    # Policy Evaluation
    while True:
        delta = 0
        new_V = V.copy()

        for s in range(n_states):

            if s in terminal:
                new_V[s] = 0
                continue

            value = 0
            a = policy[s]

            for prob, next_s, reward, done in P[s][a]:
                next_value = 0 if done else V[next_s]
                value += prob * (
                    reward + gamma * next_value
                )

            new_V[s] = value
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if delta < theta:
            break

    # Policy Improvement
    stable = True

    for s in range(n_states):

        if s in terminal:
            continue

        old_action = policy[s]
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            for prob, next_s, reward, done in P[s][a]:
                next_value = 0 if done else V[next_s]

                action_values[a] += prob * (
                    reward + gamma * next_value
                )

        policy[s] = np.argmax(action_values)

        if old_action != policy[s]:
            stable = False

    iterations += 1

    if stable:
        break

# Display results
actions = ['L', 'D', 'R', 'U']

print("Total Policy Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

print("\nOptimal Policy:")
print(np.array(actions)[policy].reshape(4, 4))

env.close()
```

## Output

The program displays the total number of policy iterations, the optimal state-value function, and the optimal policy.

**Expected output format:**

```text
Total Policy Iterations: [Calculated by program]

Optimal State-Value Function:

[[0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]]

Optimal Policy:

[['L' 'D' 'R' 'U']
 ['L' 'D' 'R' 'U']
 ['L' 'D' 'R' 'U']
 ['L' 'D' 'R' 'U']]
```

The values and actions are placeholders showing the output format. The actual results are calculated when the program is executed.

**Action symbols:**

- L = Left
- D = Down
- R = Right
- U = Up

## Result

The Policy Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

The algorithm performed policy evaluation and policy improvement repeatedly until the policy became stable.

The optimal state-value function and corresponding optimal policy were obtained.

## Inference

- Policy Iteration is used to find an optimal policy for a finite Markov Decision Process.
- Policy evaluation calculates the value of states under the current policy.
- Policy improvement selects better actions using the estimated state values.
- The discount factor determines the importance of future rewards.
- Repeating both steps gradually improves the policy.
- The algorithm terminates when no further policy changes occur.

## About

Implementation of Policy Iteration for solving a finite Markov Decision Process using Python, NumPy, and Gymnasium.
