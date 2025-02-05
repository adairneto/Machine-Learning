import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("OptimalPolicy.csv")

# Define constants
gamma = 0.9  # Discount factor
num_iterations = 100  # Number of iterations
m = 10  # Number of sampled states
k = 5  # Number of next state samples per action

# Define state and action spaces
states = list(zip(data["state_x"], data["state_y"]))
actions = data["Action"].unique()

# Feature function
def phi(s):
    x, y = s
    return np.array([1, x, y, x**2, y**2])  # Quadratic feature mapping

# Reward function
def R(s, a):
    row = data[(data["state_x"] == s[0]) & (data["state_y"] == s[1]) & (data["Action"] == a)]
    if not row.empty:
        return row["u"].values[0]  # Assuming 'u' represents a reward metric
    return -1  # Default penalty

# State transition model
def transition(s, a):
    row = data[(data["state_x"] == s[0]) & (data["state_y"] == s[1]) & (data["Action"] == a)]
    if not row.empty:
        return (row["move_x"].values[0], row["move_y"].values[0])
    return s  # Default to no movement

# Approximate the optimal value function
def FittedValue():
    theta = np.zeros(len(phi((0, 0))))  # Initialize theta to zero
    
    for _ in range(num_iterations):
        y = np.zeros(m)
        sampled_states = [states[i] for i in np.random.choice(len(states), m)]
        
        for i, s in enumerate(sampled_states):
            q = np.zeros(len(actions))
            
            for idx, a in enumerate(actions):
                s_primes = [transition(s, a) for _ in range(k)]
                V_s_prime = np.max([theta.T @ phi(s_prime) for s_prime in s_primes])
                q[idx] = np.mean(R(s, a) + gamma * V_s_prime)
            
            y[i] = np.max(q)
        
        # Update theta using least squares regression
        Phi = np.array([phi(s) for s in sampled_states])
        theta = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y
    
    return theta

# Compute optimal policy
def optimal_policy(theta):
    policy = {}
    for s in states:
        best_action = max(actions, key=lambda a: theta.T @ phi(transition(s, a)))
        policy[s] = best_action
    return policy

# Visualize optimal policy
def visualize_policy(policy):
    x_vals, y_vals, action_vals = zip(*[(s[0], s[1], policy[s]) for s in policy])
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_vals, y_vals, c='lightgray', marker='s', s=100, edgecolor='black')
    
    for x, y, action in zip(x_vals, y_vals, action_vals):
        plt.text(x, y, action, fontsize=2.5, ha='center', va='center', color='red')
    
    plt.xlabel("State X")
    plt.ylabel("State Y")
    plt.title("Optimal Policy Visualization")
    plt.grid(True)
    plt.show()

# Run the value function approximation
theta = FittedValue()
policy = optimal_policy(theta)
print("Optimal Policy:", policy)

# Visualize the policy
visualize_policy(policy)
