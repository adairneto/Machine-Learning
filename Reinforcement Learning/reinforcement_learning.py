import numpy as np

# Define constants
gamma = 0.9  # Discount factor
num_iterations = 100  # Number of iterations
m = 10  # Number of sampled states
k = 5  # Number of next state samples per action

# Define state space and action space (example)
states = np.linspace(-1, 1, 100)  # Example continuous states
actions = np.array([-1, 0, 1])  # Example discrete actions

# Feature function
def phi(s):
    return np.array([1, s, s**2])  # Quadratic feature mapping

# Reward function (example)
def R(s):
    return -s**2  # Negative quadratic cost as reward

# State transition model (example)
def transition(s, a):
    return s + 0.1 * a + np.random.normal(0, 0.1)  # Linear transition with noise

# Approximate the optimal value function
def FittedValue():
    theta = np.zeros(len(phi(0)))  # Initialize theta to zero
    
    for _ in range(num_iterations):
        y = np.zeros(m)
        sampled_states = np.random.choice(states, m)
        
        for i, s in enumerate(sampled_states):
            q = np.zeros(len(actions))
            
            for idx, a in enumerate(actions):
                s_primes = np.array([transition(s, a) for _ in range(k)])
                V_s_prime = np.max([theta.T @ phi(s_prime) for s_prime in s_primes])
                q[idx] = np.mean(R(s) + gamma * V_s_prime)
            
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

# Run the value function approximation
theta = FittedValue()
policy = optimal_policy(theta)
print("Optimal Policy:", policy)