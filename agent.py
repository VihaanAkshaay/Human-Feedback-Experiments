import torch
import numpy
import random
from collections import deque
import env


# Create a policy network that has one input and one output with two layers
class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Input to first hidden layer
        self.fc1 = torch.nn.Linear(1, 24)
        # First hidden layer to second hidden layer
        self.fc2 = torch.nn.Linear(24, 12)
        # Second hidden layer to output layer with 3 outputs (for -1, 0, and 1)
        self.fc3 = torch.nn.Linear(12, 3)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        # Output layer: raw Q-values for actions
        x = self.fc3(x)
        return x

class Agent():
    def __init__(self, env):
        self.env = env
        self.policy = PolicyNetwork()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)

    def act(self, state):
        # Pass state through policy network
        q_values = self.policy(torch.tensor([state], dtype=torch.float32))
        # Select action with the highest Q-value
        action_index = torch.argmax(q_values).item()
        # Map the action index to the actual action (-1, 0, 1)
        actions = [-1, 0, 1]
        return actions[action_index]

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train(agent, env, buffer, num_episodes, batch_size, gamma, max_steps_per_episode):
    loss_fn = torch.nn.MSELoss()
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        # To log states and actions
        episode_states = []
        episode_actions = []

        while not done and steps < max_steps_per_episode:
            # Select an action (with exploration)
            if random.random() < 0.1:  # Epsilon-greedy: 10% exploration
                action = random.choice([-1, 0, 1])
            else:
                action = agent.act(state)

            # Log the state and action
            episode_states.append(state)
            episode_actions.append(action)

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store transition in replay buffer
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            steps += 1

            # Train the policy network if enough samples are available
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)

                # Prepare batch data
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
                actions = torch.tensor([[-1, 0, 1].index(a) for a in actions], dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Q-values for current states and actions
                q_values = agent.policy(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Target Q-values
                with torch.no_grad():
                    next_q_values = agent.policy(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss and optimize
                loss = loss_fn(q_values, target_q_values)
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

        total_rewards.append(total_reward)

        # Print performance every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
            print(f"States: {episode_states}")
            print(f"Actions: {episode_actions}")

    return total_rewards

env = env.RandomNumberEnvironment(10)
agent = Agent(env)
buffer = ReplayBuffer(capacity=10000)
total_rewards = train(agent, env, buffer, num_episodes=1000, batch_size=32, gamma=0.99, max_steps_per_episode=500)

