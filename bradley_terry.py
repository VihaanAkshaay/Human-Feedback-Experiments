import torch
import torch.nn as nn
import torch.optim as optim
import random

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(1, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def query_preference(a, b):
    # Return 1 if a is preferred, 0 if b is preferred
    print(f"Which do you prefer? a: {a}, b: {b} (Enter a or b):")
    pref = input()
    while pref not in ['a', 'b']:
        print("Invalid input. Enter 0 for a, 1 for b.")
        pref = input()
    if pref == "a":
        return 1
    else:
        return 0

# Workflow:
# 1. Initialize the reward model
model = RewardModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_items = 10
num_queries = 20

# Generate random pairs of numbers
pairs = [(random.randint(1, num_items), random.randint(1, num_items)) for _ in range(num_queries)]
preferences = []

# 2. Collect preferences
for pair in pairs:
    if pair[0] != pair[1]:
        pref = query_preference(pair[0], pair[1])
        preferences.append((pair, pref))

# 3. Train the reward model
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for pair, pref in preferences:

        optimizer.zero_grad()
        reward_a = model(torch.tensor([pair[0]], dtype=torch.float32))
        reward_b = model(torch.tensor([pair[1]], dtype=torch.float32))
        regularization = 0.01 * (reward_a.pow(2).mean() + reward_b.pow(2).mean())  # L2 regularization

        # Bradley Terry probability P (a beats b) = sig (r_a - r_b) = 1/(1 + exp(r_b - r_a)))
        prob_a_beats_b = 1 / (1 + torch.exp(reward_b - reward_a))
        
        epsilon = 1e-8  # Small constant to avoid log(0)
        loss = - (pref * torch.log(prob_a_beats_b + epsilon) + (1 - pref) * torch.log(1 - prob_a_beats_b + epsilon)) + regularization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# 4. Output reward for each number
print("Reward for each item:")
for i in range(1, num_items + 1):
    reward = model(torch.tensor([i], dtype=torch.float32)).item()
    print(f"Item: {i}, Reward: {reward}")

