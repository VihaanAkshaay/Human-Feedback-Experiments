import torch 
import env
import agent
    
# Run RL experiment with random agent
env = env.RandomNumberEnvironment(10)
policy = PolicyNetwork()

# Define the number of episodes
episodes = 10

# Start the RL experiment
for episode in range(episodes):
    print('episode',episode)
    state = env.reset()
    done = False
    while not done:
        action = torch.randint(-1, 1, (1,))
        print(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        

