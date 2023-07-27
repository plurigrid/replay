import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import gym 
import random 
import numpy as np

class BayesianDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianDQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate_losses(model, inputs, targets, num_samples = 10):
    output = torch.zeros_like(targets)
    for _ in range(num_samples):
        output += model(inputs)
    output /= num_samples 
    return torch.mean((output - targets) ** 2)

class BayesianDQNAgent:
    def __init__(self, input_dim, output_dim, gamma, hidden_dim = 128, lr = 0.0001):
        self.model = BayesianDQN(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr)
        self.gamma = gamma
        
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.lr = lr
    
    def update(self, state, action, next_state, reward, done):
        self.model.train()
        self.optimizer.zero_grad()

        q_values = self.model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_vals = self.model(next_state)
            max_next_q_val = next_q_vals.max(1)[0]
            target_q_val = reward + (1 - done) * self.gamma * max_next_q_val
        
        loss = calculate_losses(self.model, state, target_q_val)
        loss.backward()
        self.optimizer.step()
    
    def act(self, state, epsilon):
        self.model.eval()

        if np.random.uniform(0, 1) < epsilon:
            return torch.tensor(random.randrange(self.output_dim))
        else:
            with torch.no_grad():
                q_vals = self.model(state)
                return q_vals.argmax().item()
    
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    num_episodes = 1000

    agent = BayesianDQNAgent(num_states, num_actions, 0.9)
    for episode in range(num_episodes):
        state = env.reset()
        done = False 
        
        while not done:
            action = agent.act(torch.tensor(state, dtype=torch.float32), epsilon=0.99)
            next_state, reward, done, _ = env.step(action.item())
            agent.update(torch.tensor(state, dtype=torch.float32),
                         torch.tensor([action], dtype=torch.long),  
                         torch.tensor(next_state, dtype=torch.float32),
                         torch.tensor(reward, dtype=torch.float32),
                         torch.tensor(done, dtype=torch.float32))
            state = next_state 
            print("Episode {} finished".format(episode + 1))
        
    

