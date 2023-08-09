import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import gym 
import tqdm 
import numpy as np 
import collections 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def update_policy(policy, state, action, reward, next_state, discount_factor, optimizer):
    q_preds = policy(state)
    q_vals = q_preds[:, action]

    with torch.no_grad():
        q_next_preds = policy(next_state)
        q_next_vals = q_next_preds.max(1).values 
        targets = reward + q_next_preds * discount_factor

    loss = F.smooth_l1_loss(targets.detach(), q_vals)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    return loss.item()

def kaiming_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def train(env, policy, optimizer, discount_factor, epsilon, device):
    policy.train()

    states = []
    actions = []
    rewards = []
    next_states = []
    done = False
    episode_reward = 0

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            q_pred = policy(state)
            action = torch.argmax(q_pred).item()
        
        next_state, reward, done, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        state = next_state 
        episode_reward += reward 
        loss = update_policy(policy, state, action, reward, next_state, discount_factor, optimizer)
    return loss, episode_reward, epsilon 

def evaluate(env, policy, device):
    policy.eval()

    done = False 
    episode_reward = 0

    state = env.reset()

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_pred = policy(state)
            action = torch.argmax(q_pred).item()

        state, reward, done, _ = env.step(action)
        episode_reward += reward 
    return episode_reward 

if __name__ == "__main__":

    SEED = 1234 

    n_runs = 10 
    n_episodes = 500 
    discount_factor = 0.8
    start_epsilon = 1.0 
    end_epsilon = 0.01 
    epsilon_decay = 0.995

    train_rewards = torch.zeros(n_runs, n_episodes)
    test_rewards = torch.zeros(n_runs, n_episodes)

    device = torch.device('cpu')
    env = gym.make('CartPole-v1')

    train_env = gym.make('CartPole-v1')
    test_env = gym.make('CartPole-v1')

    train_env.seed(SEED)
    test_env.seed(SEED + 1)

    input_dim = env.observation_space.shape[0]
    hidden_dim = 128 
    output_dim = env.action_space.n 

    for run in range(n_runs):
        policy = MLP(input_dim, hidden_dim, output_dim)
        policy = policy.to(device)
        policy.apply(kaiming_init)
        epsilon = start_epsilon 

        optimizer = torch.optim.RMSprop(policy.parameters(), lr = 1e-6)

        for episode in tqdm.tqdm(range(n_episodes), desc = f'Run: {run}'):
            loss, train_reward, epsilon = train(train_env, policy, optimizer, discount_factor, epsilon, device)
            epsilon *= epsilon_decay 

            test_reward = evaluate(test_env, policy, device)
            train_rewards[run][episode] = train_reward  
            test_rewards[run][episode] = test_reward 