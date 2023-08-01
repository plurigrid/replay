import collections 
import numpy as np 
import unittest

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha = 0.6, beta = 0.4, beta_annealing = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.buffer = collections.deque(maxlen = capacity)
        self.priorities = np.ones(capacity, dtype = np.float32)  
        self.index = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.index] = (state, action, reward, next_state, done)

        self.priorities[self.index] = max_priority
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alphas
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p = probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return zip(*samples), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def test_buffer_functionality(self):
        capacity = 100
        buffer = PrioritizedReplayBuffer(capacity)
        self.assertEqual(len(buffer), 0)

        for i in range(capacity + 10):
            state = np.random.rand(4)
            action = np.random.randint(0, 5)
            reward = np.random.rand()
            next_state = np.random.rand(4)
            done = np.random.choice([True, False])
            buffer.add(state, action, reward, next_state, done)

        self.assertEqual(len(buffer), capacity)

        batch_size = 32
        (states, actions, rewards, next_states, dones), indices, weights = buffer.sample(batch_size)
        self.assertEqual(len(states), batch_size)  
        self.assertEqual(len(actions), batch_size)  
        self.assertEqual(len(rewards), batch_size)  
        self.assertEqual(len(next_states), batch_size)  
        self.assertEqual(len(dones), batch_size)  
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(len(weights), batch_size)

        new_priorities = np.random.rand(batch_size)
        buffer.update_priorities(indices, new_priorities)

        for idx, priority in zip(indices, new_priorities):
            self.assertAlmostEqual(buffer.priorities[idx], priority, places = 5)


if __name__ == '__main__':
    unittest.main()
