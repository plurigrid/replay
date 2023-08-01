# CBIMK - Compositional Bayesian Inversion with Stochastic Markov Kernels

<p align="center">
  <img src="contourFIG1.png" />
</p>

## Markov Kernels and Bayesian Inversions

In various scientific and engineering applications, we often encounter systems that exhibit probabilistic transitions between different states over time. Such systems can be elegantly modelled using Markov kernels. A Markov kernel is a mathematical construct that characterizes the probabilistic evolution of states in a Markov process.

Intuitively, a Markov kernel can be thought of as a "transition rule" that dictates how the probabilities of being in different states change from one time step to another. The core idea is that the future state probabilities depend solely on the current state probabilities and not on the history of states.

On the other hand, Bayesian inversion is a powerful technique used to infer the underlying structure of a system from observed data. In the context of Markov kernels, Bayesian inversion aims to learn the inverse transformation that maps the probabilities of states at a given time step back to the probabilities at the previous time step.

## Compositional Bayesian Inversion

The Compositional Bayesian Inversion of Markov Kernels combines the strengths of Markov kernels and Bayesian inversion in a novel and efficient manner. The approach involves constructing a Bayesian inverter, which learns to approximate the inverse transformation of a given Markov kernel.

In this implementation, we represent the Markov kernel using a transition matrix, where each entry indicates the probability of transitioning from one state to another. The Bayesian inverter is modelled as a neural network, which is trained to learn the inverse transformation of the Markov kernel.

The key insight is that by composing the Markov kernel with the Bayesian inverter, we can obtain an approximate inverse transformation, allowing us to infer the initial state probabilities given the probabilities at a subsequent time step in a transition timeline.

## Key Components

1. **MarkovKernel**: Represents a Markov kernel with a given transition matrix, which models the probabilistic transition between states in a Markov process using a transition matrix and input data.
2. **BayesianInverter**: A neural network-based Bayesian inverter, which learns to approximate the inverse transformation of a given Markov kernel.

Additionally, the code provides a composition function, `compose`, which combines the Markov kernel and the Bayesian inverter to create a newly composed transformation from the transition matrix and inputted data.

## Usage
1. **Define the Transition Matrix**: To begin, you need to define the transition matrix for your Markov kernel. This matrix should represent the probabilities of transitioning between states in your Markov process.
2. **Create Instances of MarkovKernel and BayesianInverter**: Instantiate the `MarkovKernel` class with the defined transition matrix and the `BayesianInverter` class with the appropriate input and output dimensions.
3. **Apply the Composed Kernel**: Use the `compose` function to combine the Markov kernel and the Bayesian inverter. Apply the resulting composed kernel to your input data to approximate the inverse transformation.

## Example Usage 
```python
transition_matrix = torch.tensor([[0.8, 0.2], [0.4, 0.6], [0.5, 0.3]], dtype = torch.float32)
markov_kernel = MarkovKernel(transition_matrix)
bayesian_inverter = BayesianInverter(2, 3)

input_data = torch.tensor([[0.7, 0.3, 0.5], [0.2, 0.8, 0.9], [0.7, 0.6, 0.1]], dtype = torch.float32)
output = compose(markov_kernel, bayesian_inverter).apply(input_data)
```
## 
# Prioritized Replay Buffer

## Introduction
The **PrioritizedReplayBuffer** class is a Python implementation of a prioritized replay buffer, a data structure used in reinforcement learning algorithms, particularly in deep reinforcement learning with experience replay. This buffer prioritizes the experiences based on their importance or TD-error, allowing the agent to focus on the most relevant experiences during training.

## Class Overview
### `PrioritizedReplayBuffer(capacity, alpha=0.6, beta=0.4, beta_annealing=0.001)`
- **Parameters**:
  - `capacity` (int): The maximum capacity of the replay buffer.
  - `alpha` (float, optional): A hyperparameter that determines how much prioritization is used (0 corresponds to no prioritization). Default value is 0.6.
  - `beta` (float, optional): A hyperparameter that controls the importance sampling correction, allowing the correction of biased updates due to prioritization. Default value is 0.4.
  - `beta_annealing` (float, optional): A small value used to anneal the beta value over time to reduce the effect of the bias correction. Default value is 0.001.
  
- **Attributes**:
  - `buffer` (collections.deque): The deque used to store the experiences.
  - `priorities` (numpy.ndarray): An array storing the priorities of experiences.
  - `index` (int): The current index to add new experiences to the buffer.

### `add(state, action, reward, next_state, done)`
Add a new experience to the replay buffer.

- **Parameters**:
  - `state` (numpy.ndarray): The current state.
  - `action` (int): The action taken in the current state.
  - `reward` (float): The reward received after taking the action.
  - `next_state` (numpy.ndarray): The resulting state after taking the action.
  - `done` (bool): A flag indicating whether the episode terminated after the action.

### `sample(batch_size)`
Sample a batch of experiences from the replay buffer.

- **Parameters**:
  - `batch_size` (int): The size of the batch to sample.

- **Returns**:
  - A tuple of the following lists:
    - `states` (list): A list of states in the batch.
    - `actions` (list): A list of actions in the batch.
    - `rewards` (list): A list of rewards in the batch.
    - `next_states` (list): A list of next states in the batch.
    - `dones` (list): A list of done flags in the batch.
  - `indices` (numpy.ndarray): An array of indices corresponding to the sampled experiences in the buffer.
  - `weights` (numpy.ndarray): An array of importance sampling weights for each experience in the batch.

### `update_priorities(indices, priorities)`
Update the priorities of the specified experiences in the replay buffer.

- **Parameters**:
  - `indices` (numpy.ndarray): An array of indices corresponding to the experiences whose priorities need to be updated.
  - `priorities` (numpy.ndarray): An array of updated priorities corresponding to the specified indices.

## Testing
The `TestPrioritizedReplayBuffer` class contains unit tests to ensure the correctness of the `PrioritizedReplayBuffer` class implementation. It checks the functionality of adding experiences to the buffer, sampling batches, and updating priorities.

## Usage
To use the `PrioritizedReplayBuffer` class, follow these steps:

1. Create an instance of the `PrioritizedReplayBuffer` class with the desired capacity and optional hyperparameters.
2. Use the `add` method to add experiences to the buffer as the agent interacts with the environment.
3. During training, sample batches of experiences using the `sample` method, and provide the experiences to the learning algorithm.
4. After calculating the TD-errors or losses, update the priorities of the sampled experiences using the `update_priorities` method.

Example:
```python
# Create a prioritized replay buffer with capacity 100
buffer = PrioritizedReplayBuffer(capacity=100)

# Add experiences to the buffer
state = np.random.rand(4)
action = np.random.randint(0, 5)
reward = np.random.rand()
next_state = np.random.rand(4)
done = np.random.choice([True, False])
buffer.add(state, action, reward, next_state, done)

# Sample a batch from the buffer
batch_size = 32
(states, actions, rewards, next_states, dones), indices, weights = buffer.sample(batch_size)

# Perform training and calculate TD-errors
# ...

# Update priorities after training
new_priorities = np.random.rand(batch_size)
buffer.update_priorities(indices, new_priorities)
