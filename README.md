# Reinforcement Learning Homework

In this homework, we address the task of learning control policies for text-based games using reinforcement learning. In these games, all interactions between players and the virtual world are through text. The current world state is described by elaborate text, and the underlying state is not directly observable. Players read descriptions of the state and respond with natural language commands to take actions.

For this project you will conduct experiments on a small **Home World**, which mimic the environment of a typical house.The world consists of a few rooms, and each room contains a representative object that the player can interact with. For instance, the kitchen has an apple that the player can eat. The goal of the player is to finish some quest. An example of a quest given to the player in text is **"You are hungry now"**. To complete this quest, the player has to navigate through the house to reach the kitchen and eat the apple. In this game, the room is hidden from the player, who only receives a description of the underlying room. At each step, the player read the text describing the current room and the quest, and respond with some command (e.g., **eat apple**). The player then receives some reward that depends on the state and his/her command. In order to design an autonomous game player, we will employ a reinforcement learning framework to learn command policies using game rewards as feedback.

In order to design an autonomous game player, we will employ a reinforcement learning framework to learn command policies using game rewards as feedback. Since the state observable to the player is described in text, we have to choose a mechanism that maps text descriptions into vector representations. A naive approach is to create a map that assigns a unique index for each text description. However, such approach becomes difficult to implement when the number of textual state descriptions are huge. An alternative method is to use a bag-of-words representation derived from the text description. This project requires you to complete the following tasks:

1. Implement the tabular Q-learning algorithm for a simple setting where each text description is associated with a unique index.

2. Implement a deep Q-network.

3. Use your Q-learning algorithms on the Home World game.

To properlly understand the **Home World Game** we recommend you to read the README in the Home-world folder of this repo

## Part 1. Tabular Q-Learning

In this section, you will implement the Q-learning algorithm using the ```agent_tabular_ql.py``` file, which is a model-free algorithm used to learn an optimal Q-function. In the tabular setting, the algorithm maintains the Q-value for all possible state-action pairs. Starting from a random Q-function, the agent continuously collects experiences $(s,c,R(s,c),s')$ and updates its Q-function.

From now on, we will refer to $c = (a, b)$ as “an action" although it really is an action with an object.

### Q-learning Algorithm

The agent plays an action $c$ at state $s$, getting a reward $R(s,c)$ and observing the next state $s'$.

Update the single Q-value corresponding to each such transition:

$$Q(s,c)\leftarrow (1-\alpha )Q(s,c)+\alpha [R(s,c)+\gamma \max _{c'\in C}Q(s',c')]$$

**Write a function ```tabular_q_learning``` that updates the single Q-value, given the transition date $(s,c,R(s,c),s')$**

Note that the Q-learning algorithm does not specify how we should interact in the world so as to learn quickly. It merely updates the values based on the experience collected. If we explore randomly, i.e., always select actions at random, we would most likely not get anywhere. A better option is to exploit what we have already learned, as summarized by current Q-values. We can always act greedily with respect to the current estimates, i.e., take an action $\pi (s)=\arg \max _{c\in C}Q(s,c)$. Of course, early on, these are not necessarily very good actions. For this reason, a typical exploration strategy is to follow a so-called \varepsilon-greedy policy: with probability $\varepsilon$ take a random action out of $C$ with probability $1-\varepsilon$ follow $\pi (s)=\arg \max _{c\in C}Q(s,c)$. The value of \varepsilon here balances exploration vs exploitation. A large value of \varepsilon means exploring more (randomly), not using much of what we have learned. A small \varepsilon, on the other hand, will generate experience consistent with the current estimates of Q-values.

**Write a function ```epsilon_greedy``` that implements the $\varepsilon$-greedy exploration policy using the current Q-function.**

Once, you complete tabular_q_learning and $\varepsilon$-greedy functions, in your Q-learning algorithm, initialize Q at zero. Set ```NUM_RUNS =10```, ```NUM_EPIS_TRAIN =25```, ```NUM_EPIS_TEST=50```, $\gamma$ =0.5, ```TRAINING_EP=0.5```, ```TESTING_EP=0.05``` and the learning rate $\alpha =0.1$.

Please include in your report the number of epochs and the plot when the learning algorithm converges. That is, the testing performance become stable. Also, please enter the average episodic rewards of your Q-learning algorithm when it converges.

Check the ```useful2know``` folder to get better insights on how the code works and how are all ```.py``` files useful.

## Part 2. Deep Q-network

Since the state displayed to the agent is described in text, we have to choose a mechanism that maps text descriptions into vector representations. A naive way is to create one unique index for each text description, as we have done in previous part. However, such approach becomes infeasible when the state space becomes huge. To tackle this challenge, we can design some representation generator that does not scale as the original textual state space. In particular, a representation generator $\phi_R(\cdot)$ reads raw text displayed to the agent and converts it to a vector representation $v_{s}=\psi _{R}(s)$. One approach is to use a bag-of-words representation derived from the text description.

In this part, you will approximate $Q(s, c)$ with a neural network. You will be provided with a DQN that takes the state representation (bag-of-words) and outputs the predicted Q values for the different "actions" and "objects". Complete the function deep_q_learning that updates the model weights, given the transition date $(s,c,R(s,c),s')$. Please include in your report the average episodic rewards of your Q-learning algorithm when it converges.



