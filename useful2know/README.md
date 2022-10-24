# Tabular Q-learning for Home World game

Recall that the state observable to the player is described in text. Therefore we have to choose a mechanism that maps text descriptions into vector representations.

In tabular_q_learning you will consider a simple approach that assigns a unique index for each text description. In particular, we will build two dictionaries:

- ```dict_room_desc``` that takes the room description text as the key and returns a unique scalar index
- ```dict_quest_desc``` that takes the quest description text as the key and returns a unique scalar index.

For instance, consider an observable state $s = (s_r, s_q)$, where $s_{r}$ and $s_{q}$ are the text descriptions for the current room and the current request, respectively. Then $i_{r}=$```dict_room_desc```$[s_{r}]$ gives the scalar index for $s_{r}$ and $i_{q}=$```dict_quest_desc```$[s_{q}]$ gives the scalar index for $s_{q}$. That is, the textual state $s=(s_{r},s_{q})$ is mapped to a tuple $I=(i_{r},i_{q})$.

Normally, we would build these dictionaries as we train our agent, collecting descriptions and adding them to the list of known descriptions. For the purpose of this project, these dictionaries will be provided to you.

## Evaluating Tabular Q-learning on Home World

The following python files are provided:

- ```framework.py``` contains various functions for the text-based game environment that the staff has implemented for you. Some functions that you can call to train and testing your reinforcement learning algorithms:
    - ```newGame()```
        - Args: None
        - Return: A tuple where the first element is a description of the initial room, the second element is a description of the quest for this new game episode, and the last element is a Boolean variable with value False implying that the game is not over.

    - ```step_game()```
        - Args:
            - ```current_room_desc```: An description of the current room
            - ```current_quest_desc```: A description of the current quest state
            - ```action_index```: An integer used to represent the index of the selected action
            - ```object_index```: An integer used to indicate the index of the selected object
        - Return: the system next state when the selected command is applied at the current state.
            - ```next_room_desc```: The description of the room of the next state
            - ```next_quest_desc```: The description of the next quest
            - ```reward```: A real valued number representing the one-step reward obtained at this step
            - ```terminal```: A boolean valued number indicating whether this episode is over (either quest is finished, or the number of steps reaches the maximum number of steps for each episode).
- ```agent_tabular_QL.py``` contains various function templates that you will use to implement your learning algorithm.

In this section, you will evaluate your learning algorithm for the Home World game. The metric we use to measure an agent's performance is the cumulative discounted reward obtained per episode averaged over the episodes.

The evaluation procedure is as follows. Each experiment (or run) consists of multiple epochs (the number of epochs is ```NUM_EPOCHS```). In each epoch:

You first train the agent on ```NUM_EPIS_TRAIN``` episodes, following an $\varepsilon$-greedy policy with $\varepsilon=$```TRAINING_EP``` and updating the Q values.

Then, you have a testing phase of running ```NUM_EPIS_TEST``` episodes of the game, following an $\varepsilon$-greedy policy with $\varepsilon =$```TESTING_EP```, which makes the agent choose the best action according to its current Q-values $95\%$ of the time. At the testing phase of each epoch, you will compute the cumulative discounted reward for each episode and then obtain the average reward over the ```NUM_EPIS_TEST``` episodes.

Finally, at the end of the experiment, you will get a sequence of data (of size ```NUM_EPOCHS```) that represents the testing performance at each epoch.

Note that there is randomness in both the training and testing phase. You will run the experiment ```NUM_RUNS``` times and then compute the averaged reward performance over ```NUM_RUNS``` experiments.

Most of these operations are handled by the boilerplate code provided in the ```agent_tabular_QL.py``` file by functions run, run_epoch and main, but you will need to complete the run_episode function.