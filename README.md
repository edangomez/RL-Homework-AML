# Reinforcement Learning Homework

In this homework, we address the task of learning control policies for text-based games using reinforcement learning. In these games, all interactions between players and the virtual world are through text. The current world state is described by elaborate text, and the underlying state is not directly observable. Players read descriptions of the state and respond with natural language commands to take actions.

For this project you will conduct experiments on a small **Home World**, which mimic the environment of a typical house.The world consists of a few rooms, and each room contains a representative object that the player can interact with. For instance, the kitchen has an apple that the player can eat. The goal of the player is to finish some quest. An example of a quest given to the player in text is **"You are hungry now"**. To complete this quest, the player has to navigate through the house to reach the kitchen and eat the apple. In this game, the room is hidden from the player, who only receives a description of the underlying room. At each step, the player read the text describing the current room and the quest, and respond with some command (e.g., **eat apple**). The player then receives some reward that depends on the state and his/her command. In order to design an autonomous game player, we will employ a reinforcement learning framework to learn command policies using game rewards as feedback.

Consider a text-based game represented by the tuple < $H,C,P,R,\gamma ,\Psi$ >. Here $H$ is the set of all possible game states. The actions taken by the player are multi-word natural language **commands** such as **eat apple** or **go east**. In this homework we limit ourselves to consider commands consisting of one action (e.g., **eat**) and one argument object (e.g. **apple**).

$C=\{ (a,b)\}$ is the set of all commands (action-object pairs).

$P:H\times C\times H\rightarrow [0,1]$ is the transition matrix: $P(h'|h,a,b)$ is the probability of reaching state $h'$ if command $c = (a,b)$ is taken in state $h$.

$R:H\times C\rightarrow \mathbb {R}$ is the deterministic reward function: $R(h,a,b)$ is the immediate reward the player obtains when taking command $(a,b)$ in state $h$. We consider discounted accumulated rewards where $\gamma$ is the discount factor. In particular, the game state $h$ is hidden from the player, who only receives a varying textual description. Let $S$ denote the space of all possible text descriptions. The text descriptions $s$ observed by the player are produced by a stochastic function $\Psi :H\rightarrow S$. Assume that each observable state $s\in S$ is associated with a **unique** hidden state, denoted by $h(s)\in H$.

You will conduct experiments on a small Home World, which mimic the environment of a typical house. The world consists of four rooms- a living room, a bed room, a kitchen and a garden with connecting pathways (illustrated in figure below). Transitions between the rooms are deterministic. Each room contains a representative object that the player can interact with. For instance, the living room has a TV that the player can watch , and the kitchen has an apple that the player can eat. Each room has several descriptions, invoked randomly on each visit by the player.

**Rooms and objects in the Home world with connecting pathways**


![image](https://courses.edx.org/assets/courseware/v1/24148b9ea8dfaef68148bb9db4c196aa/asset-v1:MITx+6.86x+1T2021+type@asset+block/images_homeworld.jpg)



| Positive | Negative |
| ------------- | ------------- |
| Quest goal: +1  | Negative per step: -0.01|
|                 | Invalid command: -0.1  |

At the beginning of each episode, the player is placed at a random room and provided with a randomly selected quest. An example of a quest given to the player in text is You are hungry now. To complete this quest, the player has to navigate through the house to reach the kitchen and eat the apple (i.e., type in command eat apple). In this game, the room is hidden from the player, who only receives a description of the underlying room. The underlying game state is given by $h=(q,r)$, where $r$ is the index of room and $q$ is the index of quest. At each step, the text description $s$ provided to the player contains two parts $s=(s_r ; s_q)$, where $s_r$ is the room description (which are varied and randomly provided) and $s_q$ is the quest description. The player receives a positive reward on completing a quest, and negative rewards for invalid command (e.g., eat TV). Each non-terminating step incurs a small deterministic negative rewards, which incentives the player to learn policies that solve quests in fewer steps. (see the Table 1) An episode ends when the player finishes the quest or has taken more steps than a fixed maximum number of steps.

Each episode produces a full record of interaction $(h_{0},s_{0},a_{0},b_{0},r_{0},\ldots ,h_{t},s_{t},a_{t},b_{t},r_{t},h_{t+1}\ldots)$ where $h_{0}=(h_{r,0},h_{q,0})\sim \Gamma_{0}$ ($\Gamma_{0}$ denotes an initial state distribution), $h_{t}\sim P(\cdot |h_{t-1},a_{t-1},b_{t-1})$, $s_{t}\sim \Psi (h_{t})$, $r_{t}=R(h_{t},a_{t},b_{t})$
