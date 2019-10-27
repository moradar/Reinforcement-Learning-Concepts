import numpy as np
import matplotlib.pyplot as plt

#   Stair Climbing Markov Decision Process and Policy Implementation
#   Reinforcement Learning Labs


class StairClimbingMDP(object):
    def __init__(self):
        # States are:  { s1 <-- s2 <=> s3 <=> s4 <=> s5 <=> s6 --> s7 ]
        self.S = 7
        self.state_names = ['s1', 's2', 's3', 's4', 's5', 's6', 's7']

        # Actions are: {L,R} --> {0, 1} Left is bottom, right is up.
        self.A = 2
        self.action_names = ['L', 'R']

        # Matrix indicating absorbing states
        # B  1   2   3   4   5  6  7  T   <-- STATES
        self.absorbing = [1, 0, 0, 0, 0, 0, 1]

        # Load transition
        self.T = self.transition_matrix()

        # Load reward matrix
        self.R = self.reward_matrix()

    # Get the transition matrix
    def transition_matrix(self):
        # TL is going down
        #               1    ...    7 <-- FROM STATE
        TL = np.array([[1, 1, 0, 0, 0, 0, 0],  # 1 TO STATE
                       [0, 0, 1, 0, 0, 0, 0],  # .
                       [0, 0, 0, 1, 0, 0, 0],  # .
                       [0, 0, 0, 0, 1, 0, 0],  #
                       [0, 0, 0, 0, 0, 1, 0],  #
                       [0, 0, 0, 0, 0, 0, 0],  #
                       [0, 0, 0, 0, 0, 0, 0]])  # 7

        # TR is going up
        #               1    ...    7 <-- FROM STATE
        TR = np.array([[0, 0, 0, 0, 0, 0, 0],  # 1 TO STATE
                       [0, 0, 0, 0, 0, 0, 0],  #
                       [0, 1, 0, 0, 0, 0, 0],  # .
                       [0, 0, 1, 0, 0, 0, 0],  # .
                       [0, 0, 0, 1, 0, 0, 0],  # .
                       [0, 0, 0, 0, 1, 0, 0],  #
                       [0, 0, 0, 0, 0, 1, 1]])  # 7

        # transition_matrix[#row, #column, #action_id]
        return np.dstack([TL, TR])  # transition probabilities for each action

    # Transition subfunction
    def transition_function(self, prior_state, action, post_state):
        # Reward function (defined locally)
        prob = self.T(post_state, prior_state, action)
        return prob

    # Get the reward matrix
    def reward_matrix(self, S=None, A=None):
        # i.e. 11x11 matrix of rewards for being in state s,
        # performing action a and ending in state s'

        if S is None:
            S = self.S
        if A is None:
            A = self.A

        R = np.zeros((S, S, A))

        for i in range(S):
            for j in range(A):
                for k in range(S):
                    R[k, i, j] = self.reward_function(i, j, k)

        return R

    # Reward function (local)
    def reward_function(self, prior_state, action, post_state):
        if (prior_state == 1) and (action == 0) and (post_state == 0):
            rew = -100
        elif (prior_state == 5) and (action == 1) and (post_state == 6):
            rew = 100
        elif action == 0:
            rew = 10
        else:
            rew = -10

        return rew

    # Next state identification
    def next_state(self, state, action):
        # Returns the next TO STATE for specified action (in)
        state = np.argmax(self.T[:, state, action])
        return state

    # Policy value function
    def policy_value(self, state, gamma = 0, policy = [0, 0, 0, 0, 0, 0, 0]):
        val = 0
        discount = 1
        for step in range(500): # To prevent infinite looping
            # Absorbing state - end sequence
            if self.absorbing[state]:
                break

            # Compound rewards obtained in each state step
            action = policy[state]
            post_st = self.next_state(state, action)
            val += discount*self.reward_function(state, action, post_st)

            # Update state and discount
            discount *= gamma
            state = post_st

        return val

    # Plotting the value return of a policy depending on the varying gamma
    def gamma_opt(self, init_st, policies):
        plt.figure()
        for policy in policies:
            values = []
            gammas = [1/g for g in range(1, 100)]
            for gamma in gammas:
                values.append(self.policy_value(init_st, gamma, policy))
            plt.plot(gammas, values, label= 'Policy {}'.format(policy))
        plt.legend()
        plt.show()


UpPolicy = [1, 1, 1, 1, 1, 1, 1]
DownPolicy = [0, 0, 0, 0, 0, 0, 0]
GreedyPolicy = [0, 0, 0, 1, 1, 1, 0]

stairsim = StairClimbingMDP()
gamma = 1
init_step_up = 1
init_step_down = 5

# Test outcomes for start at step furthest from goal
policy_up = stairsim.policy_value(init_step_up, gamma, UpPolicy)
policy_down = stairsim.policy_value(init_step_down, gamma, DownPolicy)

# Test the value for designated greedy policy
greedyval = stairsim.policy_value(3, gamma, GreedyPolicy)

# Plot the values of Up and Down policies with varying gamma
stairsim.gamma_opt(3, [UpPolicy, DownPolicy])
