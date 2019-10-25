import numpy as np
import random as rm

# Basic Markov Decision Process Implementation - The Student MRP
# Reinforcement Learning Lab


# Definition of state space (using dictionary)
state_space = {'Class 1': 0,
               'Class 2': 1,
               'Class 3': 2,
               'Pass': 3,
               'Sleep': 4,
               'Facebook': 5,
               'Pub': 6
               }

# Transition matrix
transition_matrix = np.array([[0, 0.5, 0, 0, 0, 0.5, 0],
                              [0, 0, 0.8, 0, 0.2, 0, 0],
                              [0, 0, 0, 0.6, 0, 0, 0.4],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0.1, 0, 0, 0, 0, 0.9, 0],
                              [0.2, 0.4, 0.4, 0, 0, 0, 0]])

# Reward matrix
reward_matrix = np.array([[-2], [-2], [-2], [10], [0], [-1], [1]])


# Return state transition probability
def p_transition(statename):
    pt = transition_matrix[state_space[statename]]
    return pt


# Return the immediate reward
def imm_reward(statename):
    ir = reward_matrix[state_space[statename]]
    return ir


# Search the state name by identifier
def search_state(state_dict, search_id):
    ans = 0
    for state, ids in state_dict.items():
        if ids == search_id:
            ans = state
    return ans


# Search through transition state matrix to decide on the next node
def choose_next(statename, state_dict):
    x = rm.random()                             # Random number generator
    pcount = 0                                  # Probability counter
    idi = -1                                    # Indice counter
    prob_trans = p_transition(statename)

    while x > pcount:
        idi += 1
        pcount += prob_trans[idi]

    name_s = search_state(state_dict, idi)
    reward_s = imm_reward(name_s)

    return name_s, reward_s


# Calculate the total return of the completed trace
def calc_tot_return(path):
    gamma = 0.5
    tot_ret = 0
    i = 0
    for x, y in path:
        tot_ret += (gamma**i)*y
        i += 1
    return tot_ret


# Calculate the return at each traversed state
def calc_step_return(path):
    gamma = 1
    step_ret = []
    i = 0
    for x, y in path:
        step_ret.append((gamma**i)*y)
        i += 1
    return step_ret

# Calculate the return (state-specific)
def calc_state_return(path, s_val_arr):
    n_visits = np.zeros(7)
    gamma = 1
    i = 0
    for statename, rew in path:
        idi = state_space[statename]
        n_visits[idi] += 1
        s_val_arr[idi] += (gamma*n_visits[idi])*rew
        i += 1
    s_val_arr = np.true_divide(s_val_arr, n_visits, where=n_visits!=0)
    return s_val_arr


# Initial conditions
starts = 0
init_s = search_state(state_space, starts)       # Start state name
ends = 4                                         # End state ID
endit_s = search_state(state_space, ends)        # End state name

path_trace = [(init_s, imm_reward(init_s))]
state_value = np.zeros(7)
curr_node = init_s
step = 0
state_return = np.zeros(7)

print('The current episode is set to start at node {} and end at node {}. '.format(init_s, endit_s))

repeats = input('How many traces do you want to average? :')

for a in range(int(repeats)):
    # Simulating a trace
    while curr_node != endit_s:
        path_trace.append(choose_next(curr_node, state_space))
        step += 1
        curr_node = path_trace[step][0]

    # Printing the trace
    print('The path taken is:')
    for name, reward in path_trace:
        print('Node: {}, Reward: {}'.format(name, reward))

    # Evaluating total return
    total_return = calc_tot_return(path_trace)
    state_return = np.add(state_return, calc_state_return(path_trace, state_value))
    print(total_return)
    del path_trace[:]
    del state_value
    path_trace = [(init_s, imm_reward(init_s))]
    state_value = np.zeros(7)
    curr_node = init_s
    step = 0

state_return = np.true_divide(state_return, float(repeats))
print(state_return)