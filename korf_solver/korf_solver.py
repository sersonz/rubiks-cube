# ------------------- #
# --- Information --- #
# ------------------- #

# Paper:
# - Finding Optimal Solutions to Rubik's Cube Using Pattern Databases (https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf)

# Useful resources:
# - https://www.geeksforgeeks.org/iterative-deepening-a-algorithm-ida-artificial-intelligence/
# - https://stackoverflow.com/questions/12864004/tracing-and-returning-a-path-in-depth-first-search

# --------------- #
# --- Imports --- #
# --------------- #

import numpy as np
import sqlite3
import time
import random
from korf_environment import *

# ----------------------------------- #
# --- Connect to pattern database --- #
# ----------------------------------- #

connection = sqlite3.connect('korf_db.sqlite')
cursor = connection.cursor()

# ---------------------------------------------------------------------- #
# --- Iterative deepening A* search (IDA*) to solve the Rubik's Cube --- #
# ---------------------------------------------------------------------- #

def ida_star(scrambled_state):
    # Get depth information from databases and calculate heuristic function
    initial_distance = 0 # root node
    p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20 = state_to_int(scrambled_state)
    depth_1 = cursor.execute('SELECT depth FROM corners1 WHERE p1 == ' + str(p1) + ' AND p2 == ' + str(p2) + ' AND p3 == ' + str(p3) + ' AND p4 == ' + str(p4)).fetchone()[0]
    depth_2 = cursor.execute('SELECT depth FROM corners2 WHERE p1 == ' + str(p5) + ' AND p2 == ' + str(p6) + ' AND p3 == ' + str(p7) + ' AND p4 == ' + str(p8)).fetchone()[0]
    depth_3 = cursor.execute('SELECT depth FROM edges1 WHERE p1 == ' + str(p9) + ' AND p2 == ' + str(p10) + ' AND p3 == ' + str(p11) + ' AND p4 == ' + str(p12)).fetchone()[0]
    depth_4 = cursor.execute('SELECT depth FROM edges2 WHERE p1 == ' + str(p13) + ' AND p2 == ' + str(p14) + ' AND p3 == ' + str(p15) + ' AND p4 == ' + str(p16)).fetchone()[0]
    depth_5 = cursor.execute('SELECT depth FROM edges3 WHERE p1 == ' + str(p17) + ' AND p2 == ' + str(p18) + ' AND p3 == ' + str(p19) + ' AND p4 == ' + str(p20)).fetchone()[0]
    depth_estimate = max([depth_1, depth_2, depth_3, depth_4, depth_5])
    heuristic = initial_distance + depth_estimate

    pruned = []
    pruned.append(heuristic)
    while True:
        # Stack setup
        stack = []
        distances = []
        previous_actions = []
        stack.append(scrambled_state)
        distances.append(initial_distance)
        previous_actions.append([0,0])

        # Child-parent map (to track path)
        child_to_parent_map = {}

        # Pruning threshold
        threshold = min(pruned)
        pruned = [i for i in pruned if i != threshold]

        # Depth-first search
        while stack:
            # Get state from stack
            current_state = stack.pop()
            current_distance = distances.pop()
            current_action = previous_actions.pop()
            current_action_face = current_action[0]

            # Find possible actions from the current state
            possible_actions = possible_action_mapping[current_action_face]
            for action in possible_actions:
                state = rotate(current_state, action[0], action[1])

                if check_goal_state(state):
                    state_tuple = state_to_int(state)
                    child_to_parent_map[state_tuple] = []
                    child_to_parent_map[state_tuple].append(state_to_int(current_state))
                    child_to_parent_map[state_tuple].append(action)
                    return child_to_parent_map

                if np.array_equal(state, scrambled_state):
                    continue

                # Get depth information from databases and calculate heuristic function
                p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20 = state_to_int(state)
                depth_1 = cursor.execute('SELECT depth FROM corners1 WHERE p1 == ' + str(p1) + ' AND p2 == ' + str(p2) + ' AND p3 == ' + str(p3) + ' AND p4 == ' + str(p4)).fetchone()[0]
                depth_2 = cursor.execute('SELECT depth FROM corners2 WHERE p1 == ' + str(p5) + ' AND p2 == ' + str(p6) + ' AND p3 == ' + str(p7) + ' AND p4 == ' + str(p8)).fetchone()[0]
                depth_3 = cursor.execute('SELECT depth FROM edges1 WHERE p1 == ' + str(p9) + ' AND p2 == ' + str(p10) + ' AND p3 == ' + str(p11) + ' AND p4 == ' + str(p12)).fetchone()[0]
                depth_4 = cursor.execute('SELECT depth FROM edges2 WHERE p1 == ' + str(p13) + ' AND p2 == ' + str(p14) + ' AND p3 == ' + str(p15) + ' AND p4 == ' + str(p16)).fetchone()[0]
                depth_5 = cursor.execute('SELECT depth FROM edges3 WHERE p1 == ' + str(p17) + ' AND p2 == ' + str(p18) + ' AND p3 == ' + str(p19) + ' AND p4 == ' + str(p20)).fetchone()[0]
                depth_estimate = max([depth_1, depth_2, depth_3, depth_4, depth_5])
                new_heuristic = (current_distance+1) + depth_estimate

                if new_heuristic > threshold:
                    pruned.append(new_heuristic)
                    continue

                # Add to stack
                stack.append(state)
                distances.append(current_distance+1)
                previous_actions.append(action)
                state_tuple = state_to_int(state)
                child_to_parent_map[state_tuple] = []
                child_to_parent_map[state_tuple].append(state_to_int(current_state))
                child_to_parent_map[state_tuple].append(action)

def get_action_path(child_to_parent_map):
    actions = []
    current_state = state_to_int(goal_state)
    while True:
        if current_state not in child_to_parent_map:
            break
        parent = child_to_parent_map[current_state]
        current_state = parent[0]
        actions.insert(0, parent[1])
    return actions

# ------------------- #
# --- Test solver --- #
# ------------------- #

def scramble_and_solve(scrambles):
    print('1) Scramble the cube')
    print('Number of scrambles:', len(scrambles))
    print('Scrambles:', scrambles)
    scrambled_state = goal_state.copy()
    for scramble in scrambles:
        scrambled_state = rotate(scrambled_state, scramble[0], scramble[1])

    print('2) Get the solution')
    start = time.time()
    actions = get_action_path(ida_star(scrambled_state))
    end = time.time()
    elapsed_time = end - start
    n_moves_solution = len(actions)
    print('Number of moves to solve:', n_moves_solution)
    print('Time to solve (seconds):', round(elapsed_time, 2))
    print('Moves to solve:', actions)

    print('3) Solve the cube')
    solved_state = scrambled_state.copy()
    for action in actions:
        solved_state = rotate(solved_state, action[0], action[1])
    print('Solved:', check_goal_state(solved_state))

    return elapsed_time, n_moves_solution

def get_random_scrambles(n_scrambles):
    scrambles = []
    previous_action_face = 0
    for i in range(n_scrambles):
        possible_actions = possible_action_mapping[previous_action_face]
        action = random.choice(possible_actions)
        previous_action_face = action[0]
        scrambles.append(action)
    return scrambles

def run_multiple_tests(n_scrambles, n_tests):
    elapsed_times = []
    n_moves = []
    for i in range(n_tests):
        print('----------')
        print('Test', i+1)
        print('----------')
        scrambles = get_random_scrambles(n_scrambles)
        elapsed_time, n_moves_solution = scramble_and_solve(scrambles)
        elapsed_times.append(elapsed_time)
        n_moves.append(n_moves_solution)
        print()
    print('Average time to solve (seconds):', round(sum(elapsed_times)/n_tests, 2))
    print('Average number of moves to solve:', round(sum(n_moves)/n_tests, 2))
    return

# result always has same number of moves as scrambles (otherwise too slow)
n_scrambles = 3
n_tests = 10
run_multiple_tests(n_scrambles, n_tests)

# --------------------------------- #
# --- Close database connection --- #
# --------------------------------- #

connection.close()
