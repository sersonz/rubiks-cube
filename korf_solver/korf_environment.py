# ------------------- #
# --- Information --- #
# ------------------- #

# Paper:
# - Finding Optimal Solutions to Rubik's Cube Using Pattern Databases (https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf)

# --------------- #
# --- Imports --- #
# --------------- #

import numpy as np

# -------------- #
# --- States --- #
# -------------- #

# Colours - top: green, front: white, right: orange, left: red, back: yellow, bottom: blue
# 20 cubelets considered (8 corner cubelets, 12 edge cubelets, ignore 6 centre cubelets)
# Corner cubelets can be in 8 places and in 3 orientations -> 8*3=24 locations
# Edge cubelets can be in 12 places and in 2 orientations -> 12*2=24 locations
# Shape: 20x24 (rows represent cubelets with corners first then edges, columns represent locations)
goal_state = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  0: top-left-front corner (green-red-white), red-white-green, white-green-red
                       [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  1: top-front-right corner (green-white-orange), white-orange-green, orange-green-white
                       [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  2: top-right-back corner (green-orange-yellow), orange-yellow-green, yellow-green-orange
                       [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  3: top-back-left corner (green-yellow-red), yellow-red-green, red-green-yellow
                       [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], #  4: bottom-front-left corner (blue-white-red), white-red-blue, red-blue-white
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], #  5: bottom-right-front corner (blue-orange-white), orange-white-blue, white-blue-orange
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], #  6: bottom-back-right corner (blue-yellow-orange), yellow-orange-blue, orange-blue-yellow
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], #  7: bottom-left-back corner (blue-red-yellow), red-yellow-blue, yellow-blue-red
                       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  8: top-front edge (green-white), white-green
                       [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  9: right-top edge (orange-green), green-orange
                       [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 10: back-top edge (yellow-green), green-yellow
                       [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 11: top-left edge (green-red), red-green
                       [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 12: left-front edge (red-white), white-red
                       [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # 13: front-right edge (white-orange), orange-white
                       [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], # 14: right-back edge (orange-yellow), yellow-orange
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], # 15: back-left edge (yellow-red), red-yellow
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], # 16: front-bottom edge (white-blue), blue-white
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], # 17: bottom-right edge (blue-orange), orange-blue
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], # 18: bottom-back edge (blue-yellow), yellow-blue
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]  # 19: left-bottom edge (red-blue), blue-red
                      ])

# Check whether a given state is the goal state
def check_goal_state(state):
    if np.array_equal(state, goal_state):
        return True
    else:
        return False

# ------------------------------- #
# --- Actions and transitions --- #
# ------------------------------- #

# Action face - 1: top, 2: front, 3: right, 4: left, 5: back, 6: bottom
# Action rotation - 1: clockwise, 2: counter-clockwise, 3: 180 degrees

# This variable contains a mapping from the previous action face to the possible actions
# First move (None/0): 18 possible actions, after that: 15 possible actions (don't repeat the face)
# For opposite faces, choose an order that they can be moved in and don't allow the other way around (top/bottom, front/back, right/left)
possible_action_mapping = {0: [[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3], [4,1], [4,2], [4,3], [5,1], [5,2], [5,3], [6,1], [6,2], [6,3]],
                           1: [[2,1], [2,2], [2,3], [3,1], [3,2], [3,3], [4,1], [4,2], [4,3], [5,1], [5,2], [5,3], [6,1], [6,2], [6,3]],
                           2: [[1,1], [1,2], [1,3], [3,1], [3,2], [3,3], [4,1], [4,2], [4,3], [5,1], [5,2], [5,3], [6,1], [6,2], [6,3]],
                           3: [[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [4,1], [4,2], [4,3], [5,1], [5,2], [5,3], [6,1], [6,2], [6,3]],
                           4: [[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [5,1], [5,2], [5,3], [6,1], [6,2], [6,3]],
                           5: [[1,1], [1,2], [1,3], [3,1], [3,2], [3,3], [4,1], [4,2], [4,3], [6,1], [6,2], [6,3]],
                           6: [[2,1], [2,2], [2,3], [3,1], [3,2], [3,3], [4,1], [4,2], [4,3], [5,1], [5,2], [5,3]]
                          }

# These variables map the face to the locations on the face
# A face has 8 cubelets
# There are 4 corner cubelets and 3 possible orientations for each, 4*3=12
# There are 4 edge cubelets and 2 possible orientations for each, 4*2=8
face_location_mapping_corners = {1: [0,1,2,3,4,5,6,7,8,9,10,11],
                                 2: [0,1,2,3,4,5,12,13,14,15,16,17],
                                 3: [3,4,5,6,7,8,15,16,17,18,19,20],
                                 4: [0,1,2,9,10,11,12,13,14,21,22,23],
                                 5: [6,7,8,9,10,11,18,19,20,21,22,23],
                                 6: [12,13,14,15,16,17,18,19,20,21,22,23]
                                }
face_location_mapping_edges = {1: [0,1,2,3,4,5,6,7],
                               2: [0,1,8,9,10,11,16,17],
                               3: [2,3,10,11,12,13,18,19],
                               4: [6,7,8,9,14,15,22,23],
                               5: [4,5,12,13,14,15,20,21],
                               6: [16,17,18,19,20,21,22,23]
                              }

# These variables map locations (from the last 2 variables) to new locations based on rotation
clockwise_mapping_corners = {1: [9,10,11,0,1,2,3,4,5,6,7,8],
                             2: [4,5,3,17,15,16,2,0,1,13,14,12],
                             3: [7,8,6,20,18,19,5,3,4,16,17,15],
                             4: [14,12,13,1,2,0,22,23,21,11,9,10],
                             5: [10,11,9,23,21,22,8,6,7,19,20,18],
                             6: [15,16,17,18,19,20,21,22,23,12,13,14]
                            }
counter_clockwise_mapping_corners = {1: [3,4,5,6,7,8,9,10,11,0,1,2],
                                     2: [13,14,12,2,0,1,17,15,16,4,5,3],
                                     3: [16,17,15,5,3,4,20,18,19,7,8,6],
                                     4: [11,9,10,22,23,21,1,2,0,14,12,13],
                                     5: [19,20,18,8,6,7,23,21,22,10,11,9],
                                     6: [21,22,23,12,13,14,15,16,17,18,19,20]
                                    }
mapping_180_corners = {1: [6,7,8,9,10,11,0,1,2,3,4,5],
                       2: [15,16,17,12,13,14,3,6,7,0,1,2],
                       3: [18,19,20,15,16,17,6,7,8,3,4,5],
                       4: [21,22,23,12,13,14,9,10,11,0,1,2],
                       5: [21,22,23,18,19,20,9,10,11,6,7,8],
                       6: [18,19,20,21,22,23,12,13,14,15,16,17]
                      }
clockwise_mapping_edges = {1: [6,7,1,0,2,3,5,4],
                           2: [11,10,0,1,16,17,9,8],
                           3: [12,13,3,2,19,18,10,11],
                           4: [9,8,22,23,6,7,15,14],
                           5: [14,15,5,4,21,20,12,13],
                           6: [19,18,20,21,23,22,16,17]
                          }
counter_clockwise_mapping_edges = {1: [3,2,4,5,7,6,0,1],
                                   2: [8,9,17,16,1,0,10,11],
                                   3: [11,10,18,19,2,3,13,12],
                                   4: [14,15,7,6,23,22,8,9],
                                   5: [13,12,20,21,4,5,15,14],
                                   6: [22,23,17,16,18,19,21,20]
                                  }
mapping_180_edges = {1: [5,4,7,6,1,0,3,2],
                     2: [17,16,11,10,9,8,1,0],
                     3: [19,18,13,12,11,10,3,2],
                     4: [23,22,15,14,9,8,7,6],
                     5: [21,20,15,14,13,12,5,4],
                     6: [21,20,23,22,17,16,19,18]
                    }

# Rotate function
def rotate(current_state, action_face, action_rotation):
    state = np.copy(current_state)

    # Determine which cubelets will be affected
    for i in range(20): # test each cubelet
        # Get the location of the cubelet (where the 1 is)
        current_location = np.argmax(current_state[i])

        # Different processing based on whether the cubelet is a corner or edge cubelet
        if i < 8 and current_location in face_location_mapping_corners[action_face]: # this is a corner cubelet and it is in the action face, so it will be affected
            # Find the new location based on the action (face and rotation)
            to_map = face_location_mapping_corners[action_face].index(current_location) # index of current location in face_location_mapping_corners
            if action_rotation == 1: # clockwise
                location = clockwise_mapping_corners[action_face][to_map]
            elif action_rotation == 2: # counter-clockwise
                location = counter_clockwise_mapping_corners[action_face][to_map]
            elif action_rotation == 3: # 180 degrees
                location = mapping_180_corners[action_face][to_map]

            # Move the cubelet (move the 1 to the new location)
            state[i][current_location] = 0
            state[i][location] = 1
        elif i >=8 and current_location in face_location_mapping_edges[action_face]: # this is an edge cubelet and it is in the action face, so it will be affected
            # Find the new location based on the action (face and rotation)
            to_map = face_location_mapping_edges[action_face].index(current_location) # index of current location in face_location_mapping_edges
            if action_rotation == 1: # clockwise
                location = clockwise_mapping_edges[action_face][to_map]
            elif action_rotation == 2: # counter-clockwise
                location = counter_clockwise_mapping_edges[action_face][to_map]
            elif action_rotation == 3: # 180 degrees
                location = mapping_180_edges[action_face][to_map]

            # Move the cubelet (move the 1 to the new location)
            state[i][current_location] = 0
            state[i][location] = 1

    return state

# Rotate function for corners only
def rotate_corners(current_state, action_face, action_rotation):
    state = np.copy(current_state)

    # Determine which cubelets will be affected
    for i in range(len(current_state)): # test each cubelet
        # Get the location of the cubelet (where the 1 is)
        current_location = np.argmax(current_state[i])

        if current_location in face_location_mapping_corners[action_face]: # the cubelet is in the action face, so it will be affected
            # Find the new location based on the action (face and rotation)
            to_map = face_location_mapping_corners[action_face].index(current_location) # index of current location in face_location_mapping_corners
            if action_rotation == 1: # clockwise
                location = clockwise_mapping_corners[action_face][to_map]
            elif action_rotation == 2: # counter-clockwise
                location = counter_clockwise_mapping_corners[action_face][to_map]
            elif action_rotation == 3: # 180 degrees
                location = mapping_180_corners[action_face][to_map]

            # Move the cubelet (move the 1 to the new location)
            state[i][current_location] = 0
            state[i][location] = 1

    return state

# Rotate function for edges only
def rotate_edges(current_state, action_face, action_rotation):
    state = np.copy(current_state)

    # Determine which cubelets will be affected
    for i in range(len(current_state)): # test each cubelet
        # Get the location of the cubelet (where the 1 is)
        current_location = np.argmax(current_state[i])

        if current_location in face_location_mapping_edges[action_face]: # the cubelet is in the action face, so it will be affected
            # Find the new location based on the action (face and rotation)
            to_map = face_location_mapping_edges[action_face].index(current_location) # index of current location in face_location_mapping_edges
            if action_rotation == 1: # clockwise
                location = clockwise_mapping_edges[action_face][to_map]
            elif action_rotation == 2: # counter-clockwise
                location = counter_clockwise_mapping_edges[action_face][to_map]
            elif action_rotation == 3: # 180 degrees
                location = mapping_180_edges[action_face][to_map]

            # Move the cubelet (move the 1 to the new location)
            state[i][current_location] = 0
            state[i][location] = 1

    return state

# --------------------------------- #
# --- Database helper functions --- #
# --------------------------------- #

# Convert state (array) to int (location)
def state_to_int(state):
    permutations = []
    for i in range(len(state)):
        permutations.append(int(np.argmax(state[i])))
    return tuple(permutations)

# Convert int (location) to state (array)
def int_to_state(permutations):
    state = np.zeros((len(permutations), 24), dtype=np.int64)
    for i in range(len(permutations)):
        state[i][permutations[i]] = 1
    return state