import numpy

class Cube:
    def __init__(self):
        # -------------- #
        # --- States --- #
        # -------------- #

        # Colours - top: green, front: white, right: orange, left: red, back: yellow, bottom: blue
        # 20 cubelets considered (8 corner cubelets, 12 edge cubelets, ignore 6 centre cubelets)
        # Corner cubelets can be in 8 places and in 3 orientations -> 8*3=24 locations
        # Edge cubelets can be in 12 places and in 2 orientations -> 12*2=24 locations
        # Shape: 20x24 (rows represent cubelets with corners first then edges, columns represent locations)
        self.goal_state = numpy.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  0: top-left-front corner (green-red-white), red-white-green, white-green-red
                                       [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  1: top-front-right corner (green-white-orange), white-orange-green, orange-green-white
                                       [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  2: top-right-back corner (green-orange-yellow), orange-yellow-green, yellow-green-orange
                                       [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  3: top-back-left corner (green-yellow-red), yellow-red-green, red-green-yellow
                                       [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], #  4: bottom-front-left corner (blue-white-red), white-red-blue, red-blue-white
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], #  5: bottom-right-front corner (blue-orange-white), orange-white-blue, white-blue-orange
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], #  6: bottom-back-right corner (blue-yellow-orange), yellow-orange-blue, orange-blue-yellow
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], #  7: bottom-left-back corner (blue-red-yellow), red-yellow-blue, yellow-blue-red
                                       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  8: top-front edge (green-white), white-green
                                       [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #  9: top-right edge (green-orange), orange-green
                                       [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 10: top-back edge (green-yellow), yellow-green
                                       [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 11: top-left edge (green-red), red-green
                                       [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 12: left-front edge (red-white), white-red
                                       [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # 13: front-right edge (white-orange), orange-white
                                       [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], # 14: right-back edge (orange-yellow), yellow-orange
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], # 15: back-left edge (yellow-red), red-yellow
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], # 16: bottom-front edge (blue-white), white-blue
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], # 17: bottom-right edge (blue-orange), orange-blue
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], # 18: bottom-back edge (blue-yellow), yellow-blue
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]  # 19: bottom-left edge (blue-red), red-blue
                                      ])

        # ------------------------------- #
        # --- Actions and transitions --- #
        # ------------------------------- #

        # action face - 1: top, 2: front, 3: right, 4: left, 5: back, 6: bottom
        # action rotation - 1: clockwise, 2: counter-clockwise

        # These variables map the face to the locations on the face
        # A face has 8 cubelets
        # There are 4 corner cubelets and 3 possible orientations for each, 4*3=12
        # There are 4 edge cubelets and 2 possible orientations for each, 4*2=8
        self.face_location_mapping_corners = {1: [0,1,2,3,4,5,6,7,8,9,10,11],
                                              2: [0,1,2,3,4,5,12,13,14,15,16,17],
                                              3: [3,4,5,6,7,8,15,16,17,18,19,20],
                                              4: [0,1,2,9,10,11,12,13,14,21,22,23],
                                              5: [6,7,8,9,10,11,18,19,20,21,22,23],
                                              6: [12,13,14,15,16,17,18,19,20,21,22,23]
                                             }
        self.face_location_mapping_edges = {1: [0,1,2,3,4,5,6,7],
                                            2: [0,1,8,9,10,11,16,17],
                                            3: [2,3,10,11,12,13,18,19],
                                            4: [6,7,8,9,14,15,22,23],
                                            5: [4,5,12,13,14,15,20,21],
                                            6: [16,17,18,19,20,21,22,23]
                                           }
        
        # These variables map locations (from the last 2 variables) to new locations based on rotation
        self.clockwise_mapping_corners = {1: [9,10,11,0,1,2,3,4,5,6,7,8],
                                          2: [4,5,3,17,15,16,2,0,1,13,14,12],
                                          3: [7,8,6,20,18,19,5,3,4,16,17,15],
                                          4: [14,12,13,1,2,0,22,23,21,11,9,10],
                                          5: [10,11,9,23,21,22,8,6,7,19,20,18],
                                          6: [15,16,17,18,19,20,21,22,23,12,13,14]
                                         }
        self.counter_clockwise_mapping_corners = {1: [3,4,5,6,7,8,9,10,11,0,1,2],
                                                  2: [13,14,12,2,0,1,17,15,16,4,5,3],
                                                  3: [16,17,15,5,3,4,20,18,19,7,8,6],
                                                  4: [11,9,10,22,23,21,1,2,0,14,12,13],
                                                  5: [19,20,18,8,6,7,23,21,22,10,11,9],
                                                  6: [21,22,23,12,13,14,15,16,17,18,19,20]
                                                 }
        self.clockwise_mapping_edges = {1: [6,7,0,1,2,3,4,5],
                                        2: [11,10,0,1,17,16,8,9],
                                        3: [13,12,2,3,19,18,10,11],
                                        4: [9,8,23,22,6,7,14,15],
                                        5: [15,14,4,5,21,20,12,13],
                                        6: [18,19,20,21,22,23,16,17]
                                       }
        self.counter_clockwise_mapping_edges = {1: [2,3,4,5,6,7,0,1],
                                                2: [8,9,16,17,1,0,11,10],
                                                3: [10,11,18,19,3,2,13,12],
                                                4: [14,15,7,6,22,23,9,8],
                                                5: [12,13,20,21,5,4,15,14],
                                                6: [22,23,16,17,18,19,20,21]
                                               }

        # --------------- #
        # --- Rewards --- #
        # --------------- #

        self.reward_goal_state = 1
        self.reward_not_goal_state = -1
    
    # Rotate function
    def rotate(self, current_state, action_face, action_rotation):
        state = numpy.copy(current_state)

        # Determine which cubelets will be affected
        for i in range(20): # test each cubelet
            # Get the location of the cubelet (where the 1 is)
            current_location = numpy.argmax(current_state[i])

            # Different processing based on whether the cubelet is a corner or edge cubelet
            if i < 8 and current_location in self.face_location_mapping_corners[action_face]: # this is a corner cubelet and it is in the action face, so it will be affected
                # Find the new location based on the action (face and rotation)
                to_map = self.face_location_mapping_corners[action_face].index(current_location) # index of current location in face_location_mapping_corners
                if action_rotation == 1: # clockwise
                    location = self.clockwise_mapping_corners[action_face][to_map]
                elif action_rotation == 2: # counter-clockwise
                    location = self.counter_clockwise_mapping_corners[action_face][to_map]
                
                # Move the cubelet (move the 1 to the new location)
                state[i][current_location] = 0
                state[i][location] = 1
            elif i >=8 and current_location in self.face_location_mapping_edges[action_face]: # this is an edge cubelet and it is in the action face, so it will be affected
                # Find the new location based on the action (face and rotation)
                to_map = self.face_location_mapping_edges[action_face].index(current_location) # index of current location in face_location_mapping_edges
                if action_rotation == 1: # clockwise
                    location = self.clockwise_mapping_edges[action_face][to_map]
                elif action_rotation == 2: # counter-clockwise
                    location = self.counter_clockwise_mapping_edges[action_face][to_map]
                
                # Move the cubelet (move the 1 to the new location)
                state[i][current_location] = 0
                state[i][location] = 1

        return state
    
    # Check whether a given state is the goal state
    def check_goal_state(self, state):
        if numpy.array_equal(state, self.goal_state):
            return True
        else:
            return False
    
    # Get a reward for the current state
    def get_reward(self, state):
        if self.check_goal_state(state):
            return self.reward_goal_state
        else:
            return self.reward_not_goal_state
