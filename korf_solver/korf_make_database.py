# ------------------- #
# --- Information --- #
# ------------------- #

# Paper:
# - Finding Optimal Solutions to Rubik's Cube Using Pattern Databases (https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf)

# Useful resources:
# - https://drlee.io/crafting-a-database-in-google-colab-a-beginners-guide-ccfb5e3ba22d
# - https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/

# --------------- #
# --- Imports --- #
# --------------- #

import numpy as np
import sqlite3
from korf_environment import *

# ----------------------------------- #
# --- Connect to pattern database --- #
# ----------------------------------- #

connection = sqlite3.connect('korf_db.sqlite') # will generate the database file if it does not exist
cursor = connection.cursor()


# ---------------------------------------------------------------------------------- #
# --- Breadth-first search (BFS) to make pattern database tables (ONLY RUN ONCE) --- #
# ---------------------------------------------------------------------------------- #

def make_tables():
    # Corners 1-5
    cursor.execute('CREATE TABLE corners1 (queue INT, action INT, depth INT, p1 INT, p2 INT, p3 INT, p4 INT, PRIMARY KEY (p1,p2,p3,p4))')
    cursor.execute('CREATE UNIQUE INDEX corners1idx ON corners1 (queue)')

    # Corners 6-8
    cursor.execute('CREATE TABLE corners2 (queue INT, action INT, depth INT, p1 INT, p2 INT, p3 INT, p4 INT, PRIMARY KEY (p1,p2,p3,p4))')
    cursor.execute('CREATE UNIQUE INDEX corners2idx ON corners2 (queue)')

    # Edges 1-5
    cursor.execute('CREATE TABLE edges1 (queue INT, action INT, depth INT, p1 INT, p2 INT, p3 INT, p4 INT, PRIMARY KEY (p1,p2,p3,p4))')
    cursor.execute('CREATE UNIQUE INDEX edges1idx ON edges1 (queue)')

    # Edges 6-10
    cursor.execute('CREATE TABLE edges2 (queue INT, action INT, depth INT, p1 INT, p2 INT, p3 INT, p4 INT, PRIMARY KEY (p1,p2,p3,p4))')
    cursor.execute('CREATE UNIQUE INDEX edges2idx ON edges2 (queue)')

    # Edges 11-12
    cursor.execute('CREATE TABLE edges3 (queue INT, action INT, depth INT, p1 INT, p2 INT, p3 INT, p4 INT, PRIMARY KEY (p1,p2,p3,p4))')
    cursor.execute('CREATE UNIQUE INDEX edges3idx ON edges3 (queue)')

    connection.commit()
    return

def bfs(cubelet_type, table_number, table_name):
    # Queue counters
    counter_1 = 0 # for "popping" from queue
    counter_2 = 0 # for adding to queue

    # Initial state
    if cubelet_type == 'corner':
        if table_number == 1:
            initial_state = goal_state[:4]
        elif table_number == 2:
            initial_state = goal_state[4:8]
    elif cubelet_type == 'edge':
        if table_number == 1:
            initial_state = goal_state[8:12]
        elif table_number == 2:
            initial_state = goal_state[12:16]
        elif table_number == 3:
            initial_state = goal_state[16:]

    # Add goal state to queue
    to_queue = (counter_2, 0, 0, *state_to_int(initial_state))
    cursor.execute('INSERT INTO ' + table_name + ' VALUES (' + ','.join(['?'] * len(to_queue)) + ')', to_queue)

    # Breadth-first search
    while counter_2 - counter_1 + 1 > 0:
        # Print progress every 10,000 iterations
        if counter_1%10000 == 0:
            print(table_name + ': counter_1=' + str(counter_1) + ', counter_2=' + str(counter_2) + ', n_rows=' + str(cursor.execute('SELECT COUNT(1) FROM ' + table_name).fetchone()[0]))
        
        # Get the next element in the queue
        from_queue = cursor.execute('SELECT * FROM ' + table_name + ' WHERE queue == ' + str(counter_1)).fetchone()
        counter_1 += 1

        # Get the current state, depth (number of actions), and action face
        current_state = int_to_state(from_queue[3:])
        current_depth = from_queue[2]
        current_action_face = from_queue[1]

        # Find possible actions from the current state
        possible_actions = possible_action_mapping[current_action_face]
        for action in possible_actions:
            if cubelet_type == 'corner':
                state = state_to_int(rotate_corners(current_state, action[0], action[1]))
            elif cubelet_type == 'edge':
                state = state_to_int(rotate_edges(current_state, action[0], action[1]))

            # Add the state to the queue if it is not already in the database
            query = ''
            for i in range(len(state)):
                query += 'p' + str(i+1) + ' == ' + str(state[i])
                if i < len(state)-1:
                    query += ' AND '
            if cursor.execute('SELECT 1 FROM ' + table_name + ' WHERE ' + query).fetchone() is None:
                counter_2 += 1
                to_queue = (counter_2, action[0], current_depth+1, *state)
                cursor.execute('INSERT INTO ' + table_name + ' VALUES (' + ','.join(['?'] * len(to_queue)) + ')', to_queue)
    
    connection.commit()
    return

def drop_unnecessary_columns():
    cursor.execute('DROP INDEX corners1idx')
    cursor.execute('DROP INDEX corners2idx')
    cursor.execute('DROP INDEX edges1idx')
    cursor.execute('DROP INDEX edges2idx')
    cursor.execute('DROP INDEX edges3idx')

    cursor.execute('CREATE TABLE corners1copy AS SELECT depth,p1,p2,p3,p4 FROM corners1')
    cursor.execute('DROP TABLE corners1')
    cursor.execute('ALTER TABLE corners1copy RENAME TO corners1')

    cursor.execute('CREATE TABLE corners2copy AS SELECT depth,p1,p2,p3,p4 FROM corners2')
    cursor.execute('DROP TABLE corners2')
    cursor.execute('ALTER TABLE corners2copy RENAME TO corners2')

    cursor.execute('CREATE TABLE edges1copy AS SELECT depth,p1,p2,p3,p4 FROM edges1')
    cursor.execute('DROP TABLE edges1')
    cursor.execute('ALTER TABLE edges1copy RENAME TO edges1')

    cursor.execute('CREATE TABLE edges2copy AS SELECT depth,p1,p2,p3,p4 FROM edges2')
    cursor.execute('DROP TABLE edges2')
    cursor.execute('ALTER TABLE edges2copy RENAME TO edges2')

    cursor.execute('CREATE TABLE edges3copy AS SELECT depth,p1,p2,p3,p4 FROM edges3')
    cursor.execute('DROP TABLE edges3')
    cursor.execute('ALTER TABLE edges3copy RENAME TO edges3')
    
    connection.commit()

    return

def fill_pattern_database():
    make_tables()

    bfs('corner', 1, 'corners1')
    print('Number of rows in corners1:', cursor.execute('SELECT COUNT(1) FROM corners1').fetchone()[0])

    bfs('corner', 2, 'corners2')
    print('Number of rows in corners2:', cursor.execute('SELECT COUNT(1) FROM corners2').fetchone()[0])

    bfs('edge', 1, 'edges1')
    print('Number of rows in edges1:', cursor.execute('SELECT COUNT(1) FROM edges1').fetchone()[0])

    bfs('edge', 2, 'edges2')
    print('Number of rows in edges2:', cursor.execute('SELECT COUNT(1) FROM edges2').fetchone()[0])

    bfs('edge', 3, 'edges3')
    print('Number of rows in edges3:', cursor.execute('SELECT COUNT(1) FROM edges3').fetchone()[0])

    drop_unnecessary_columns()

    return

fill_pattern_database()

# --------------------------------- #
# --- Close database connection --- #
# --------------------------------- #

connection.close()