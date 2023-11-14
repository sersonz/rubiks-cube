import numpy as np
import pycuber as pc

CORNERS = [
    {"white", "orange", "blue"},
    {"white", "orange", "green"},
    {"white", "red", "green"},
    {"white", "red", "blue"},
    {"yellow", "orange", "blue"},
    {"yellow", "orange", "green"},
    {"yellow", "red", "green"},
    {"yellow", "red", "blue"},
]

EDGES = [
    {"white", "blue"},
    {"white", "orange"},
    {"white", "green"},
    {"white", "red"},
    {"yellow", "blue"},
    {"yellow", "orange"},
    {"yellow", "green"},
    {"yellow", "red"},
    {"orange", "blue"},
    {"blue", "red"},
    {"green", "red"},
    {"orange", "green"},
]
CORNER_COLORS = {
    "white": 0,
    "yellow": 0,
    "blue": 1,
    "green": 1,
    "orange": 2,
    "red": 2,
}

EDGE_COLORS = {
    "white": 0,
    "yellow": 0,
    "blue": 0,
    "green": 1,
    "orange": 1,
    "red": 1,
}

ACTIONS = ["F", "B", "L", "R", "U", "D", "F'", "B'", "L'", "R'", "U'", "D'"]


def get_state(cube):
    def compare(x):
        elems = list(x.facings.keys())
        elems.sort()
        return elems

    state = np.zeros((20, 24))
    i = 0
    corner_cubes = list(cube.select_type("corner"))
    corner_cubes.sort(key=compare)
    edge_cubes = list(cube.select_type("edge"))
    edge_cubes.sort(key=compare)
    for corner in corner_cubes:
        cubes = list(corner.facings.keys())
        cubes.sort()
        state[i][
            3 * CORNERS.index(set(map(lambda x: x.colour, corner.facings.values())))
            + CORNER_COLORS[corner.facings[cubes[0]].colour]] = 1
        i += 1
    for edge in edge_cubes:
        cubes = list(edge.facings.keys())
        cubes.sort()
        state[i][2 *
                 EDGES.index(set(map(lambda key: edge.facings[key].colour, cubes))) +
                 EDGE_COLORS[edge.facings[cubes[0]].colour]] = 1
        i += 1
    return state


solved_cube = pc.Cube()


def is_solved(cube):
    return cube == solved_cube
