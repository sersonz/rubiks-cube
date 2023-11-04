import pycuber
import numpy as np
from utils import *

class CubeModel:
	def __init__(self):
		self.cube = pycuber.Cube()
		self.solved_cube = self.cube.copy()
	
	def getState(self):
		def compare(x):
			elems = list(x.facings.keys())
			elems.sort()
			return elems
		state = np.zeros((20, 24))
		i = 0
		corner_cubes = list(self.cube.select_type("corner"))
		corner_cubes.sort(key=compare)
		edge_cubes = list(self.cube.select_type("edge"))
		edge_cubes.sort(key=compare)
		for corner in corner_cubes:
			cubes = list(corner.facings.keys())
			cubes.sort()
			state[i][3*CORNERS.index(set(map(lambda x: x.colour, corner.facings.values()))) + CORNER_COLORS[corner.facings[cubes[0]].colour]] = 1
			i += 1
		for edge in edge_cubes:
			cubes = list(edge.facings.keys())
			cubes.sort()
			state[i][2*EDGES.index(set(map(lambda key: edge.facings[key].colour, cubes))) + EDGE_COLORS[edge.facings[cubes[0]].colour]] = 1
			i += 1
		return state