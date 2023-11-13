import pycuber
import numpy as np
from utils import *
import tensorflow as tf
from tensorflow import keras
import random

class CubeModel:
	def __init__(self):
		self.cube = pycuber.Cube()
		self.solved_cube = self.cube.copy()
	    # Make model
		model_input = keras.Input(shape=(20*24,))
		first_layer = tf.keras.layers.Dense(4096, activation="relu")(model_input)
		second_layer = tf.keras.layers.Dense(1024, activation="relu")(first_layer)
		value_layer = tf.keras.layers.Dense(480, activation="relu")(second_layer)
		policy_layer = tf.keras.layers.Dense(480, activation="relu")(second_layer)
		value_output = tf.keras.layers.Dense(1, name="value")(value_layer)
		policy_output = tf.keras.layers.Dense(12, name="policy")(policy_layer)
		self.model = tf.keras.Model(inputs=model_input, outputs=[value_output, policy_output])
		
	def adi_step(self, k=100, l=100):
		'''
		Performs an autodidactic iteration step.
		'''
		samples = []
		for i in range(l):
			formula = []
			for j in range(k):
				formula.append(random.choice(ACTIONS))
			samples.append((pycuber.cube(" ".join(formula)), k))
	
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
	