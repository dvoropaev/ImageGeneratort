#!/usr/bin/python3

import numpy as np
import json
import random

class NeuralNetwork:

	def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.5):
		self.learning_rate = learning_rate
		self.layers = [input_size] + hidden_layers + [output_size] # layers[i] - колличество нейронов на i-том слое
		self.weights = []  # self.weights[i][x][y]  i - слой, x - нейрон первого слоя, y - нейрон второго слоя
		self.biases = []

		for i in range(len(self.layers) - 1):
			weight = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01  # weight[x][y] - вес связи между x-нейроном первого слоя и y-нейроном второго слоя
			bias = np.zeros((1, self.layers[i + 1]))
			self.weights.append(weight)
			self.biases.append(bias)
		#print(self.biases) #!!!!!!!!!!!!!

	def forward(self, X):
		self.a = [X]
		for i in range(len(self.weights)):
			z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
			# Используем sigmoid для всех слоев, включая последний
			a = self.sigmoid(z)  
			self.a.append(a)
		return [a.flatten().tolist() if isinstance(a, np.ndarray) else a for a in self.a]
#		return self.a[-1]

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def sigmoid_derivative(self, a):
		return a * (1 - a)




