#!/usr/bin/python3

import numpy as np
import json
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        
        # Инициализация весов и смещений
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            # Используем sigmoid для всех слоев, включая последний
            a = self.sigmoid(z)  
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, Y):
        self.a = [X] # хз что это, на угад поставил
        m = Y.shape[0]
        self.d_weights = [0] * len(self.weights)
        self.d_biases = [0] * len(self.biases)

        # Output layer error
        da = self.a[-1] - Y
        for i in reversed(range(len(self.weights))):
            dz = da * self.sigmoid_derivative(self.a[i + 1]) if i < len(self.weights) - 1 else da
            self.d_weights[i] = np.dot(self.a[i].T, dz) / m
            self.d_biases[i] = np.sum(dz, axis=0, keepdims=True) / m
            da = np.dot(dz, self.weights[i].T)

        # Обновление весов и смещений
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def get_weights(self):
        return self.weights

    def load_weights(self, weights):
        self.weights = weights


    def randomize_weight_or_bias(self):
        layer_index = random.randint(0, len(self.weights) + len(self.biases) - 1)

        if layer_index < len(self.weights):
            # Изменяем вес
            weight_index = random.randint(0, self.weights[layer_index].size - 1)
            weight_shape = self.weights[layer_index].shape
            flat_weights = self.weights[layer_index].flatten()
            flat_weights[weight_index] = random.uniform(0, 1)  # Случайное значение от 0 до 1
            self.weights[layer_index] = flat_weights.reshape(weight_shape)
        else:
            # Изменяем смещение
            bias_index = layer_index - len(self.weights)  # Определяем индекс смещения
            bias_element_index = random.randint(0, self.biases[bias_index].shape[1] - 1)  # Выбираем элемент смещения
            self.biases[bias_index][0, bias_element_index] = random.uniform(-10.0, 10.0)  # Случайное значение от -10.0 до 10.0

def save_weights(nn, filename):
    """Сохранить веса и смещения нейросети в файл."""
    weights_and_biases = {
        'weights': [w.tolist() for w in nn.weights],
        'biases': [b.tolist() for b in nn.biases]
    }
    with open(filename, 'w') as f:
        json.dump(weights_and_biases, f)

def load_weights(nn, filename):
    """Загрузить веса и смещения нейросети из файла."""
    try:
        with open(filename, 'r') as f:
            weights_and_biases = json.load(f)
        nn.weights = [np.array(w) for w in weights_and_biases['weights']]
        nn.biases = [np.array(b) for b in weights_and_biases['biases']]
        return True
    except FileNotFoundError:
        print(f"Файл '{filename}' не найден.")
        return False

