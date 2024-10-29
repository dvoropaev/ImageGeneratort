#!/usr/bin/python3


import numpy as np
from SN_gen import generator
import random
from PIL import Image



def image_to_array(image_path):
	"""Преобразует изображение в одномерный массив, нормализуя значения от 0 до 1."""
	img = Image.open(image_path)
	img = img.resize((100, 100))  # Убедитесь, что изображение IMAGE_SIZExIMAGE_SIZE
	img_array = np.array(img) / 255.0  # Нормализация
	return img_array.flatten()  # Преобразуем в одномерный массив

def array_to_image(array, output_path):
	"""Сохраняет изображение из одномерного массива."""
	img_array = np.array(array).reshape((10, 10, 3))  # Восстанавливаем форму IMAGE_SIZExIMAGE_SIZE, 3 канала (RGB)
	img = Image.fromarray((img_array * 255).astype(np.uint8))  # Обратное преобразование и создание изображения
	img.save(output_path)

def rand_name(prefix="_"):
	file_name = prefix + ''.join(random.choices('0123456789', k=5)) + '.png'
	return file_name

a = np.array([1]*300)
b = generator.forward(a)[-1]
print(b)

for i in range(1000):
	a = np.array([1]*300)
	generator.forward(a)
	generator.backward(a, a)
	a = np.array([0]*300)
	generator.forward(a)
	generator.backward(a, a)

a = np.array([1]*300)
b = generator.forward(a)[-1]
print(b)

#for i in range(10):
#	a = np.random.uniform(low=0.0, high=1.0, size=300)
#	b = generator.forward(a)[-1]
#	array_to_image(b, rand_name(prefix="./"))

a = np.array([1]*300)
b = generator.forward(a)[-1]
array_to_image(b, rand_name(prefix="./AAA"))

a = np.array([0]*300)
b = generator.forward(a)[-1]
array_to_image(b, rand_name(prefix="./BBB"))


#a = np.array([0]*300)
#b = generator.forward(a)[-1]
#array_to_image(b, rand_name(prefix="./BBB"))

