import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def fitness(x, y):
    return x**2 + y**2 - 2*x - 2*y + 1

def create_individual():
    x = random.uniform(-10, 10)  # Limitar x entre -10 y 10
    y = random.uniform(-10, 10)  # Limitar y entre -10 y 10
    return (x, y)

# Evaluar la población
def evaluate_population(population):
    fitness_scores = [fitness(x, y) for x, y in population]
    print("\nEvaluando población:")
    for individual, score in zip(population, fitness_scores):
        print(f"Individuo: {individual}, Aptitud: {score}")
    return fitness_scores

# Seleccionar los 2 mejores individuos
def select_best(population, fitness_scores, n=3):
    sorted_population = sorted(zip(population, fitness_scores), key=lambda pair: pair[1], reverse=False)
    best_individuals = [pair[0] for pair in sorted_population[:n]]
    
    print("\nSeleccionando los mejores individuos:")
    for individual in best_individuals:
        print(f"Individuo: {individual}, Aptitud: {fitness(*individual)}")
    
    return best_individuals

def mutate(individual, mutation_rate=0.1):
    x, y = individual
    if random.random() < mutation_rate:  # Aplicar mutación según la tasa de mutación
        mutation_value_x = random.uniform(-1, 1)  # Cambiar x en un rango de -1 a 1
        mutation_value_y = random.uniform(-1, 1)  # Cambiar y en un rango de -1 a 1
        x += mutation_value_x
        y += mutation_value_y
        # Limitar x y y entre -10 y 10
        x = max(-10, min(10, x))
        y = max(-10, min(10, y))
    return (x, y)

def crossover(parent1, parent2):
    # Combina los genes de dos padres para crear un nuevo individuo
    child_x = (parent1[0] + parent2[0]) / 2  # Promedio de x
    child_y = (parent1[1] + parent2[1]) / 2  # Promedio de y
    return (child_x, child_y)

def replace_population(best_individuals, population_size):
    # Reemplazar la población con los mejores individuos
    new_population = best_individuals.copy()
    while len(new_population) < population_size:
        new_population.append(create_individual())  # Completar con nuevos individuos aleatorios
    return new_population

#funcion principal del algoritmo
def genetic_algorithm(generations, population_size):

    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        print(f"\n--- Generación {generation + 1} ---")

        fitness_scores = evaluate_population(population)
        best_individuals = select_best(population, fitness_scores)

        # Crear nuevos individuos a través de cruzamiento
        new_population = []
        while len(new_population) < population_size:
            # Elegir aleatoriamente dos padres de los mejores individuos
            parent1, parent2 = random.sample(best_individuals, 2)
            child = crossover(parent1, parent2)  # Cruzar los padres para crear un hijo
            new_population.append(mutate(child))  # Mutar el hijo

        population = replace_population(best_individuals, population_size)

    best_solution = best_individuals[0]
    print("\nMejor solución encontrada:")
    print(f"Individuo: {best_solution}, Aptitud: {fitness(*best_solution)}")

# Parámetros
generations = 5
population_size = 10

genetic_algorithm(generations, population_size)

#--------------plotear la funcion objetivo--------------#
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)

Z = fitness(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#plotea la superficie
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title('Gráfica de la función objetivo en 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(np.min(Z), np.max(Z))

plt.show()
