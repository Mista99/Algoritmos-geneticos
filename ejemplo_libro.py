import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Función objetivo (ejemplo: minimizar la función de Rastrigin)
def funcion_objetivo(individual):
    # Asegúrate de que devuelve una tupla con un solo valor
    return (sum(x**2 - 10*np.cos(2*np.pi*x) for x in individual),)  

# Configuración del algoritmo genético
toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Generación de individuos
toolbox.register("attr_float", random.uniform, -5.12, 5.12)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                   toolbox.attr_float, 30)  # 30 genes por individuo
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 100)  # Población inicial de 100 individuos

# Operadores genéticos
toolbox.register("evaluate", funcion_objetivo)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parámetros del algoritmo
CXPB, MUTPB, NGEN = 0.5, 0.2, 20  # Probabilidad de cruce, mutación y número de generaciones

# Ejecución del algoritmo genético
pop = toolbox.population()
hof = tools.HallOfFame(1)

# Cambié la forma en que se manejan los valores de fitness en las estadísticas
stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness else float('inf'))  # Para evitar errores

stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + stats.fields

for gen in range(NGEN):
    # Generación de la siguiente generación
    offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

    # Evaluación de los individuos
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit  # Asigna el valor de fitness

    # Comprobar que todos los individuos tienen valores de fitness antes de la selección
    for ind in offspring + pop:
        if not ind.fitness.valid:  # Si no es válido, se recalcula
            ind.fitness.values = toolbox.evaluate(ind)

    # Muestra solo el mejor individuo de la generación
    best_individual = tools.selBest(pop, 1)[0]
    print(f"\nMejor individuo de la generación {gen}: \n{best_individual}")


    # Selección para la próxima generación
    pop[:] = toolbox.select(offspring + pop, len(pop))  # Selección correcta
    hof.update(pop)
    
    # Registro de estadísticas
    record = stats.compile(pop)
    logbook.record(gen=gen, nevals=len(offspring), **record)

# Función para visualizar la evolución de la población
def plot_evolucion(logbook):
    gen = logbook.select('gen')
    avg = logbook.select('avg')
    min_ = logbook.select('min')
    max_ = logbook.select('max')

    plt.figure(figsize=(12, 6))
    plt.plot(gen, avg, label='Promedio', color='blue')
    plt.plot(gen, min_, label='Mínimo', color='red')
    plt.plot(gen, max_, label='Máximo', color='green')
    plt.xlabel('Generación')
    plt.ylabel('Valor de Fitness')
    plt.title('Evolución del Fitness a lo largo de las generaciones')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

# Visualización de los resultados
plot_evolucion(logbook)
