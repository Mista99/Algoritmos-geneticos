import random
from deap import base, creator, tools

# Definimos la función objetivo
def fitness(individual):
    x, y = individual  # Descomponer el individuo en x e y
    return (x**2 + y**2 - 2*x - 2*y + 1,)  # Retornar como una tupla

# Crear las clases necesarias para DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizar la función, hereda de la funcion Fitness de la libreria, y configura para minimizar
creator.create("Individual", list, fitness=creator.FitnessMin) #coloca un atributo para "Individual" en el que debe estar el respectivo valor fitness, segun la funcion definida para minimizar

# Configuración de la caja de herramientas DEAP
#toolbox: Permite organizar y registrar las funciones relacionadas con la creación de individuos, evaluación de aptitud, selección, cruce y mutación.
toolbox = base.Toolbox()
toolbox.register("x", random.uniform, -10, 10)
toolbox.register("y", random.uniform, -10, 10)
print("probando el toolbox")
print(toolbox.x)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.x, toolbox.y), n=1) #initCycle: inicializa un individuo o un conjunto de individuos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #inicializar una colección (por ejemplo, una lista) mediante la repetición de una función generadora de individuos.
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5) #"mate" es la operación de cruce. cxBlend es un método de cruce utilizando una media ponderada, promedio equilibrado entre los dos padres.
                                                   #Valores m 0 favorecerán al primer padre, mientras que cercanos a 1 favorecerán al segundo padre.
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) # mutacion basada en una distribución normal (Gaussiana), con uan probabilidad de un 20% de sufrir una mutacion para cada gen
toolbox.register("select", tools.selTournament, tournsize=3) #para cada selección, se eligen 3 individuos aleatorios de la población, y el individuo con mejor fitness entre ellos es el que se seleccion

# Parámetros del algoritmo
generations = 5
population_size = 10

""" 
#parametros individual son objetos con parametros:
    .fitness.values

    """
"""population es una lista de individuos(instancia)"""
# Inicializar la población
def main():
    # Generar la población inicial
    population = toolbox.population(n=population_size)
    print("probando population: ")
    #-------------iteraciones o generaciones-------------#
    for generation in range(generations):
        print(f"\n--- Generación {generation + 1} ---")

        # generar la lista fitness a partir de la poblacion
        fitness_values = list(map(toolbox.evaluate, population)) #evalua cada individuo de la poblacion y lo mete a una lista
        
        for individual, fit in zip(population, fitness_values):
            individual.fitness.values = fit #le asigno al atributo fitness de cada individuo su valor correspondiente
            print(f"Individuo: {individual}, Aptitud: {fit[0]}")

        # Seleccionar los mejores individuos
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring)) #se clonan para no afectar la poblacion original

        # Cruzamiento y mutación
        for child1, child2 in zip(offspring[::2], offspring[1::2]): 
            #[::2]: selecciona a los individuos en las posiciones pares 
            #[1::2]: selecciona a los individuos en las posiciones impares
            #tira una moneda, si sale menor a 0.5, se cruzan
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.1:  # Probabilidad de mutación
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Reemplazar la población
        population[:] = offspring

    # Mejor solución encontrada
    best_individual = tools.selBest(population, k=1)[0]
    print("\nMejor solución encontrada:")
    print(f"Individuo: {best_individual}, Aptitud: {best_individual.fitness.values[0]}")

if __name__ == "__main__":
    main()
