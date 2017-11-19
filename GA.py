import numpy as np
import random
import sys
from termcolor import colored

def printUsage():
	print("Usage: GA.py <population_size> <generations> <parents> "
	      "<mutation_rate> <restart_threshold> <board_size>")

class Board:
	size = 0
	optima = 0

	def __init__(self, verbose):
		self.chromosome = []
		self.evaluation = 0
		self.verbose = verbose
		self.survival_probability = 0.0

	def evaluate(self):
		clashes = 0
		row_cols = abs(
			len(self.chromosome) - (len(np.unique(self.chromosome))))
		clashes += row_cols
		for i in range(len(self.chromosome)):
			for j in range(len(self.chromosome)):
				if (i != j):
					dx = abs(i - j)
					dy = abs(self.chromosome[i] - self.chromosome[j])
					if (dx == dy):
						clashes += 1
		self.evaluation = clashes

	def generateChromosome(self):
		initial_solution = np.arange(self.size)
		np.random.shuffle(initial_solution)
		if self.verbose is True:
			print('Initial Solution: ' + ', '.join(self.chromosome))
		self.chromosome = initial_solution

	def __str__(self):
		result = 'Chromosome: ' + ', '.join(str(self.chromosome))
		result += ' Evaluation: ' + str(self.evaluation) + ' clashes'
		return result

	def __repr__(self):
		return self.__str__()

class GeneticAlgorithm:
	def __init__(self, population, generations, parents, mutation_rate,
	             restart_threshold, verbose):
		self.population_size = population
		self.generations = generations
		self.parents = parents
		self.mutation_rate = mutation_rate
		self.restart_threshold = restart_threshold
		self.population = []
		self.verbose = verbose

	def print_solution(self, solution):
		print(colored('#' * 100, 'red'))
		print(colored('\nPrinting best solution found\n', 'red'))
		print(colored(solution, 'red'))
		print('\n')

	def generate_initial_population(self):
		population = [Board(True) for i in range(self.population_size)]
		for i in range(self.population_size):
			population[i].generateChromosome()
			population[i].evaluate()
			if self.verbose is True:
				print("Population[{}]: ".format(i))
				print(population[i])
		return population

	# Simple Elitist selection
	def selection(self, population):
		sorted_pop = sorted(population, key=lambda x: x.evaluation,
		                    reverse=False)
		return sorted_pop[0:self.parents]

	def generate_offspring(self, selected):
		# offspring = [Board(True) for i in range(self.population_size)]
		offspring = selected
		while len(offspring) < len(self.population):
			first_parent_index, second_parent_index = 0, 0
			while first_parent_index == second_parent_index:
				first_parent_index = random.randint(0, len(selected) - 1)
				second_parent_index = random.randint(0, len(selected) - 1)
			child = self.crossover(selected[first_parent_index],
			                       selected[second_parent_index])
			child = self.mutate(child)
			offspring.append(child)
		return offspring

	def crossover(self, first_parent, second_parent):
		child = Board(True)
		n = len(first_parent.chromosome)
		c = np.random.randint(n, size=1)
		i = 0
		while i < len(first_parent.chromosome):
			if i < c:
				child.chromosome.append(first_parent.chromosome[i])
			else:
				child.chromosome.append(second_parent.chromosome[i])
			i += 1
		child.evaluate()
		return child

	def mutate(self, child):
		probability = np.random.uniform(0.0, 1.0)
		if probability > self.mutation_rate:
			chromosome = np.random.randint(8)
			child.chromosome[chromosome] = np.random.randint(8)
		return child

	def runGeneration(self, generation):
		if self.verbose is True:
			print("Running Generation #{}".format(generation))
		self.population = self.generate_offspring()

	def stopCriteria(self, population, generation):
		sorted_population = sorted(population, key=lambda x: x.evaluation,
		                           reverse=False)
		if sorted_population[0].evaluation == Board.optima:
			self.print_solution(sorted_population[0])
			return True
		if generation == self.generations:
			print(colored('Finished for generations criteria\n', 'red'))
			self.print_solution(sorted_population[0])
			return True
		return False

	def runGA(self):
		generation = 0
		self.population = self.generate_initial_population()
		while not self.stopCriteria(self.population, generation):
			if self.verbose is True:
				print("Running Generation #{}".format(generation))
			selected = self.selection(self.population)
			offspring = self.generate_offspring(selected)  # Crossover and
			# mutate done
			self.population = offspring  # Update population
			generation += 1

if __name__ == "__main__":
	if len(sys.argv) < 7:
		printUsage()
	else:
		args = sys.argv[1:]
		popsize = int(args[0])
		generations = int(args[1])
		selection = int(args[2])
		mutation = float(args[3])
		restart = float(args[4])
		board_size = int(args[5])
		Board.size = board_size
		ga = GeneticAlgorithm(popsize, generations, selection, mutation,
		                      restart, True)
		ga.runGA()
