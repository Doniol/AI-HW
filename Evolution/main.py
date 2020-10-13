import random
import copy
import operator
from typing import List

class Individual:
    def __init__(self, name: str, input_deck_sum: List[int] = None, input_deck_multi: List[int] = None):
        ''' Creates a individual

        name: A string with the name of the individual
        input_deck_sum: A list with integers of the cards in the deck
        input_deck_multi: A list with intergers of the cards in the deck
        '''
        self.name = name
        # create random decks if there are no decks given
        if( (input_deck_sum == None) and (input_deck_multi == None) ): 
            temp_card_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            self.deck_sum = random.sample(temp_card_list, random.randint(0,10))
            temp_card_list = [card for card in temp_card_list if not card in self.deck_sum]
            self.deck_multi = temp_card_list
        else:
            self.deck_sum = input_deck_sum
            self.deck_multi = input_deck_multi
        # random chance of mutation
        if random.randint(0, 100) == 100:
            self.mutate()
        self.calc_fitness()

    def get_deck_sum(self)-> List[int]:
        ''' Function to get the deck_sum
        return: A list with integers of the cards in the deck
        '''
        return self.deck_sum

    def get_deck_multi(self) -> List[int]:
        ''' Function to get the deck_multi
        return: A list with integers of the cards in the deck
        '''
        return self.deck_multi

    def calc_fitness(self):
        ''' This calculates the fitness of the individual
        '''
        difference_sum = abs(sum(self.deck_sum) - 36)
        difference_multi = 1
        for card in self.deck_multi:
            difference_multi *= card
        difference_multi = abs(difference_multi - 360)
        self.fitness = difference_multi + difference_sum

    def get_fitness(self) -> int:
        ''' Function to get the fitness
        return: The fitness as a int
        '''
        return self.fitness

    def mutate(self):
        ''' This function causes a random mutation
        '''
        mutation = random.randint(0,3)
        if mutation == 0:
            self.mutate_remove_sum()
        elif mutation == 1:
            self.mutate_remove_multi()
        elif mutation == 2:
            self.mutate_swap()
        else:
            self.mutate_switch_deck()

    def mutate_remove_sum(self):
        ''' This mutation function removes a random item from the sum deck and adds it to the multi deck
        '''
        if self.deck_sum:
            mutation = random.choice(self.deck_sum)
            self.deck_sum.remove(mutation)
            self.deck_multi.append(mutation)

    def mutate_remove_multi(self):
        ''' This mutation function removes a random item from the multi deck and adds it to the sum deck
        '''
        if self.deck_multi:
            mutation = random.choice(self.deck_multi)
            self.deck_multi.remove(mutation)
            self.deck_sum.append(mutation)

    def mutate_swap(self):
        ''' This mutation function swaps a value from the sum deck and the multi deck
        '''
        if self.deck_sum and self.deck_multi:
            mutation_sum = random.choice(self.deck_sum)
            mutation_multi = random.choice(self.deck_multi)
            self.deck_sum.remove(mutation_sum)
            self.deck_multi.append(mutation_sum)
            self.deck_multi.remove(mutation_multi)
            self.deck_sum.append(mutation_multi)
    
    def mutate_switch_deck(self):
        ''' This mutation function switches the sum and multi decks
        '''
        temp = self.deck_sum
        self.deck_sum = self.deck_multi
        self.deck_multi = temp

class population:
    def __init__(self, population_size: int):
        ''' Creates a population
        
        population_size: The amount of individuals you want in the population
        '''
        self.individuals = []
        self.generation = 0
        self.size = population_size
        for individual_counter in range(population_size):
            new_individual = Individual("G0_ID" + str(individual_counter))
            self.individuals.append(new_individual)

    def battle_royale(self, pool_size: int) -> List[Individual]:
        ''' Returns a list of candidates with which to create the next generation

        Takes the previous generation, sorts it and stores the best individual for use in the next generation.
        Afterwards the function puts each individual from the previous generation into a group with a few other individuals
         from the same generation, and stores the best one from the group for use in the next generation.

        pool_size: How big each group of individuals should be
        return: A list containing all of the selected individuals to be used for the next generation
        '''
        prev_gen = copy.deepcopy(self.individuals)
        next_gen = [sorted(prev_gen, key=operator.attrgetter('fitness'))[0]]
        prev_gen.remove(next_gen[0])

        while len(prev_gen) > pool_size:
            contestants = []
            for i in range(pool_size):
                contestants.append(random.choice(prev_gen))
                prev_gen.remove(contestants[i])
            winner = contestants[0]
            for contestant in contestants:
                if contestant.get_fitness() < winner.get_fitness():
                    winner = contestant
            next_gen.append(contestant)
        
        winner = prev_gen[0]
        for contestant in prev_gen:
            if contestant.get_fitness() < winner.fitness:
                winner = contestant
        next_gen.append(contestant)
        return next_gen        

    def child_gen_crossover(self, pool_size):
        ''' Function to generate the next generation
        This function will call the function to get the survivors of the generation and applies crossover on them for the next generation.
        The mutation happens when the child is created.
        
        pool_size:  The amount of individuals you want to contest in a pool for survival
        '''
        possible_parents = self.battle_royale(pool_size)
        new_gen = [possible_parents[0]]
        self.generation += 1
        # while the next gen isn't the size of the requested population
        while len(new_gen) < self.size:
            card_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            child_deck_sum = []
            child_deck_multi = []
            
            # get random parents
            parent_1 = random.choice(possible_parents)
            parent_2 = random.choice(possible_parents)
            while parent_1 == parent_2:
                parent_2 = random.choice(possible_parents)
            
            # get a deck of the parents
            parent_1_deck_sum = parent_1.get_deck_sum()
            parent_2_deck_sum = parent_2.get_deck_sum()

            # crossover
            # create the sum deck
            if parent_1_deck_sum:
                if parent_2_deck_sum:
                    child_deck_sum = parent_1_deck_sum[:int(len(parent_1_deck_sum)/2)]
                    for card in parent_2_deck_sum[int(len(parent_2_deck_sum)/2):]:
                        if card not in child_deck_sum:
                            child_deck_sum.append(card)
                else:
                    child_deck_sum = parent_1_deck_sum[:int(len(parent_1_deck_sum)/2)]
            else:
                child_deck_sum = parent_2_deck_sum[int(len(parent_2_deck_sum)/2):]

            #create the multi deck
            child_deck_multi = [card for card in card_list if not card in child_deck_sum]

            new_individual = Individual("G" + str(self.generation) + "Id" + str(len(new_gen)-1), child_deck_sum, child_deck_multi)
            new_gen.append(new_individual)
        self.individuals = new_gen

    def run(self, runs=100, pool_size=8)-> Individual:
        ''' This function runs the evolution algorithm
        This function creates as many generation as there are runs declared using the evolution algorithm

        runs: The amount of generations the algorithm is alowed to create
        pool_size: The amount of individuals you want to contest in a pool for survival
        return: The individual with the highest fitness of the last generation
        '''
        for run in range(runs):
            self.child_gen_crossover(pool_size)

        best_individual = self.individuals[0]
        best_fitness = self.individuals[0].get_fitness()
        for individual in self.individuals:
            if individual.get_fitness() < best_fitness:
                best_individual = individual
                best_fitness = individual.get_fitness()
        return best_individual


def main():
    for i in range(100, 2000, 100):
        avg = 0
        for j in range(0, 100):
            evolve_card = population(250)
            best = evolve_card.run(runs=i)
            avg += best.fitness
        print(i, avg/100)
    # print()
    # print(best.fitness)
    # print("sum_deck: " + str(best.deck_sum))

main()