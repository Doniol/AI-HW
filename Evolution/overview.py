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


    def get_deck_sum(self)-> List[int]:
        ''' Function to get the deck_sum
        return: A list with integers of the cards in the deck
        '''


    def get_deck_multi(self) -> List[int]:
        ''' Function to get the deck_multi
        return: A list with integers of the cards in the deck
        '''


    def calc_fitness(self):
        ''' This calculates the fitness of the individual
        '''


    def get_fitness(self) -> int:
        ''' Function to get the fitness
        return: The fitness as a int
        '''


    def mutate(self):
        ''' This function causes a random mutation
        '''


    def mutate_remove_sum(self):
        ''' This mutation function removes a random item from the sum deck and adds it to the multi deck
        '''


    def mutate_remove_multi(self):
        ''' This mutation function removes a random item from the multi deck and adds it to the sum deck
        '''


    def mutate_swap(self):
        ''' This mutation function swaps a value from the sum deck and the multi deck
        '''

    
    def mutate_switch_deck(self):
        ''' This mutation function switches the sum and multi decks
        '''

class population:
    def __init__(self, population_size: int):
        ''' Creates a population
        
        population_size: The amount of individuals you want in the population
        '''


    def battle_royale(self, pool_size: int) -> List[Individual]:
        ''' Returns a list of candidates with which to create the next generation

        Takes the previous generation, sorts it and stores the best individual for use in the next generation.
        Afterwards the function puts each individual from the previous generation into a group with a few other individuals
         from the same generation, and stores the best one from the group for use in the next generation.

        pool_size: How big each group of individuals should be
        return: A list containing all of the selected individuals to be used for the next generation
        '''
      

    def child_gen_crossover(self, pool_size):
        ''' Function to generate the next generation
        This function will call the function to get the survivors of the generation and applies crossover on them for the next generation.
        The mutation happens when the child is created.
        
        pool_size:  The amount of individuals you want to contest in a pool for survival
        '''


    def run(self, runs=1000, pool_size=8)-> Individual:
        ''' This function runs the evolution algorithm
        This function creates as many generation as there are runs declared using the evolution algorithm

        runs: The amount of generations the algorithm is alowed to create
        pool_size: The amount of individuals you want to contest in a pool for survival
        return: The individual with the highest fitness of the last generation
        '''


def main():
    evolve_card = population(100)
    best = evolve_card.run()
    print()
    print(best.fitness)
    print("sum_deck: " + str(best.deck_sum))

main()