import numpy as np
from typing import List, Tuple
import math
import random

random.seed(0)


def load_data(file: str= "dataset1.csv", year: str="2000")-> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    ''' This function loads the csv files and turns into an ndarray which we use to calculate the neigbour distance
    
    file: String with the file name
    year: The year of the data
    return: A ndarray with the data, a ndarray with the dates and a regular list with the seasonal labels for said days
    '''

def find_min_max(original_dataset: np.ndarray) -> List[List[float]]:
    ''' This function determines the lowest and highest value of each of the different data types, this will be used to normalize the data

    orginal_dataset: The dataset which you want to normalize
    return: A list containing the lists for both the lowest values of the dataset and a list with highest values of the dataset for each data type 
    '''

def normalize(dataset_to_be_changed: np.ndarray, min_values: List[int], max_values: List[int]):
    ''' This function normalizes the given dataset
    
    dataset_to_be_changed: The data set which you want to normalize
    min_values: A list containing the lowest values for the different data types
    max_values: A list containing the highest values for the different data types
    '''

def calculate_distance(given_day: np.ndarray, data: np.ndarray, labels: List[str]):
    ''' This function calculates the distance from the to test day to it's neighbours

    given_day: The day, and its' data, for which a season has to be calculated
    data: A list of all available data about other days that are already in the system
    labels: A list of all labels corresponding to the data entries at the same index
    return: A list with tuples which contain the distances to the neigbours with the corresponding season
    '''
    
def pinpoint_season(distance_to_neighbour: List[Tuple[int, str]], k: int = 1) -> str:
    ''' Calculates the most logical season for the given day

    k: The amount of nearest data-points that are to be used to determine the correct season for the given_day
    distance_to_neigbour: A list with tuples which contain the distances to the neigbours with the corresponding season
    return: The calculated season for the given_day
    '''

def success_rate_calculation(results: List[str], answers: List[str]) -> float:
    ''' Calculates and returns the % of correct results
    
    results: List of results from the pinpoin_season calculations
    answers: List of correct answers from said calculations, these need to be sorted
     in the same order as the results
    return: The % of results that are the same as their corresponding desired answers
    '''
    
def calculate_optimal_k(test_days: List[np.ndarray], test_labels: List[str], original_data: List[np.ndarray], original_labels: List[np.ndarray], k_min: int, k_max: int) -> Tuple[int, float]:
    ''' Calculates the optimal k-value
    Calculates the success rate of the calculations with all k-values between k_min and k_max and returns the best tested one together with its'
     success rate.

    test_data: Days that will be used to calculate the best k-value
    test_labels: The correct answers for the test-data
    original_data: The dataset that will be used for the coming calculations
    original_labels: The corresponding labels for the dataset days
    k_min: The minimum amount of neigbours(k) you want to test
    k_max: The maximum amount of neighbours(k) you want to test
    return: The best tested k-value and the associated success rate
    '''

def main() -> None:
    # Load the data from dataset1.csv
    original_data, original_dates, original_labels = load_data()
    
    # Either load data from days.csv for testing or validation1.csv for validating the program
    test_days = load_data("days.csv")[0]
    # test_days, test_dates, test_labels = load_data("validation1.csv", "2001")

    # Normalize all data
    min, max = find_min_max(original_data)
    normalize(original_data, min, max)
    normalize(test_days, min, max)

    # Predict seasons of given days
    predicted_seasons = []
    for day in test_days:
        calculated_distance = calculate_distance(day, original_data, original_labels)
        predicted_seasons.append(pinpoint_season(calculated_distance, 1))

    # Print either the success rate of the validated days
    # print(success_rate_calculation(predicted_seasons, test_labels))
    # Or the results from the test days
    print(predicted_seasons)
    # Or calculate optimal k-value
    # print(calculate_optimal_k(test_days, test_labels, original_data, original_labels, 1, 100))

if __name__ == "__main__":
    main()