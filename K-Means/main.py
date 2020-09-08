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
    # Extract data from .csv file
    data = np.genfromtxt(file, delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    dates = np.genfromtxt(file, delimiter=";", usecols=[0])
    # Assign labels based on the date said data was recorded
    labels = []
    for label in dates:
        if label < int(year + "0301"):
            labels.append("winter")
        elif int(year + "0301") <= label < int(year + "0601"):
            labels.append("lente")
        elif int(year + "0601") <= label < int(year + "0901"):
            labels.append("zomer")
        elif int(year + "0901") <= label < int(year + "1201"):
            labels.append("herfst")
        else: # from 01-12 to end of year
            labels.append("winter")
            
    return data, dates, labels


def find_min_max(original_dataset: np.ndarray) -> List[List[float]]:
    ''' This function determines the lowest and highest value of each of the different data types, this will be used to normalize the data

    orginal_dataset: The dataset which you want to normalize
    return: A list containing the lists for both the lowest values of the dataset and a list with highest values of the dataset for each data type 
    '''
    min= [float('inf')] * len(original_dataset[0])
    max= [float('-inf')] * len(original_dataset[0])
    # Calculate te min and max values for all the days for later normalization
    for data_type in range(len(original_dataset[0])):
        for day in original_dataset:
            if day[data_type] < min[data_type]:
                min[data_type] = day[data_type]
            if day[data_type] > max[data_type]:
                max[data_type] = day[data_type]
    return min, max
    

def normalize(dataset_to_be_changed: np.ndarray, min_values: List[int], max_values: List[int]):
    ''' This function normalizes the given dataset
    
    dataset_to_be_changed: The data set which you want to normalize
    min_values: A list containing the lowest values for the different data types
    max_values: A list containing the highest values for the different data types
    '''
    #normalize the data
    for data_type in range(0, len(dataset_to_be_changed[0])):
        for day in dataset_to_be_changed:
            day[data_type] = (day[data_type] - min_values[data_type]) / (max_values[data_type] - min_values[data_type])


def calculate_distance(given_day: np.ndarray, data: np.ndarray, labels: List[str]):
    ''' This function calculates the distance from the to test day to it's neighbours

    given_day: The day, and its' data, for which a season has to be calculated
    data: A list of all available data about other days that are already in the system
    labels: A list of all labels corresponding to the data entries at the same index
    return: A list with tuples which contain the distances to the neigbours with the corresponding season
    '''
    distance_to_neighbour = []
    # Loop through all days in dataset
    for data_index in range(0, len(data)):
        distance_neighbour = 0
        # Loop through data type from each day and calculate the distance between selected day and given_day
        for data_type_index in range(0, len(given_day)):
            if given_day[data_type_index] > data[data_index][data_type_index]:
                distance_neighbour += given_day[data_type_index] - data[data_index][data_type_index]
            else:
                distance_neighbour += data[data_index][data_type_index] - given_day[data_type_index]
        distance_to_neighbour.append((distance_neighbour, labels[data_index]))

    # Sort the neighbours by distance then use the top k neigbours for season calculation
    distance_to_neighbour.sort()
    return distance_to_neighbour
    

def pinpoint_season(distance_to_neighbour: List[Tuple[int, str]], k: int = 1) -> str:
    ''' Calculates the most logical season for the given day

    k: The amount of nearest data-points that are to be used to determine the correct season for the given_day
    distance_to_neigbour: A list with tuples which contain the distances to the neigbours with the corresponding season
    return: The calculated season for the given_day
    '''
    selected_neighbours = distance_to_neighbour[0:k]
    season_dict = {
        "winter": 0,
        "herfst": 0,
        "zomer": 0,
        "lente": 0
    }
    for neighbour in selected_neighbours:
        season_dict[neighbour[1]] += 1
    
    chosen_season = max(season_dict, key=season_dict.get)
    # Check whether there are ties between the season counts in season_dict
    for key in season_dict:
        if key != chosen_season and season_dict[chosen_season] == season_dict[key]:
            return pinpoint_season(selected_neighbours, k - 1)
    return chosen_season


def success_rate_calculation(results: List[str], answers: List[str]) -> float:
    ''' Calculates and returns the % of correct results
    
    results: List of results from the pinpoin_season calculations
    answers: List of correct answers from said calculations, these need to be sorted
     in the same order as the results
    return: The % of results that are the same as their corresponding desired answers
    '''
    correct_results = 0
    for result_index in range(0, len(results)):
        if results[result_index] == answers[result_index]:
            correct_results += 1
        # print(results[result_index], answers[result_index])
    
    return (correct_results / len(results)) * 100


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
    # Make sure that the k-value doesn't exceed the amount of available data-points
    if k_max > len(original_data):
        k_max = original_data
    if k_min < 1:
        k_min = 1

    # Calculate the success rate for each k-value and return the best one
    success_rate = 0
    optimal_k_value = 0
    for k_value in range(k_min, k_max):
        print(k_value)
        predicted_seasons = []
        for day in test_days:
            calculated_distances = calculate_distance(day, original_data, original_labels)
            predicted_seasons.append(pinpoint_season(calculated_distances, k_value))
        calculated_success_rate = success_rate_calculation(predicted_seasons, test_labels)
        if calculated_success_rate > success_rate:
            success_rate = calculated_success_rate
            optimal_k_value = k_value
    
    return optimal_k_value, success_rate


def main() -> None:
    # Load the data from dataset1.csv
    original_data, original_dates, original_labels = load_data()
    
    # Either load data from days.csv for testing or validation1.csv for validating the program
    # test_days = load_data("days.csv")[0]
    test_days, test_dates, test_labels = load_data("validation1.csv", "2001")

    # Normalize all data
    min, max = find_min_max(original_data)
    normalize(original_data, min, max)
    normalize(test_days, min, max)

    # Predict seasons of given days
    # predicted_seasons = []
    # for day in test_days:
    #     calculated_distance = calculate_distance(day, original_data, original_labels)
    #     predicted_seasons.append(pinpoint_season(calculated_distance, 1))

    # Print either the success rate of the validated days
    # print(success_rate_calculation(predicted_seasons, test_labels))
    # Or the results from the test days
    # print(predicted_seasons)
    # Or calculate optimal k-value
    print(calculate_optimal_k(test_days, test_labels, original_data, original_labels, 1, 100))

if __name__ == "__main__":
    main()