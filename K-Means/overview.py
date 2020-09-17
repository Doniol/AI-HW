import numpy as np
from typing import List, Tuple
import math
import random
import copy
import time

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


def success_rate_calculation(results: List[str], answers: List[str]) -> float:
    ''' Calculates and returns the % of correct results
    
    results: List of results from the pinpoint_season calculations
    answers: List of correct answers from said calculations, these need to be sorted
     in the same order as the results
    return: The % of results that are the same as their corresponding desired answers
    '''


def create_starting_centroids(num_data_types: int, k=4) -> List[List[float]]:
    ''' Calculate starting centroids

    num_data_types: The amount of data_types the centroid has
    k: The number centroids you want to have
    return: A list containing all created centroids
    '''


def calculate_centroid_location(centroids: List[List[float]], clusters: List[List[int]], dataset: List[np.ndarray]) -> List[List[float]]:
    ''' Calculate the new mean/location of the centroid

    centroids: The already existing centroids
    clusters: The clusters corresponding to the given centroids
    dataset: A dataset containing the trainingdata for the algorithm
    return: The updated centroid data
    '''


def calculate_clusters(new_centroids: List[List[float]], old_centroids: List[List[float]], dataset: List[np.ndarray], old_clusters: List[List[int]], old_distances: List[List[int]]) -> List[List[int]]:
    ''' Calculates cluster formations and returns them

    new_centroids: A list of new centroids that need new clusters to be calculated for them
    old_centroids: A list of old centroids that already have clusters and are necessary for comparison between old and new
    dataset: A dataset containing the trainingdata for the algorithm
    old_clusters: A list of old clusters that correspond to the old_centroids
    old_distances: A list of old distances that correspond to both the old clusters and old centroids
    return: A list containing the index of matching datapoints for each of the new_centroids
    '''

    
def calculate_centroid_diff(new_centroids: List[List[float]], old_centroids: List[List[float]]) -> List[float]:
    ''' Calculate the difference in distance between two iterations of centroid
    
    new_centroids: The newly made centroids
    old_centroids: The previous itteration of centroids
    return: A list containing the distance between the iterations of centroids
    '''
            
        
def calculate_distance_to_centroids(given_day: np.ndarray, centroids: List[List[float]]) -> Tuple[int, float]:
    ''' This function calculates the distance from the given day to the centroids

    given_day: A list containing the data of the day you want to cluster
    centroids: A list of centroids that need new clusters to be calculated for them
    return: The index of the centroid and the difference between the distance to the nearest and second nearest centroids
    '''


def calculate_final_centroids(k: int, dataset: List[np.ndarray]) -> Tuple[List[List[float]], List[List[int]], List[List[float]]]:
    ''' Calculates the final centroid locations needed to process new data

    k: The amount of to be used centroids/clusters
    dataset: A dataset containing the trainingdata for the algorithm
    return: A tuple containing a list of centroids and a list of the corresponding clusters 
            and a list containing the distance between the centroids and the points they contain
    '''


def get_centroid_seasons(clusters: List[List[int]], dataset_labels: List[str]):
    ''' Returns the corresponding season for every cluster

    clusters: The clusters that need to be analysed
    dataset_labels: The labels for the dataset entries inside of the clusters
    return: A list filled with the seasons, corresponding to the cluster at the same index
    '''


def pinpoint_season(dataset: np.ndarray, dataset_labels: List[str], given_days: List[np.ndarray], k: int) -> Tuple[List[str], List[List[float]]]:
    ''' Pinpoint the season of the given data
    
    dataset: A dataset containing the trainingdata for the algorithm
    dataset_labels: The labels for the dataset entries inside of the clusters
    given_days: A list containing the data of the day you want to cluster
    k: The amount of to be used centroids/clusters
    return: A tuple containing a list with the seasons for each given_day and a list with all of the distances between centroid and the points it contains
    '''


def calculate_optimal_k(test_days: List[np.ndarray], test_labels: List[str], original_data: List[np.ndarray], original_labels: List[np.ndarray], k_min: int, k_max: int) -> Tuple[int, float]:
    ''' Calculates the optimal k-value

    test_days: A list containing all of the days that need to be assigned a season
    test_labels: A list containing all of the corresponding seasons to the test_days, for calculating a success rate
    original_data: A dataset containing the trainingdata for the algorithm
    original_labels: A list containing all of the corresponding seasons to the original_data
    k_min: The minimal k-value to be tested
    k_max: The maximum k-value to be tested
    return: The highest calculated succes rate and its corresponding k-value
    '''


def inter_cluster_distance(distances: List[List[float]]) -> float:
    ''' This function calculates the total distance of the point in the clusters

    distances: A list for each of the centroids with the distance between the centroid and the points it contains
    return: The calculated total distance
    '''


def plot_k(total_inter_cluster_distances: List[float], k_min: int, k_max: int):
    ''' A function for plotting the k-values against the total intercluster distances

    total_inter_cluster_distances: A list filled with the total intercluster distance for each cluster
    k_min: The lowest tested k-value
    k_max: The highest tested k-value
    '''


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

    # Calculate the seasons for the unknown data
    # print(pinpoint_season(original_data, original_labels, test_days, 4))

    # Calculate the optimal k-value
    print(calculate_optimal_k(test_days, test_labels, original_data, original_labels, 0, 40))


if __name__ == "__main__":
    # Run time test to compare with not optimised version of function
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))