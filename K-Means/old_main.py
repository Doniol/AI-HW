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
    for data_type in range(0, len(dataset_to_be_changed[0])):
        for day in dataset_to_be_changed:
            day[data_type] = (day[data_type] - min_values[data_type]) / (max_values[data_type] - min_values[data_type])


def success_rate_calculation(results: List[str], answers: List[str]) -> float:
    ''' Calculates and returns the % of correct results
    
    results: List of results from the pinpoint_season calculations
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


def create_starting_centroids(num_data_types: int, k=4) -> List[List[float]]:
    ''' Calculate starting centroids
    num_data_types: The amount of data_types the centroid has
    k: The number centroids you want to have
    return: A list containing all created centroids
    '''
    centroids = []
    # Create k amount of centroids, with each datatype being random
    for centroid in range(0, k):
        new_centroid = []
        for datapoint in range(0, num_data_types):
            new_centroid.append(random.uniform(0, 1))
        centroids.append(new_centroid)
    return centroids


def calculate_centroid_location(centroids: List[List[float]], clusters: List[List[int]], dataset: List[np.ndarray]) -> List[List[float]]:
    ''' Calculate the new mean/location of the centroid
    centroids: The already existing centroids
    clusters: The clusters corresponding to the given centroids
    dataset: A dataset containing the trainingdata for the algorithm
    return: The updated centroid data
    '''
    new_centroids = []
    # Loop through each centroid
    for centroid_index in range(0, len(centroids)):
        # Check whether centroid has its own cluster
        if clusters[centroid_index]:
            recalculated_centroid = []
            # Loop through each datatype in the data_point
            for data_index in range(0, len(dataset[0])):    
                average = 0
                # Calculate the average of the datatype for the centroid
                for data_point_index in clusters[centroid_index]:   
                    average += dataset[data_point_index][data_index]
                average =  average / len(clusters[centroid_index])
                recalculated_centroid.append(average)
            new_centroids.append(recalculated_centroid)
        # If centroid has no existing cluster, recalculation aren't necessary and add the original centroid-data
        else:
            new_centroids.append(centroids[centroid_index])
    return new_centroids


def calculate_clusters(centroids: List[List[float]], dataset: List[np.ndarray]) -> List[List[int]]:
    ''' Calculates cluster formations and returns them
    centroids: A list of centroids that need new clusters to be calculated for them
    dataset: A dataset containing the trainingdata for the algorithm
    return: A list containing the index of matching datapoints for each of the centroids
    '''
    # Create empty list of clusters 
    # (necessary because the corresponding centroids and clusters have to be stored at the same index of their respective lists)
    clusters = [[] for i in range(0, len(centroids))]
    # Select for each dataset entry the closest centroid, and save a reference to said entry at the correct index of clusters
    for dataset_index in range(0, len(dataset)):
        closest_centroid = calculate_distance_to_centroids(dataset[dataset_index], centroids)
        clusters[closest_centroid].append(dataset_index)
    return clusters
        
        
def calculate_distance_to_centroids(given_day: np.ndarray, centroids: List[List[float]]) -> int:
    ''' This function calculates the distance from the given day to it's neighbours
    given_day: A list containing the data of the day you want to cluster
    centroids: A list of centroids that need new clusters to be calculated for them
    return: the integer of the centroid
    '''
    distance_to_centroids = []
    # loop through each centroid
    for centroid in centroids:
        distance = 0
        # Calculate the distance between the centroid en the data to compare
        for data_type_index in range(0, len(given_day)):
            if given_day[data_type_index] > centroid[data_type_index]:
                    distance += given_day[data_type_index] - centroid[data_type_index]
            else:
                distance += centroid[data_type_index] - given_day[data_type_index]
        distance_to_centroids.append(distance)
    # Return the index of the centroid with the lowest distance
    return distance_to_centroids.index(min(distance_to_centroids))
    

def calculate_final_centroids(k: int, dataset: List[np.ndarray]) -> Tuple[List[List[float]], List[List[int]]]:
    ''' Calculates the final centroid locations needed to process new data
    k: The amount of to be used centroids/clusters
    dataset: A dataset containing the trainingdata for the algorithm
    return: A tuple containing a list of centroids and a list of the corresponding clusters
    '''
    centroids = []
    new_centroids = create_starting_centroids(len(dataset[0]), k)
    clusters = []
    # As long as the data of the centroids changes, keep recalculating the centroids 
    while centroids != new_centroids:
        centroids = new_centroids
        clusters = calculate_clusters(new_centroids, dataset)
        new_centroids = calculate_centroid_location(new_centroids, clusters, dataset)
    return new_centroids, clusters


def get_centroid_seasons(clusters: List[List[int]], dataset_labels: List[str]):
    ''' Returns the corresponding season for every cluster
    clusters: The clusters that need to be analysed
    dataset_labels: The labels for the dataset entries inside of the clusters
    return: A list filled with the seasons, corresponding to the cluster at the same index
    '''
    seasons = []
    for cluster_index in range(0, len(clusters)):
        # Create a dictionary that keeps count of the counted seasons within the cluster, then select the most common one
        season_dict = {
            "winter": 0,
            "herfst": 0,
            "zomer": 0,
            "lente": 0
        }
        for label_index in clusters[cluster_index]:
            season_dict[dataset_labels[label_index]] += 1
        seasons.append(max(season_dict, key=season_dict.get))
    return seasons


def pinpoint_season(dataset: np.ndarray, dataset_labels: List[str], given_days: List[np.ndarray], k: int) -> List[str]:
    ''' Pinpoint the season of the given data
    
    dataset: A dataset containing the trainingdata for the algorithm
    dataset_labels: The labels for the dataset entries inside of the clusters
    given_days: A list containing the data of the day you want to cluster
    k: The amount of to be used centroids/clusters
    return: A list containing the seasons for each given_day
    '''
    # Calculate the starting data
    centroids, clusters = calculate_final_centroids(k, dataset)
    seasons = get_centroid_seasons(clusters, dataset_labels)
    results = []
    # Calculate for each given_day the correct season
    for given_day in given_days:
        closest_sentroid = calculate_distance_to_centroids(given_day, centroids)
        results.append(seasons[closest_sentroid])
    return results


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
        predicted_seasons = pinpoint_season(original_data, original_labels, test_days, k_value)
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

    # Calculate the seasons for the unknown data
    # print(pinpoint_season(original_data, original_labels, test_days, 4))

    # Calculate the optimal k-value
    print(calculate_optimal_k(test_days, test_labels, original_data, original_labels, 0, 40))


if __name__ == "__main__":
    # Run time test to compare with optimised version of function
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))