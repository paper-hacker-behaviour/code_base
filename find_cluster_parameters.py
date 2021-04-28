# -*- coding: utf-8 -*-
"""
Optimise the number of clusters used in each dataset
"""

import glob
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_user_parameters(timebase, masking = 1):
    """
    Load the user parameters belonging to a single category

    Parameters
    ----------
    timebase : int
        0 = day
        1 = week
        2 = month
        3 = quarter
        4 = year
        
    masking : int, optional
        whether or not masking (dimensionality reduction) should be used. The default is 1.

    Returns
    -------
    parameterMatrix : numpy matrix
        (N,D) matrix of transition probabilities. N is the number of users in the timebase.
        D is the number of dimensions used.
    """
    folder_list = ["parameters_day/", "parameters_week/", "parameters_month/", "parameters_quarter/", "parameters_year/"]
    file_postfix_parameters = "parameter_matrix.csv"
    file_postfix_mask = "state_mask.csv"

    folder = folder_list[timebase]
    
    parameterMatrix = pd.read_csv(folder + file_postfix_parameters, header=None).to_numpy() 
    stateMask = pd.read_csv(folder + file_postfix_mask, header=None).to_numpy() 
    
    
    isMasked = 0
    #Has the parameter matrix already been masked off?
    if(len(parameterMatrix[0]) < len(stateMask)):
        isMasked = 1 #Data has been previously masked off
    
    #What to see if data needs changing
    if (masking == 1) and (isMasked == 0):
        print("Applying mask to {}".format(folder + file_postfix_parameters))
        #Mask needs to be applied to the data
        maskedMatrix = np.zeros((len(parameterMatrix), sum(stateMask))) #N.b. note the dimensions 
        #Loop over all users
        for i in range(len(parameterMatrix)):
            #Loop over valid user parameters
            index = 0
            for j in range(len(parameterMatrix[i])):
                if stateMask[j] == 1:
                    #Copy over the relevant values
                    maskedMatrix[i][index] = parameterMatrix[i][j]
                    index += 1
        
        parameterMatrix = maskedMatrix              
    elif (masking == 0) and (isMasked == 1):
        print("Removing mask from {}".format(folder + file_postfix_parameters))
        #Data needs to be unmasked (restores unused dimensions)
        unmaskedMatrix = np.zeros((len(parameterMatrix), len(stateMask))) #N.b. note the dimensions 
        #Loop over all users
        for i in range(len(parameterMatrix)):
            #Loop over all unreduced parameters
            index = 0
            for j in range(len(unmaskedMatrix[i])):
                if stateMask[j] == 1:
                    #Copy over the relevant values
                    unmaskedMatrix[i][j] = parameterMatrix[i][index]
                    index += 1
        
        parameterMatrix = unmaskedMatrix
        
    return parameterMatrix


def perform_clustering_optimisation(parameterMatrix, kMax, nInit, maxIterations):
    """
    For k clusters ranging from 2 to kMax, calculate the inertia and the silhoutte score

    Parameters
    ----------
    parameterMatrix : numpy matrix
        (N,D) matrix of transition probabilities. N is the number of users in the timebase.
        D is the number of dimensions used.
    kMax : int
        Maximum number of clusters to use in the search.
    nInit : int
        "Number of time the k-means algorithm will be run with different centroid seeds".
    maxIterations : int
        "Maximum number of iterations of the k-means algorithm for a single run.".

    Returns
    -------
    costVec : numpy array
        Used for the Elbow method.
    silhoutteVec : numpy array
        Used for the Silhoutte method.
    """
    #Initialise the vectors to return
    #costVec = np.zeros(kMax+1)
    #silhoutteVec = np.zeros(kMax+1)
    costVec = np.zeros(kMax-1)
    silhoutteVec = np.zeros(kMax-1)
    
    #k=0,1 are invalid
    #costVec[0:2] = -1
    #silhoutteVec[0:2] = -1
    
    #Loop through each value of k
    for k in range(2, kMax+1):
        print("    Testing, clusters: {}".format(k))
        kmeans = KMeans(n_clusters = k, n_init = nInit, max_iter = maxIterations).fit(parameterMatrix)
        labels = kmeans.labels_
        #silhoutteVec[k] = silhouette_score(parameterMatrix, labels, metric = 'euclidean')
        #costVec[k] = kmeans.inertia_
        silhoutteVec[k-2] = silhouette_score(parameterMatrix, labels, metric = 'euclidean')
        costVec[k-2] = kmeans.inertia_
        
    return costVec, silhoutteVec


def plot_optimisation_scores(costVec, silhoutteVec, kMax, plotTitle):
    """
    Plot the cost and silhoutte scores

    Parameters
    ----------
    costVec : numpy array
        Used for the Elbow method.
    silhoutteVec : numpy array
        Used for the Silhoutte method.
    kMax : int
        Maximum number of clusters to use in the search.
    plotTitle : string
        Title of the plot

    Returns
    -------
    None
    """
    #plt.clf()
    plt.figure()
    #Normalise the vectors
    costVec = costVec / max(costVec)
    silhoutteVec = silhoutteVec / max(silhoutteVec)

    #X-values
    xVec = range(2, kMax + 1)
    
    #Plot the arrays
    plt.plot(xVec, costVec, color ='r', linewidth ='2', label='Cost')
    plt.plot(xVec, silhoutteVec, color ='b', linewidth ='2', label='Silhoutte')
    plt.xlabel("Clusters (k)") 
    plt.ylabel("Normalised score") 
    plt.title(plotTitle)
    
    #Plot horizontal line across at max(silhoutteVec), as it is normalised y=1
    maxSilhoutteLine = np.ones(kMax - 1)
    plt.plot(xVec, maxSilhoutteLine, color ='b', linewidth ='1', linestyle='dashed')
    #Plot vertical line across at index that gives max(silhoutteVec)
    plt.axvline(x=np.argmax(silhoutteVec)+2, color ='b', linewidth ='1', linestyle='dashed' )
    
    plt.show()
    

def optimise_all_timebase_clusters():
    """
    Wrapper function allowing for clustering on all parameter matricies

    Returns
    -------
    None.
    """
    #These serve as rough estimtes as to where the optimal k will be found
    k_max = [50,100,150,150,100]
    n_Init = [100,100,100,100,100]
    max_Iterations = [1000,1000,1000,1000,1000]
    plot_titles = ["Day","Week","Month","Quarter","Year"]
    
    for i in range(5):
        print("Processing {} data".format(plot_titles[i]))
        user_matrix = load_user_parameters(i)
        cVec, sVec = perform_clustering_optimisation(user_matrix, k_max[i], n_Init[i], max_Iterations[i])
        plot_optimisation_scores(cVec, sVec, k_max[i], plot_titles[i])
        
        
def compute_cluster_centre_parameter_matrix(userMatrix, kOpt, nInit, maxIterations):
    """
    For the optimal number of clusters, save all the cluster centres as a parameter matrix

    Parameters
    ----------
    userMatrix : numpy matrix
        (N,D) matrix of transition probabilities. N is the number of users in the timebase.
        D is the number of dimensions used.
    kOpt : int
        Optimal number of clusters (found manually).
    nInit : int
        "Number of time the k-means algorithm will be run with different centroid seeds".
    maxIterations : int
        "Maximum number of iterations of the k-means algorithm for a single run".

    Returns
    -------
    clusterMatrix : numpy matrix
    (M,D) matrix of transition probabilities. M is the number of clusters (=kOpt).
        D is the number of dimensions used.
    userLabels : numpy array
    (N,1) array representing the cluster nearest to each of the N users.
    """
    #Initialise the return matrix and array
    clusterMatrix = np.zeros((kOpt, len(userMatrix[0])))
    userLabels = np.zeros(len(userMatrix))
    
    #Perform k-means clustering on the userMatrix
    kmeans = KMeans(n_clusters = kOpt, n_init = nInit, max_iter = maxIterations).fit(userMatrix)
    
    #Save the labels
    for i in range(len(userLabels)):
        userLabels[i] = kmeans.labels_[i]
    
    #Minimum value below which values should be rounded to zero
    minVal = 1.0e-6
    
    #Loop over each cluster
    for i in range(kOpt):
        #Loop over the indiviual values
        for j in range(len(userMatrix[0])):
            #Load the cluster parameter
            val = kmeans.cluster_centers_[i][j]
            if val < minVal:
                val = 0.0
                
            #Save the cluster parameter 
            clusterMatrix[i][j] = val
    
    return clusterMatrix, userLabels


def compute_all_cluster_centre_parameter_matrices(kOptList, nInit, maxIterations):
    """
    Wrapper function used to find all cluster centres and convert them to a parameter matrix
    This matrix is then saved

    Parameters
    ----------
    kOptList : list of ints
        Optimal number of clusters for each timebase (found manually).
    nInit : int
        "Number of time the k-means algorithm will be run with different centroid seeds".
    maxIterations : TYPE
        "Maximum number of iterations of the k-means algorithm for a single run".

    Returns
    -------
    None.
    """
    file_list_matrix = ["parameters_day/cluster_parameter_matrix.csv", "parameters_week/cluster_parameter_matrix.csv", 
                 "parameters_month/cluster_parameter_matrix.csv", "parameters_quarter/cluster_parameter_matrix.csv", 
                 "parameters_year/cluster_parameter_matrix.csv"]
    
    file_list_labels = ["parameters_day/cluster_membership.csv", "parameters_week/cluster_membership.csv", 
                 "parameters_month/cluster_membership.csv", "parameters_quarter/cluster_membership.csv", 
                 "parameters_year/cluster_membership.csv"]
    
    for i in range(5):
        print("Finding cluster centres")
        #Load in the user parameter matrix
        userMatrix = load_user_parameters(i)
        #Compute the parameter matrix for the cluster centres
        clusterMatrix, userLabels = compute_cluster_centre_parameter_matrix(userMatrix, kOptList[i], nInit, maxIterations)
        #Save the clusterMatrix into the parameters_x folder
        filename_matrix = file_list_matrix[i]
        pd.DataFrame(clusterMatrix).to_csv(filename_matrix,header=None, index=None)
        print("        Saved to {}".format(filename_matrix))
        
        #Save the userLabels into the parameters_x folder
        filename_labels = file_list_labels[i]
        pd.DataFrame(userLabels).to_csv(filename_labels,header=None, index=None)
        print("        Saved to {}".format(filename_labels))
    

##########################
"""   Main section    """
#########################
clustersPreviouslyFound = 1
kOptimal = [23, 50, 100, 100, 52] #These are the values I have chosen after running the script
nInit_clusters = 200
maxIterations_clusters = 2000

if clustersPreviouslyFound == 1:
    compute_all_cluster_centre_parameter_matrices(kOptimal, nInit_clusters, maxIterations_clusters)
else:
    plt.close()
    optimise_all_timebase_clusters()