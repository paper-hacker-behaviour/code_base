# -*- coding: utf-8 -*-
"""
This script performs validation on the HMM models using invidiual and clustered parameters
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

class User_HMM:
    """
    For all computations using the hidden Markov model, this class is used
    """
    #The number of times a HMM is run in the simulations
    __iterations = 10000
    
    #Maximum allowable error in the sum of probabilities
    __maxError = 1e-3
    
    def __init__(self, userID, userTimeseries, userParameterMatrix, clusterID, clusterParameterMatrix):
        #Store the index within the timebase data, e.g. timebase_list[i]
        self.userID = userID
        
        #Store the classified user timeseries data
        self.userTimeseries = userTimeseries
        
        #Store the initial state
        self.initialState = userTimeseries[0]
        
        #Store the length of the user timeseries
        self.timeseriesLength = len(userTimeseries)
        
        #Calculate the number of submission states in a user timeseries
        count = 0
        for i in range(len(userTimeseries)):
            if userTimeseries[i] == 3:
                count +=1
                
        self.submissionStates = count
        
        #Store the HMM parameters generated from the user data
        self.userParameterMatrix = np.reshape(userParameterMatrix,(5,5))
        
        #Store the cluster that the user belongs to
        self.clusterID = clusterID
        
        #Store the HMM parameters generated from the cluster data
        self.clusterParameterMatrix = np.reshape(clusterParameterMatrix,(5,5))
        
        
    def compute_user_submission_error(self):
        #Error is stored in a vector
        errorVec = np.zeros(self.__iterations)
        
        #Generate the choice matrix
        choiceMatrix = self.generate_choice_matrix(self.userParameterMatrix)
        for i in range(self.__iterations):
            #Run the hidden Markov model
            syntheticTimeseries = self.run_hmm_iteration(choiceMatrix[i])
            
            #Count the number of submitting states
            errorVec[i] = np.count_nonzero(syntheticTimeseries == 3)
        
        errorVec = np.mean(errorVec) - self.submissionStates
            
        return errorVec
    
    
    def compute_cluster_submission_error(self):
        #Error is stored in a vector
        errorVec = np.zeros(self.__iterations)
        
        #Generate the choice matrix, note that the parameter (cluster) matrix is used
        choiceMatrix = self.generate_choice_matrix(self.clusterParameterMatrix)
        for i in range(self.__iterations):
            #Run the hidden Markov model
            syntheticTimeseries = self.run_hmm_iteration(choiceMatrix[i])
            
            #Count the number of submitting states
            errorVec[i] = np.count_nonzero(syntheticTimeseries == 3)
        
        errorVec = np.mean(errorVec) - self.submissionStates
            
        return errorVec
        
        
    def calculate_next_state(self, stateProbabilities):
        """
        CURRENTLY UNUSED
        
        Using a list of the possible next states (with their associated probabilities)
        calculate a random value for the next state

        Parameters
        ----------
        stateProbabilities : numpy array
            (5,1) array containing the probabilities of going to the next states 
            (for a given current state).

        Returns
        -------
        nextState : int
            the next state represented as an integer.
        """
        stateProbabilities = np.reshape(stateProbabilities, (5))
        
        #Check that the sum of probabilities = 1
        if abs(1.0 - sum(stateProbabilities)) >= self.__maxError:
            #Normalise the probabilities unless sum == 0.0
            if sum(stateProbabilities) == 0.0:
                #This shouldn't occur, return a random state
                nextState = np.random.randint(5)
                return nextState
            
            #Normalise so that the sum == 1
            stateProbabilities /= sum(stateProbabilities)
        
        potentialStates = np.array([0,1,2,3,4])
        #This is super slow, needs optimising
        nextState = np.random.choice(potentialStates, p=stateProbabilities)
        return nextState
    
    
    def generate_choice_matrix(self, parameterMatrix):
        """
        Generates a Ix5xL matrix containing possible state transitions
        This is used to precompute state transitions

        Parameters
        ----------
        parameterMatrix : numpy matrix
            The matrix to be used to precompute the values.

        Returns
        -------
        choiceMatrix : numpy matrix
            Ix5xL matrix, I is the number of iterations, L is the length of the timeseries.
        """
        #choiceMatrix = np.zeros((5,self.timeseriesLength),dtype=int)
        tempStateArray = np.zeros(self.timeseriesLength, dtype=int)
        choiceMatrix = np.zeros((self.__iterations,5,self.timeseriesLength),dtype=int)
        
        #From each possible current state
        for i in range(5):
            #Load the transition probabilities
            stateProbabilities = np.reshape(parameterMatrix[i], (5))
            
            #Check to see if there are valid probability values
            if sum(stateProbabilities) == 0.0:
                #Set random states if all transitions are unused
                tempStateArray = np.random.randint(5, size=(self.__iterations,self.timeseriesLength), dtype=int)
                choiceMatrix[:,i] = tempStateArray
                #choiceMatrix[i] = tempStateArray
            else:
                #Precompute state transitions
                #Normalise so that the sum == 1
                stateProbabilities /= sum(stateProbabilities)
                potentialStates = np.array([0,1,2,3,4])
                tempStateArray = np.random.choice(potentialStates, size=(self.__iterations,self.timeseriesLength), p=stateProbabilities)
                choiceMatrix[:,i] = tempStateArray
        
        return choiceMatrix
        
        
    def run_hmm_iteration(self, choiceMatrix):
        """
        Run a single iteration of the HMM
        
        Parameters
        ----------
        choiceMatrix : numpy matrix
            5xL matrix, L is the length of the timeseries.

        Returns
        -------
        syntheticTimeseries : numpy array
        The list of states synthesized
        """
        syntheticTimeseries = np.zeros(self.timeseriesLength)
        #Initialise the current state to the stored initial state
        currentState = int(self.initialState)
        syntheticTimeseries[0] = currentState
        
        for i in range(1,self.timeseriesLength):
            #Calculate the next state to travel to
            nextState = choiceMatrix[currentState][i]
            #Store the next state
            syntheticTimeseries[i] = nextState            
            currentState = nextState
            
        return syntheticTimeseries
    
    
    def compute_user_MAE(self):
        """
        Compute the MAE for the time spent in each state for a user

        Returns
        -------
        MeanStateError : numpy array
            MAE for time spent in each state.
        """
        #Initialise the error for time spent in each state
        StateError = np.zeros((5,self.__iterations))
        MeanStateError = np.zeros(5)
        
        #Generate the choice matrix
        choiceMatrix = self.generate_choice_matrix(self.userParameterMatrix)
        
        #Predicted state count
        predictedCounts = np.zeros(5)
        
        #Actualy state count
        actualCounts = np.zeros(5)
        for i in range(5):
            actualCounts[i] = np.count_nonzero(self.userTimeseries == i)
        
        
        for i in range(self.__iterations):
            #Run the hidden Markov model
            syntheticTimeseries = self.run_hmm_iteration(choiceMatrix[i])
            
            #Count the occurances
            for j in range(5):
                predictedCounts[j] = np.count_nonzero(syntheticTimeseries == j)
                #Compute MAE
                StateError[j,i] = np.abs(predictedCounts[j] - actualCounts[j])
            
            
        #Compute the mean of the absolute error
        for i in range(5):
            MeanStateError[i] = np.mean(StateError[i])
        
        return MeanStateError
    
    
    def compute_cluster_MAE(self):
        """
        Compute the MAE for the time spent in each state for a cluster

        Returns
        -------
        StateError : numpy array
            MAE for time spent in each state.
        """
        #Initialise the error for time spent in each state
        StateError = np.zeros((5,self.__iterations))
        MeanStateError = np.zeros(5)
        
        #Generate the choice matrix
        choiceMatrix = self.generate_choice_matrix(self.clusterParameterMatrix)
        
        #Predicted state count
        predictedCounts = np.zeros(5)
        
        #Actualy state count
        actualCounts = np.zeros(5)
        for i in range(5):
            actualCounts[i] = np.count_nonzero(self.userTimeseries == i)
        
        
        for i in range(self.__iterations):
            #Run the hidden Markov model
            syntheticTimeseries = self.run_hmm_iteration(choiceMatrix[i])
            
            #Count the occurances
            for j in range(5):
                predictedCounts[j] = np.count_nonzero(syntheticTimeseries == j)
                #Compute MAE
                StateError[j,i] = np.abs(predictedCounts[j] - actualCounts[j])
            
            
        #Compute the mean of the absolute error
        for i in range(5):
            MeanStateError[i] = np.mean(StateError[i])
        
        return MeanStateError
            
        

def load_user_data(timebase):
    """
    Load the user data belonging to a single category
    E.g. Find all users that have data displayed at 1 year resolution

    Parameters
    ----------
    timebase : int
        0 = day
        1 = week
        2 = month
        3 = quarter
        4 = year

    Returns
    -------
    timebase_list : list of numpy arrays
        A list containing all user data from a single timebase category.
    """
    folder_list = ["classified_data_day/", "classified_data_week/", "classified_data_month/", "classified_data_quarter/", "classified_data_year/"]

    folder = folder_list[timebase]
    
    file_list = [f for f in glob.glob(folder + "*.csv")]
    timebase_list = [None]*len(file_list)
    counter = 0
    for f in file_list:
        df_temp = pd.read_csv(f, header=None)
        timebase_list[counter] = df_temp.to_numpy() 
        counter += 1
        
    return timebase_list


def load_user_metadata(timebase):
    """
    For each timebase load the following .csv files:
        (is) initial_states : the recorded first state for each user
        (sm) state_mask : the mask used for dimensionality reduction
        (pm) parameter_matrix : the HMM parameters previously calculated for each user
        (cpm) cluster_patameter_matrix : the HMM parameters previously calculated for each cluster 
        (cm) cluster_membership : for each user, the index of the cluster that they belong to
        
    Parameters
    ----------
    timebase : int
        the timebase to load represented as an integer (see function definition above).

    Returns
    -------
    metadata_is : numpy array
        Array contianing the initial states for each user.
    metadata_sm : numpy array
        The state mask used for dimensionality reduction
    metadata_pm : numpy matrix
        HMM parameters for all users, without dimensionality reduction
    metadata_cpm : numpy matrix
        HMM parameters for each cluster, without dimensionality reduction
    metadata_cm : numpy array
        For each user the label of the nearest cluster is stored
    """
    #Define all folder and file names
    folder_list = ["parameters_day/", "parameters_week/", "parameters_month/", "parameters_quarter/", "parameters_year/"]
    file_postfix_is = "initial_states.csv"
    file_postfix_sm = "state_mask.csv"
    file_postfix_pm = "parameter_matrix.csv"
    file_postfix_cpm = "cluster_parameter_matrix.csv"
    file_postfix_cm = "cluster_membership.csv"
    
    #Load all relevant files
    metadata_is = pd.read_csv(folder_list[timebase] + file_postfix_is, header=None).to_numpy()
    metadata_sm = pd.read_csv(folder_list[timebase] + file_postfix_sm, header=None).to_numpy() 
    metadata_pm = pd.read_csv(folder_list[timebase] + file_postfix_pm, header=None).to_numpy() 
    metadata_cpm = pd.read_csv(folder_list[timebase] + file_postfix_cpm, header=None).to_numpy() 
    metadata_cm = pd.read_csv(folder_list[timebase] + file_postfix_cm, header=None).to_numpy()
    
    #Check to see if the unused dimensions need restoring
    if len(metadata_pm[0]) < len(metadata_sm):
        #Restore dimensions in parameter matrix
        metadata_pm = remove_masking(metadata_pm, metadata_sm)
        
    if len(metadata_cpm[0]) < len(metadata_sm):
        #Restore dimensions in cluster parameter matrix
        metadata_cpm = remove_masking(metadata_cpm, metadata_sm)
    
    return metadata_is, metadata_sm, metadata_pm, metadata_cpm, metadata_cm


def remove_masking(parameterMatrix, stateMask):
    """
    Used to restore unused (masked off) dimensions to HMM parameter matrices

    Parameters
    ----------
    parameterMatrix : numpy matrix 
        HMM parameters for all users/clusters.
    stateMask : numpy array
        a mask of dimensions.

    Returns
    -------
    unmaskedMatrix : numpy matrix 
        the parameterMatrix expanded to include unused dimensions.
    """
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
        
    return unmaskedMatrix
    

def calculate_start_end_state_distribution():
    """
    For all users over all timebases, calculate the distribution of starting 
    and final states. Saves into CSV files

    Returns
    -------
    None.
    """
    timebaseLabels = ["Day    ", "Week   ", "Month  ", "Quarter", "Year   "]
    
    #Use 3D matrix to store values
    stateCountMatrix = np.zeros((2,5,5)) #2 values (start,end), 5 timebases, 5 states
    #Loop over all timebases
    for i in range(5):
        print("Loading timebase {}".format(timebaseLabels[i]))
        #Load in the user data
        userData = load_user_data(i)
        #Loop over each user
        for j in range(len(userData)):
            #Access the first state for user j, use that as an index to increment the count
            stateCountMatrix[0][i][userData[j][0]] += 1
            
            #Access the last state for user j, use that as an index to increment the count
            stateCountMatrix[1][i][userData[j][-1]] += 1
            
    #Print out the results
    #Starting states
    print("-----------------------------------")
    print("Distribution of initial states")
    print("{}    {:4.0f} {:4.0f} {:4.0f} {:4.0f} {:4.0f}".format("State  ", 0, 1, 2, 3, 4))
    print("-----------------------------------")
    for i in range(5):
        print("{}    {:4.0f} {:4.0f} {:4.0f} {:4.0f} {:4.0f}".format(timebaseLabels[i], stateCountMatrix[0][i][0],
                                                                     stateCountMatrix[0][i][1],stateCountMatrix[0][i][2],
                                                                     stateCountMatrix[0][i][3],stateCountMatrix[0][i][4]))
    #Final states
    print("-----------------------------------")
    print("Distribution of final states")
    print("{}    {:4.0f} {:4.0f} {:4.0f} {:4.0f} {:4.0f}".format("State  ", 0, 1, 2, 3, 4))
    print("-----------------------------------")
    for i in range(5):
        print("{}    {:4.0f} {:4.0f} {:4.0f} {:4.0f} {:4.0f}".format(timebaseLabels[i], stateCountMatrix[1][i][0],
                                                                     stateCountMatrix[1][i][1],stateCountMatrix[1][i][2],
                                                                     stateCountMatrix[1][i][3],stateCountMatrix[1][i][4]))
    print("-----------------------------------")
    
    #Save the data into a .csv file
    pd.DataFrame(stateCountMatrix[0],columns=["state 0","state 1","state 2","state 3","state 4"]).to_csv("initial_state_distribution.csv", index=None)
    pd.DataFrame(stateCountMatrix[1],columns=["state 0","state 1","state 2","state 3","state 4"]).to_csv("final_state_distribution.csv", index=None)


def calculate_expected_submission_error_user(timebase):
    """
    Calculate the expected error when predicting the number of submission states 
    a user has

    Parameters
    ----------
    timebase : int
        the timebase to load represented as an integer (see function definition above).

    Returns
    -------
    expectedError : float
        the expected error across all users in the timebase.
    """
    #Load in the classified user data
    userData = load_user_data(timebase)
    #Load in the user metadata
    metadataIS, metadataSM, metadataPM, metadataCPM, metadataCM = load_user_metadata(timebase)
    #Initialise the submission error array
    userError = np.zeros(len(userData))
    
    #Loop over all users
    for i in range(len(userData)):
        #Create object
        userModelInstance = User_HMM(i, userData[i], metadataPM[i], metadataCM[i], metadataCPM[int(metadataCM[i])])
        #Calculate error
        submissionError = userModelInstance.compute_user_submission_error()
        userError[i] = submissionError
        
    expectedError = np.mean(userError)
    return expectedError


def calculate_all_expected_submission_error_user():
    """
    Wrapper function for calculate_expected_submission_error_user, calls for all timebases
    Saves the values in a csv file

    Returns
    -------
    None.
    """
    expectedError = np.zeros(5)
    timebaseLabels = ["Day", "Week", "Month", "Quarter", "Year"]
    for i in range(5):
        print("Processing {} users".format(timebaseLabels[i]))
        expectedError[i] = calculate_expected_submission_error_user(i) * 100.0

    #Save the expected error for users
    expectedError = np.transpose(expectedError)
    pd.DataFrame(expectedError,columns=["Error"]).to_csv("expected_submission_error_user.csv", index=None)
    
    
def calculate_expected_submission_error_cluster(timebase):
    """
    Calculate the expected error when predicting the number of submission states 
    a cluster has

    Parameters
    ----------
    timebase : int
        the timebase to load represented as an integer (see function definition above).

    Returns
    -------
    expectedError : float
        the expected error across all clusters in the timebase.
    """
    #Load in the classified user data
    userData = load_user_data(timebase)
    #Load in the user metadata
    metadataIS, metadataSM, metadataPM, metadataCPM, metadataCM = load_user_metadata(timebase)
    #Initialise the submission error array
    userError = np.zeros(len(userData))
    
    #Loop over all users
    for i in range(len(userData)):
        #Create object
        userModelInstance = User_HMM(i, userData[i], metadataPM[i], metadataCM[i], metadataCPM[int(metadataCM[i])])
        #Calculate error
        submissionError = userModelInstance.compute_cluster_submission_error()
        userError[i] = submissionError
        
    expectedError = np.mean(userError)
    return expectedError
    

def calculate_all_expected_submission_error_cluster():
    """
    Wrapper function for calculate_expected_submission_error_cluster, calls for all timebases
    Saves the values in a csv file

    Returns
    -------
    None.
    """
    expectedError = np.zeros(5)
    timebaseLabels = ["Day", "Week", "Month", "Quarter", "Year"]
    for i in range(5):
        print("Processing {} clusters".format(timebaseLabels[i]))
        expectedError[i] = calculate_expected_submission_error_cluster(i) * 100.0

    #Save the expected error for users
    expectedError = np.transpose(expectedError)
    pd.DataFrame(expectedError,columns=["Error"]).to_csv("expected_submission_error_cluster.csv", index=None)
    
    
def calculate_all_MAE_user():
    """
    Wrapper function for calculate_MAE_user, calls for all timebases
    Saves the values in a csv file

    Returns
    -------
    None.
    """
    
    AllStateMAE = np.zeros((5,5))
    timebaseLabels = ["Day", "Week", "Month", "Quarter", "Year"]
    for i in range(5):
        print("Processing {} user MAE".format(timebaseLabels[i]))
        AllStateMAE[i] = calculate_MAE_user(i)
        
    pd.DataFrame(AllStateMAE,columns=["0","1","2","3","4"]).to_csv("MAE_state_user.csv", index=None)
    
    
def calculate_MAE_user(timebase):
    """
    Calculate the MAE when predicting the time spent in each state for users

    Parameters
    ----------
    timebase : int
        the timebase to load represented as an integer (see function definition above).

    Returns
    -------
    MAE : numpy array of floats
        the MAE for each state across all users in the timebase.
    """
    #Load in the classified user data
    userData = load_user_data(timebase)
    #Load in the user metadata
    metadataIS, metadataSM, metadataPM, metadataCPM, metadataCM = load_user_metadata(timebase)
    #Initialise the error for time spent in each state
    StateError = np.zeros((len(userData), 5))
    
    #Loop over all users
    for i in range(len(userData)):
        #Create object
        userModelInstance = User_HMM(i, userData[i], metadataPM[i], metadataCM[i], metadataCPM[int(metadataCM[i])])
        #Calculate error
        submissionError = userModelInstance.compute_user_MAE()
        StateError[i] = submissionError
    
    MAE = np.zeros(5)
    #Loop over each state
    for i in range(5):
        MAE[i] = np.mean(StateError[:,i])

    return MAE
    

def calculate_all_MAE_cluster():
    """
    Wrapper function for calculate_MAE_cluster, calls for all timebases
    Saves the values in a csv file

    Returns
    -------
    None.
    """
    
    AllStateMAE = np.zeros((5,5))
    timebaseLabels = ["Day", "Week", "Month", "Quarter", "Year"]
    for i in range(5):
        print("Processing {} cluster MAE".format(timebaseLabels[i]))
        AllStateMAE[i] = calculate_MAE_cluster(i)
        
    pd.DataFrame(AllStateMAE,columns=["0","1","2","3","4"]).to_csv("MAE_state_cluster.csv", index=None)
    
    
def calculate_MAE_cluster(timebase):
    """
    Calculate the MAE when predicting the time spent in each state for clusters

    Parameters
    ----------
    timebase : int
        the timebase to load represented as an integer (see function definition above).

    Returns
    -------
    expectedError : numpy array of floats
        the MAE for each state across all clusters in the timebase.
    """
    #Load in the classified user data
    userData = load_user_data(timebase)
    #Load in the user metadata
    metadataIS, metadataSM, metadataPM, metadataCPM, metadataCM = load_user_metadata(timebase)
    #Initialise the error for time spent in each state
    StateError = np.zeros((len(userData), 5))
    
    #Loop over all users
    for i in range(len(userData)):
        #Create object
        userModelInstance = User_HMM(i, userData[i], metadataPM[i], metadataCM[i], metadataCPM[int(metadataCM[i])])
        #Calculate error
        submissionError = userModelInstance.compute_cluster_MAE()
        StateError[i] = submissionError
    
    MAE = np.zeros(5)
    #Loop over each state
    for i in range(5):
        MAE[i] = np.mean(StateError[:,i])

    return MAE


##########################
"""   Main section    """
#########################
recalculateStartEndStates = 0
recalculateSubmissionErrorUser = 0
recalculateSubmissionErrorCluster = 0
recalculateMaeUser = 0
recalculateMaeCluster = 0

if recalculateStartEndStates:
    calculate_start_end_state_distribution()

if recalculateSubmissionErrorUser:
    calculate_all_expected_submission_error_user()
    
if recalculateSubmissionErrorCluster:
    calculate_all_expected_submission_error_cluster()

if recalculateMaeUser:
    calculate_all_MAE_user()

if recalculateMaeCluster:
    calculate_all_MAE_cluster()
    
"""
Validation results:

All use 10,000 synthetic timeseries

2)
Mean absolute error - time in each state for users

5)
Mean absolute error - time in each state for clusters

"""

