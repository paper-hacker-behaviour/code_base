# -*- coding: utf-8 -*-
"""
Generate the Hidden Markov Model parameter matricies
"""

"""
State :
    0 = Initial inactivity
    1 = Searching 
    2 = inactive
    3 = submission
    4 = No further activity
"""

import glob
import pandas as pd
import numpy as np
import os
import shutil

class User:
    #Users are now represented as an object
    
    def __init__(self):
        #Define the class instance variables
        self.initialState = -1
        
        #Rows represent the current state
        #Columns represent the next state
        self.transitionMatrix = np.zeros((5,5))
        
    def set_initial_state(self, stateNumber):
        """
        Set the initial state of the HMM
        
        Parameters
        ----------
        stateNumber : int
            The first state in the user timeseries data.

        Returns
        -------
        None.
        """
        self.initialState = stateNumber
        
    def compute_transition_matrix(self, classifiedTimeseries):
        """
        Calculate the transition matrix parameteres using the classified timeseries data

        Parameters
        ----------
        classifiedTimeseries : numpy array
            classified timeseries data, each value is from set {0,1,2,3,4}.

        Returns
        -------
        None.
        """
        #Count state transitions 
        for i in range(len(classifiedTimeseries) - 1):
            #First order Markov model only considers current state -> next state
            currentState = int(classifiedTimeseries[i])
            nextState = int(classifiedTimeseries[i + 1])
            self.transitionMatrix[currentState][nextState] += 1
            
        #Convert the occurances to probabilities
        for row in range(len(self.transitionMatrix)):
            #Compute the total number of transitions from state(row) to all other states
            rowTotal = sum(self.transitionMatrix[row])
            
            if rowTotal >= 1.0:
                #Convert integer count to probability (0, 1]
                self.transitionMatrix[row] = self.transitionMatrix[row] / rowTotal
            else:
                #If there is no data in a row set all elements to zero
                self.transitionMatrix[row] *= 0.0
                #Although these elements should already be zero, this value can be modified
                #in order to give different probabilities to impossible state transitions
    
    
    def convert_transition_matrix_vector(self):
        """
        Reshape the transition matrix (5,5) into a transition vector (25,1)

        Returns
        -------
        transitionVector : numpy array
            25x1 vector representing the transition probabilities.
        """
        transitionVector = np.reshape(self.transitionMatrix, (25,1))
        return transitionVector
    

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


def generate_state_mask(timebaseList):
    """
    A mask used for dimensionality reduction.
    Any unused transitions across the timebase are asigned value of 0
    Any transitions that are used are asigned value of 1
    
    This produces a binary vector e.g. (0,0,0,0,1,0,0.....,0)
    that allows for the transition vector (25,1) to be cut down

    Parameters
    ----------
    timebaseList : list of numpy arrays
        A list containing all user data from a single timebase category.

    Returns
    -------
    stateMask : numpy array
        A fixed length vector (25,1) used for masking.
    usedTransitions : int
        The number of transitions used across all data within timebase.
    """
    #Initially defined as (5,5) matrix
    stateMask = np.zeros((5,5), dtype=int)
    for userData in timebaseList:
        for i in range(len(userData) - 1):
            currentState = int(userData[i])
            nextState = int(userData[i + 1])
            stateMask[currentState][nextState] += 1
            
    #Reshaped into a (25,1) vector
    stateMask = np.reshape(stateMask, (25,1))
    
    #Convert to binary mask
    for i in range(len(stateMask)):
        if stateMask[i] > 0:
            stateMask[i] = 1
            
    usedTransitions = sum(stateMask)
            
    return stateMask, usedTransitions


def reduce_transition_vector(transitionVector, stateMask, usedTransitions):
    """
    Reduce the number of dimensions of the transition vector 
    using the binary stateMask
    
    This removes transitions that are unused across all timeseries data
    within a timebase

    Parameters
    ----------
    transitionVector : numpy array
        A (25,1) vector of transition probabilities.
    stateMask : numpy array
        A (25,1) binary vector used to mask off dimensions.
    usedTransitions : int
        The number of used dimenions, sum(stateMask).

    Returns
    -------
    reducedVector : numpy array
        A (usedTransitions, 1) length vector of transition probabilities.
    """
    #The return vector will have it's size determined by the total number of
    #unique transitions used within the timebase
    reducedVector = np.zeros(usedTransitions)
    
    j = 0 #Used for indexing the reducedVector
    for i in range(len(transitionVector)):
        if stateMask[i] == 1:
            #If allowed by mask, store the transition probability
            reducedVector[j] = transitionVector[i]
            j += 1
    
    return reducedVector


def compute_timebase_parameter_matrix(timebase, masking = 1):
    """
    Convert all user timeseries data to a single matrix of transition probabilities
    
    Each row represents the state transition probabilities for a single user
    Dimensionality reduction may be used to remove unused columns 

    Parameters
    ----------
    timebase : int
        the timebase to process.
    masking : int, optional
        whether or not masking (dimensionality reduction) should be used. The default is 1.

    Returns
    -------
    None.
    """
    userData = load_user_data(timebase)
    
    
    #Generate the mask for dimensionality reduction (may not be used)
    stateMask, usedTransitions = generate_state_mask(userData)
        
    initialStateVector = np.zeros(len(userData), dtype=int)
    
    parameterMatrix = []
    
    #Process the data of each user
    for i in range(len(userData)):
        userObject = User()
        
        #Find the initial state
        initialState = userData[i][0]
        userObject.set_initial_state(initialState)
        initialStateVector[i] = initialState
    
        #Compute transition matrix
        userObject.compute_transition_matrix(userData[i])
        
        #Convert matrix to vector
        userVector = userObject.convert_transition_matrix_vector()
        #If needed, reduce the user transition vector
        if masking == 1:
            userVector = reduce_transition_vector(userVector, stateMask, usedTransitions)
    
        #Append the user transition vector to the matrix (currently stored as a list)
        parameterMatrix.append(userVector)
    
    
    #Convert the list to a numpy array
    parameterMatrix = np.asarray(parameterMatrix)
    
    #Save all the data
    save_parameter_estimations(timebase, parameterMatrix, stateMask, initialStateVector, overwrite = 1)
    

def save_parameter_estimations(timebase, parameterMatrix, stateMask, initialStateVector, overwrite = 0):
    """
    Save the relevant data generated within this file

    Parameters
    ----------
    timebase : int
        the timebase of the data to save.
    parameterMatrix : numpy matrix
        (N,D) matrix of transition probabilities. N is the number of users in the timebase.
        D is the number of dimensions used.
    stateMask : numpy array
        A (25,1) binary vector used to mask off dimensions.
    initialStateVector : numpy array
        (N,1) vector containing the initial state for all users.
    overwrite : int
        whether to overwrite the previously saved values

    Returns
    -------
    status_code : int
        has it been successful?
    """
    
    output_folder_dict = {0 : "parameters_day", 1 : "parameters_week", 2 : "parameters_month", 3 : "parameters_quarter", 4 : "parameters_year"}
    output_folder = output_folder_dict[timebase]
    
    #Try to create a new folder 
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        print("        Error, {} already exists".format(output_folder))
        if overwrite:
            shutil.rmtree(output_folder)
            print("        {} overwritten".format(output_folder))
            os.mkdir(output_folder)
        else:
            print("        Data not saved")
            return -1
    
    #Saves files to the new folder
    filename = output_folder + "/" + "parameter_matrix" + ".csv"
    pd.DataFrame(parameterMatrix).to_csv(filename,header=None, index=None)
    
    filename = output_folder + "/" + "state_mask" + ".csv"
    pd.DataFrame(stateMask).to_csv(filename,header=None, index=None)
    
    filename = output_folder + "/" + "initial_states" + ".csv"
    pd.DataFrame(initialStateVector).to_csv(filename,header=None, index=None)

    print("        Saved to {}".format(output_folder_dict[timebase]))
    
    return 1


def estimate_all_parameters(masking = 1):
    """
    This is used as a wrapper function to call compute_timebase_parameter_matrix
    for all timebases

    Parameters
    ----------
    masking : int, optional
        whether or not to produced masked parameter matricies. The default is 1.

    Returns
    -------
    None.
    """
    #Estimate parameters for all timebases
    for i in range(5):
        compute_timebase_parameter_matrix(i, masking)


##########################
"""   Main section    """
#########################
estimate_all_parameters()