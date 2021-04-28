# -*- coding: utf-8 -*-

import shutil
import os
import glob
import numpy as np
import pandas as pd

"""
State :
    0 = Initial inactivity
    1 = Searching 
    2 = inactive
    3 = submission
    4 = No further activity
"""

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
    folder_list = ["users_day/", "users_week/", "users_month/", "users_quarter/", "users_year/"]

    folder = folder_list[timebase]
    
    file_list = [f for f in glob.glob(folder + "*.csv")]
    timebase_list = [None]*len(file_list)
    counter = 0
    for f in file_list:
        df_temp = pd.read_csv(f).drop(columns=["Date"])
        df_temp = df_temp[:-1] #N.B last data point always removed 
        timebase_list[counter] = df_temp.to_numpy() 
        counter += 1
        
    return timebase_list

def load_inactivity_limit(filename, percentile):
    """
    Load the inactivity limit from the csv file.

    Parameters
    ----------
    filename : string
        filename of csv file.
    percentile : string
        the percentile of users that are to be classified as active.

    Returns
    -------
    df.iloc[1] : float
        The mean number of days after which a user can be classified as inactive (permanently).
    """
    df = pd.read_csv(filename)
    try:
        df = df[percentile]
    except KeyError:
        print("Error: Pre-computed inactivity limit not found, defaulting to initial value")
        return df.loc[1][1]
    #Return only the mean number of days
    return df.iloc[1]

def convert_inactive_days_to_timebase(inactivity_limit_days):
    """
    Convert the limit in days to an appropriate value in all timebases

    Parameters
    ----------
    inactivity_limit_days : float
        The mean number of days after which a user can be classified as inactive (permanently).

    Returns
    -------
    inactivity_timebase_limit : numpy array
        The maximum number of lagging inactive intervals before a user is inactive.
    """
    
    inactivity_timebase_limit = np.zeros(5, dtype=int)
    inactivity_limit_days = int(inactivity_limit_days)
    timebase_ratio = np.array([1,7,30,90,365])
    
    inactivity_timebase_limit = inactivity_limit_days / timebase_ratio
    inactivity_timebase_limit = np.floor(inactivity_timebase_limit)
    
    return inactivity_timebase_limit


def classify_submissions(user_data_list):
    """
    For all user data in the dataframe,
    classify all submission events.
    
    As shown at the start of the file, 
    the state code for submission events is: 3
    
    All unclassified events are given code -1
    
    This classification MUST be performed first

    Parameters
    ----------
    user_data_list : list of arrays
        each row contains the data for one user,
        the column contains the timerseies data as an array.

    Returns
    -------
    classified_data : list of arrays
        Same format as input, but data is partially classified.
    """
    #Create an empty list for the return data
    classified_data = [None] * len(user_data_list)
    
    #Loop over all users
    for i in range(len(user_data_list)):
        #Define a temporary array
        classified_array = np.zeros(len(user_data_list[i]),dtype=int)
        
        for j in range(len(user_data_list[i])):
            val = user_data_list[i][j]
            
            #If the user has submitted vulnerability reports in a 
            #given interval, assign state code 3    
            if val > 0:
                classified_array[j] = 3
            else:
                classified_array[j] = -1
                
        #Put the temporary array in the return list
        classified_data[i] = classified_array
       
    return classified_data


def classify_searching(user_data_list, timebase):
    """
    Following the classification of submission events,
    searching behaviour is classified using preset values
    
    This must be done after classifying the submitting states

    Parameters
    ----------
    user_data_list : list of arrays
        user data - submission states must be classified.    
    timebase : int
        the integer value of the timebase (see top of file)

    Returns
    -------
    classified_data : list of arrays
        user data - with any neccessary searching states added.
    """
    
    #The search limits come from other literature
    #This corresponds to a max search period before a submission, e.g. 7 days, or 1 week, or 1 month
    search_limits = [7,1,1,0,0]
    
    #No classifying neccesary for quarterly or yearly data
    if timebase == 3 or timebase == 4:
        return user_data_list
    
    classified_data = user_data_list
    
    #The search limit to be used given the timebase
    search_limit = search_limits[timebase]
    
    #Loop through the data for each user
    for i in reversed(range(len(user_data_list))):
        start_search = 0
        #Loop through the data for a given user
        for j in reversed(range(len(user_data_list[i]))):
            #Load in the current state
            current_state = user_data_list[i][j]
            
            #If previously found submitting state
            #Classify subsequent as searching
            if start_search > 0:
                if current_state != 3:
                    #Set state to searching
                    classified_data[i][j] = 1
                    start_search -= 1
            
            #Check if state == submitting
            if current_state == 3:
                start_search = search_limit
    
    return classified_data


def classify_inactivity_leading(user_data_list):
    """
    This classifies any intervals before the initial submission 
    (and searching period if applicable) as "Initial inactivity"

    Parameters
    ----------
    user_data_list : list of arrays
        user data - submission states must be classified.

    Returns
    -------
    classified_data : list of arrays
        user data - with any neccessary "initial inactivity" states added.
    """
    
    classified_data = user_data_list
    
    for i in range(len(user_data_list)):
        #Find the first classified index in the user data
        initial_index = -1
        for j in range(len(user_data_list[i])):
            #Check to see if interval has been previously classified
            if user_data_list[i][j] > -1:
                #If classified, save index and exit loop
                initial_index = j
                break
        
        
        #If the first state is classified OR error, move to the next user
        if initial_index == 0 or initial_index == -1:
            continue
        
        #Classify all neccessary intervals as being initially inactive
        for j in range(initial_index):
            classified_data[i][j] = 0
    
    return classified_data


def classify_inactivity_lagging(user_data_list, timebase, inactivity_limit):
    """
    For lagging inactivity that is longer than the acceptible value
    the state "No further activity" is assigned to the neccessary intervals
    
    If (number of lagging inactive intervals) > (limit):
        Reclassify those states

    Parameters
    ----------
    user_data_list : list of arrays
        user data - submission states must be classified.
    timebase : int
        the integer value of the timebase (see top of file).
    inactivity_limit : numpy array 
        maximum number of inactive lagging intervals afterwhich user is to be reclassified.

    Returns
    -------
    classified_data : list of arrays
        user data - with any neccessary "no further activity" states added.
    """
    classified_data = user_data_list
    number_inactive_users = 0
    
    #Is is possible that all user data within a timebase will be too short 
    #for any users to be long term inactive
    limit = inactivity_limit[timebase]
    for i in range(len(user_data_list)):
        #Check if user timeseries is longer than limit
        if len(user_data_list[i] > limit):
            #Calculate the number of lagging inactive intervals
            lagging_intervals = 0
            for j in reversed(range(len(user_data_list[i]))):
                #Check that states are unclassified
                if user_data_list[i][j] == -1:
                    lagging_intervals += 1
                else:
                    break
                
            #Is the number of lagging intervals too large?
            if lagging_intervals > limit:
                #Reclassify lagging intervals
                number_inactive_users += 1
                for j in reversed(range(len(user_data_list[i]))):
                    if user_data_list[i][j] == -1:
                        classified_data[i][j] = 4
                    else:
                        break
    
    return classified_data, number_inactive_users


def classify_inactivity_internal(user_data_list):
    """
    All remaining unclassified intervals are therefore internal intervals
    and are given the state "inactive"
    
    Following this, all states are classified

    Parameters
    ----------
    user_data_list : list of arrays
        user data - submission states must be classified.

    Returns
    -------
    classified_data : list of arrays
        user data - with any neccessary "inactive" states added..
    """
    classified_data = user_data_list
    
    for i in range(len(user_data_list)):
        for j in range(len(user_data_list[i])):
            current_state = user_data_list[i][j]
            #If state is undefined, it is reclassified to the correct value
            if current_state == -1:
                classified_data[i][j] = 2
    
    return classified_data


def classify_timebase_user_data(unclassified_data, timebase, inactivity_limit):
    """
    Wrapper function that appropriately calls the classification functions

    Parameters
    ----------
    unclassified_data : list of arrays
        user data - unclassified.
    timebase : int
        the integer value of the timebase (see top of file).
    inactivity_limit : numpy array 
        maximum number of inactive lagging intervals afterwhich user is to be reclassified.

    Returns
    -------
    classified_data : list of arrays
        user data - fully classified.
    inactive_users : int
        total number of inactive users within timebase
    """
    classified_data = classify_submissions(unclassified_data)
    classified_data = classify_searching(classified_data, timebase)
    classified_data = classify_inactivity_leading(classified_data)
    classified_data, inactive_users = classify_inactivity_lagging(classified_data, timebase, inactivity_limit)
    classified_data = classify_inactivity_internal(classified_data)
    
    return classified_data, inactive_users


def classify_all_user_data(percentile, overwrite = 0):
    """
    Classify ALL user data, and save the results in new folders
    Also save the number of inactive users found

    Parameters
    ----------
    percentile : int
        the percentile of users that are to be classified as active.
    overwrite : int
        If 0 (default), previously classified data will not be overwritten

    Returns
    -------
    None.
    """
    #From "generate_inactivity_distribution.py" load the inactivity limit
    inactivity_value = load_inactivity_limit("inactivity_limits.csv", str(percentile))
    inactivity_limit = convert_inactive_days_to_timebase(inactivity_value)
    
    #For users in all timebases, classify the user data
    timebase = [0,1,2,3,4]
    inactive_users_list = [0,0,0,0,0]
    timebase_dict = {0 : "Day", 1 : "Week", 2 : "Month", 3 : "Quarter", 4 : "Year"}
    users_classified = 0
    
    for i in timebase:
        #Load all user data within the current timebase
        user_data = load_user_data(i)
        #Completely classify all loaded user data
        classified_data, inactive_users = classify_timebase_user_data(user_data, i, inactivity_limit)
        #Save the number of inactive users found within the timebase
        inactive_users_list[i] = inactive_users
        #Print stats
        print("Classifying group: {} ({} profiles found, {} inactive)".format(timebase_dict[i], len(user_data),inactive_users))
        #Save the classified user data
        save_classified_timebase_data(classified_data, i, overwrite)
        
        users_classified += len(classified_data)
    
    
    #Save the inactive_users_list
    filename = "inactive_users_" + str(int(percentile * 100)) + ".csv"
    print(filename)
    pd.DataFrame(inactive_users_list).to_csv(filename,header=None)
    
    
    print("Users classified: {}".format(users_classified))


def save_classified_timebase_data(classified_data, timebase, overwrite = 0):
    """
    All the classified user is saved within a new folder
    At this point all filenames are changed from usernames to x_timebase.csv (e.g. 1_day.csv)

    Parameters
    ----------
    classified_data : list of arrays
        user data - fully classified.
    timebase : int
        the integer value of the timebase (see top of file).
    overwrite : int
        whether to overwrite the previously saved values

    Returns
    -------
    status_code : int
        has it been successful?
    """
    output_folder_dict = {0 : "classified_data_day", 1 : "classified_data_week", 2 : "classified_data_month", 3 : "classified_data_quarter", 4 : "classified_data_year"}
    output_file_prefix_dict = {0 : "day.csv", 1 : "week.csv", 2 : "month.csv", 3 : "quarter.csv", 4 : "year.csv"}
    
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
    
    for i in range(len(classified_data)):
        #Generate output filename 
        filename = output_folder_dict[timebase] + "/" + str(i) + "_" + output_file_prefix_dict[timebase]
        #Save the file
        pd.DataFrame(classified_data[i]).to_csv(filename,header=None, index=None)
    
    print("        Saved to {}".format(output_folder_dict[timebase]))
    return 0



##########################
"""   Main section    """
#########################
classify_all_user_data(0.95, overwrite = 1)