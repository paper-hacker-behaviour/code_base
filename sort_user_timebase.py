# -*- coding: utf-8 -*-

import shutil
import os
import glob
import numpy as np
import pandas as pd


def generate_file_list(filepath):
    """
    Parameters
    ----------
    filepath : string
        location of the directory holding the unsorted user data .csv files

    Returns
    -------
    file_list : python list of strings
        python list containing the filenames of all .csv files in filepath
    """
    file_list = [f for f in glob.glob(filepath + "*.csv")]
    return file_list

def timebase_to_int(timebase):
    """
    Convert the timebase string into a pre-defined
    integer value (see categorise_user_data function)
    
    Parameters
    ----------
    timebase : string
        user timebase as a string.

    Returns
    -------
    retVal : int
        the type of timebase.
    """
    retVal = 0
    
    if("\'" in timebase): # MM\YY
        retVal = 3
    elif("-" in timebase): # Week 1 - Week 2
        retVal = 2
    elif("Q" in timebase): # Q1
        retVal = 4 
    elif(timebase[0] == "2"): #20xx
        retVal = 5
    else:
        retVal = 1 #Day 
    
    return retVal


def categorise_user_data(file_list):
    """
    The data for each user can be categorised by the timestep
    used in the data
    -1 = INVALID
    0 = undefined 
    1 = day
    2 = week
    3 = month
    4 = quarter
    5 = year
    
    Parameters
    ----------
    file_list : python list of strings
        python list containing all filenames to sort.

    Returns
    -------
    timebase_vec : numpy vector
        Array of values representing the timebase of each file.
    """
    timebase_vec = np.zeros(len(file_list), dtype=int)
    
    for i in range(len(file_list)):
        f = file_list[i]
        df = pd.read_csv(f)
        
        #Check for empty user data
        if df.size == 0:
            timebase_vec[i] = -1
            continue
        #df will contain > 0 entries 
        timebase = str(df["Date"][0])
        timebase_vec[i] = timebase_to_int(timebase)
    
    return timebase_vec


def print_summary(timebase_vec, file_list):
    """
    Print out a summary of the timebase vector and
    save to dataframe
    
    Parameters
    ----------
    timebase_vec : numpy vector
        Array of values representing the timebase of each file.
        
    file_list : python list of strings
        python list containing all user filenames.

    Returns
    -------
    None.
    """
    timebase_str = {-1: "No. I:", 0: "No. U:", 1: "No. D:", 2: "No. W:", 3: "No. M:", 4: "No. Q:", 5: "No. Y:"}
    unique, counts = np.unique(timebase_vec, return_counts=True)
    total_proc = sum(counts)
    
    print("------------------\nSummary: User data\n------------------")
    
    count = 0
    for i in unique:
        print("{} {}".format(timebase_str[i], counts[count]))
        count += 1
    
    print("\n{}/{} files succesfully processed. ({:.1f}%)".format(total_proc, len(file_list), total_proc / len(file_list) * 100.0))
    
    d = np.array([unique, counts])
    d = np.transpose(d)
    df = pd.DataFrame(d, columns=["Timebase", "Counts"])
    
    filename = "user_timebase_summary.csv"
    df.to_csv(filename, sep=',',index=False)
    

def sort_files(file_list, timebase_vec, overwrite = 0):
    """
    Sort all files into relevant folders

    Parameters
    ----------
    file_list : python list of strings
        python list containing all user filenames.
    timebase_vec : numpy vector
        Array of values representing the timebase of each file.
    overwrite : integer
        Allows for previous folders to be deleted

    Returns
    -------
    None.
    """
    
    folder_list = ["users_invalid", "users_undefined", "users_day", "users_week", "users_month", "users_quarter", "users_year"]
    copy_dict = {-1: "users_invalid/", 0: "users_undefined/", 1: "users_day/", 2: "users_week/", 3: "users_month/", 4: "users_quarter/", 5: "users_year/"}
    for folder in folder_list:
        try:
            os.mkdir(folder)
        except FileExistsError:
            print("Error, {} already exists".format(folder))
            if overwrite:
                shutil.rmtree(folder)
                print("{} overwritten".format(folder))
                os.mkdir(folder)
            else:
                return -1
    
    for i in range(len(file_list)):
        f = file_list[i]
        shutil.copy2(f, copy_dict[timebase_vec[i]])
   
"""
All user data CSV files should be stored in a single folder
before running this script
"""

filepath = "user_data/"
file_list = generate_file_list(filepath)
timebase_vec = categorise_user_data(file_list)
print_summary(timebase_vec, file_list)
sort_files(file_list,timebase_vec, 1)

