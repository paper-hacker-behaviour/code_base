# -*- coding: utf-8 -*-
"""
Calculate the inactivity distributions for all users
"""

import glob
import numpy as np
import pandas as pd


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


def calculate_leading_intervals(timebase_list):
    """
    Find all leading intervals of inactivity in the user data
    A leading interval is the period of time before a user makes their first submission

    Parameters
    ----------
    timebase_list : list of numpy arrays
        A list containing all user data from a single timebase category.

    Returns
    -------
    leading_intervals : numpy array
        The number of inactive leading data points for each user.

    """
    leading_intervals = np.zeros(len(timebase_list), dtype=int)
    
    for i in range(len(timebase_list)):
        inactive_intervals = 0
        start_found = 0
        
        #Loop over the data for an individual user to find the start of the data
        for j in range(len(timebase_list[i])):
            val = timebase_list[i][j]
            if start_found == 0:
                if val == 0:
                    inactive_intervals += 1
                else:
                    start_found = 1
                    break
        
        #Save the number of inactive leading intervals
        leading_intervals[i] = inactive_intervals
    
    return leading_intervals


def calculate_lagging_intervals(timebase_list):
    """
    Find all lagging intervals of inactivity in the user data
    A lagging interval is the period of time after their last recorded submission
    
    Parameters
    ----------
    timebase_list : list of numpy arrays
        A list containing all user data from a single timebase category.

    Returns
    -------
    lagging_intervals : numpy array
        The number of inactive lagging data points for each user.

    """
    lagging_intervals = np.zeros(len(timebase_list), dtype=int)
    
    for i in range(len(timebase_list)):
        inactive_intervals = 0
        end_found = 0
        
        #Loop IN REVERSE over the data for an individual user to find the end of the data
        for j in reversed(range(len(timebase_list[i]))):
            val = timebase_list[i][j]
            if end_found == 0:
                if val == 0:
                    inactive_intervals += 1
                else:
                    end_found = 1
                    break
        
        #Save the number of inactive leading intervals
        lagging_intervals[i] = inactive_intervals
    
    return lagging_intervals


def calculate_internal_intervals(timebase_list):
    """
    Compute a list of all internal intervals within the user timeseries
    Finds the interval of time between two adjacent submissions, for all submissions
    
    Leading and lagging intervals are NOT counted in this function

    Parameters
    ----------
    timebase_list : list of numpy arrays
        A list containing all user data from a single timebase category.

    Returns
    -------
    internal_intervals : list of lists
        For each user, a list of internal intervals is computed

    """
    
    internal_intervals = [None]*len(timebase_list)
    
    for i in range(len(timebase_list)):
        user_internal_intervals = []
        current_run = 0 #Number of consecutive inactive intervals e.g. 1,0,0 => 2
        first_value = 1
        for j in range(len(timebase_list[i])):
            val = timebase_list[i][j]
            if val == 0:
                current_run += 1
            else:
                if first_value == 0: #Do not record the leading interval
                    user_internal_intervals.append(current_run)
                
                current_run = 0
                first_value = 0
                
        internal_intervals[i] = user_internal_intervals
    
    return internal_intervals


def generate_inactivity_list(timebase_list):
    """
    Convert all user data within a timebase into inactivity data (leading, internal, lagging)
    The timeseries data for a user is converted into (leading)(internal)(lagging) intervals of time

    Parameters
    ----------
    timebase_list : list of numpy arrays
        A list containing all user data from a single timebase category.

    Returns
    -------
    inactivity_list : list of numpy arrays
        inactivity_list[0] - all leading intervals of inactivity
        inactivity_list[1] - all internal intervals of inactivity, N.b. list of lists of integers
        inactivity_list[2] - all lagging intervals of inactivity

    """
    inactivity_list = [None]*3
    inactivity_list[0] = calculate_leading_intervals(timebase_list)
    inactivity_list[1] = calculate_internal_intervals(timebase_list)
    inactivity_list[2] = calculate_lagging_intervals(timebase_list)
    
    return inactivity_list


def generate_internal_inactivity_distribution(inactivity_list):
    """
    From all interal interval lengths within a timebase, 
    the occurances are counted

    Parameters
    ----------
    inactivity_list : list of lists 
        For each user, a list of internal intervals is computed.

    Returns
    -------
    interval_distribution : list
        How frequently each internal interval length has occured.
    """
    combined_list = []
    
    #Combine all lists together into a single list of intervals
    for user in inactivity_list:
        if user is not None:
            combined_list = combined_list + user
    
    combined_list = sorted(combined_list)
    interval_distribution = np.zeros(max(combined_list) + 1)
    
    #Count occurances of each interval value
    for i in combined_list:
        interval_distribution[i] += 1
    
    return interval_distribution


def interval_distribution_to_CDF(interval_distribution):
    """
    Convert the occurance counter to a CDF

    Parameters
    ----------
    interval_distribution : numpy array
        How frequently each internal interval length has occured.

    Returns
    -------
    cumulative_distribution : numpy array
        CDF of interval occurance.

    """
    #Normalise
    cumulative_distribution = interval_distribution / sum(interval_distribution)
    
    #Convert to PDF to CDF
    for i in range(1,len(cumulative_distribution)):
        cumulative_distribution[i] += cumulative_distribution[i-1]
    
    return cumulative_distribution

def resample_interval_distribution(src_interval_dist, dst_interval_CDF, timebase_ratio):
    """
    Using the dst_interval_CDF as a prior distribution, 
    this function resamples intervals in a longer timebase 
    into a shorter one. E.g. Years -> Quarters
    
    Where the dst_interval_CDF cannot be used as a prior, 
    a uniform prior is used instead

    Parameters
    ----------
    src_interval_dist : numpy array
        For the source timebase, a list of interval occurances.
    dst_interval_CDF : numpy array
        From the destination timebase, the CDF of internal intervals.
    timebase_ratio : int
        The ratio of the destination timebase to the source
        E.g. 4 quarters to 1 year

    Returns
    -------
    rs_interval_dist : numpy array
        src_interval_dist resampled to the dst timebase.

    """
    #The resampled array will be longer than the original (by * timebase_ratio)
    rs_interval_dist = np.zeros(len(src_interval_dist) * timebase_ratio, dtype=int)
    
    #The CDF can only be used in multiples of timebase_ratio
    #Cut CDF to the nearest multiple
    dst_interval_CDF = dst_interval_CDF[:-(len(dst_interval_CDF)%timebase_ratio)]
    complete_intervals = int(len(dst_interval_CDF) / timebase_ratio) #Number of intervals to use CDF prior 
    
    #There will be a very slight additional error in the resampled distribution 
    #error = 1.0 - max(dst_interval_CDF)
    
    #Distribute the counts according to the prior distribution
    lower_p = 0.0
    lower_index = 0
    upper_index = timebase_ratio - 1
    CDF_subset = np.zeros(4)
    for i in range(len(src_interval_dist)):
        
        #Occurances to distribute
        counts = int(src_interval_dist[i])
        if counts == 0:
            continue #If none, skip to next iteration
        
        lower_index = i * timebase_ratio
        upper_index = (i+1) * timebase_ratio
        
        if i < complete_intervals:
            #CDF prior
            if lower_index > 0:
                lower_p = dst_interval_CDF[lower_index - 1] #Find lower bound
                
            CDF_subset = dst_interval_CDF[lower_index:upper_index]
            
            #Normalise to new CDF
            divisor = (max(CDF_subset) - lower_p)
            CDF_subset = (CDF_subset - lower_p) / (divisor)
            
            #Random vector [0.0,1.0] to use with CDF
            random_assign = np.random.uniform(size=counts)
            
            #Distribute using CDF_subset
            for j in random_assign:
                index = 0
                while(j > CDF_subset[index]):
                    index += 1
                    
                rs_interval_dist[lower_index + index] += 1
            
        else:
            #Uniform prior
            random_assign = np.random.randint(timebase_ratio, size=counts) #Random vector with values 0,...,(timebase_ratio - 1)
            
            #Distribute
            for j in random_assign:
                rs_interval_dist[lower_index + j] += 1
                
    return rs_interval_dist


def resample_all_intervals(interval_count_list, interval_CDF_list):
    """
    Used to resample and combine all internal intervals to the DAY timebase
    This only performs a single iteration
    
    timebase index
        0 = day
        1 = week
        2 = month
        3 = quarter
        4 = year

    Parameters
    ----------
    interval_count_list : list of numpy arrays
        For each timebase, each array contains the interval occurances.
    interval_CDF_list : list of numpy arrays
        For each timebase, each array contains the CDF generated from the interval_count.

    Returns
    -------
    resampled_counts : numpy array
        Array containing the internal intervals resamples to the DAY timebase.
    """
    
    #The ratio of all adjacent timebases
    timebase_ratio_list = [7,4,3,4] #Days in a week, weeks in a month, months in a quarter, quarters in a year
        
    #Initially want to resample the YEARS timebase
    resampled_counts = interval_count_list[4]
    for i in reversed(range(len(timebase_ratio_list))):
        resampled_counts = resample_interval_distribution(resampled_counts, interval_CDF_list[i], timebase_ratio_list[i])
        
        #Combine resampled counts with underlying (e.g. years_to_quarters and quarters)
        for j in range(len(interval_count_list[i])):
            resampled_counts[j] += interval_count_list[i][j]
    
    
    return resampled_counts


def generate_data_lists():
    """
    Wrapper function used to load all user data and 
    convert in internal interval counts and CDF lists

    Returns
    -------
    interval_count_list : list of numpy arrays
        For each timebase, each array contains the interval occurances.
    interval_CDF_list : list of numpy arrays
        For each timebase, each array contains the CDF generated from the interval_count.

    """
    interval_count_list = []
    interval_CDF_list = []
    
    for i in range(5):
        #Load all user data in the same timebase
        user_data = load_user_data(i)
        #Generate the inactivity lists
        inactivity_list = generate_inactivity_list(user_data)
        #Count internal intervals
        internal_interval_count = generate_internal_inactivity_distribution(inactivity_list[1])
        #Generate the CDF for internal intervals
        internal_interval_CDF = interval_distribution_to_CDF(internal_interval_count)
        #Append to the overall lists
        interval_count_list.append(internal_interval_count)
        interval_CDF_list.append(internal_interval_CDF)
    
    return interval_count_list, interval_CDF_list


def find_percentile(percentile, CDF):
    """
    Find the x percentile from a CDF

    Parameters
    ----------
    percentile : float
        The desired value of x.
    CDF : numpy array
        The CDF

    Returns
    -------
    index : int
        (Nearest) position within the CDF of the percentile.
    """
    index = 0
    for i in CDF:
        if i <= percentile:
            index += 1
        else:
            return index
    return index


def find_activity_limits(repeats, percentile, interval_count_list, interval_CDF_list):
    """
    Use a Monte Carlo simulation to find the x percentile of user inactivty.
    This is used to set a limit after which a user can be declared inactive.

    Parameters
    ----------
    repeats : int
        Number of simulation iterations.
    percentile : float
        The x percentile of the inactivity CDF (e.g. 0.95)
    interval_count_list : list of numpy arrays
        For each timebase, each array contains the interval occurances.
    interval_CDF_list : list of numpy arrays
        For each timebase, each array contains the CDF generated from the interval_count.

    Returns
    -------
    stats : list of floats
        stats[0] : mean - 2std
        stats[1] : mean 
        stats[2] : mean + 2std
        stats[3] : std    (N.b. redundant)
    """
    stats = np.zeros(4)
    
    percentile_day_list = np.zeros(repeats)
    
    #Perform simulation
    for i in range(repeats):
        
        #Print progress out to console
        if i % int(repeats / 10) == 0:
            print("        Iteration {} of {}".format(i, repeats))
    
        resampled_counts = resample_all_intervals(interval_count_list, interval_CDF_list)
        resampled_CDF = interval_distribution_to_CDF(resampled_counts)
        percentile_day_list[i] = find_percentile(percentile, resampled_CDF)
    
    
    stats[0] = np.mean(percentile_day_list) - 1.96 * np.std(percentile_day_list)
    stats[1] = np.mean(percentile_day_list)
    stats[2] = np.mean(percentile_day_list) + 1.96 * np.std(percentile_day_list)
    stats[3] = np.std(percentile_day_list)
    
    return stats


def grid_search_activity_limits(repeats, percentile_list):
    """
    Performs a simulation using each of the values specified in the percentile_list.
    This is used to find the activity limits of users, e.g. 90% of users, 95% etc.
    The results (a range in days) are placed in a dataframe

    Parameters
    ----------
    repeats : int
        Number of simulation iterations.
    percentile_list : list of floats
        List of - The x percentile of the inactivity CDF (e.g. 0.95).

    Returns
    -------
    df : pandas dataframe
        Containing the statistics from each simulation.
    """
    #Load in all the user data
    interval_count_list, interval_CDF_list = generate_data_lists()
    
    data = []
    for p in percentile_list:
        print("Finding limit for {:2.3f}% of users".format(p * 100.0))
        stats = find_activity_limits(repeats, p, interval_count_list, interval_CDF_list)
        data.append(stats)
    
    data = np.transpose(data)
    df = pd.DataFrame(data, columns=percentile_list)
    return df
    




##########################
"""   Main section    """
#########################
percentile_list = np.linspace(0.80,0.99, num=20)
repeats = 1000
filename = "inactivity_limits.csv"
df = grid_search_activity_limits(repeats, percentile_list)
df.to_csv(filename)
