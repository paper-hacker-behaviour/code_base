# -*- coding: utf-8 -*-
"""
Collect the user data
"""

from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
import time
import pandas as pd
import numpy as np
import re
from datetime import datetime

def create_webdriver(executable_path, headless=True):
    """ Initialise the chome webdriver, allowing for headless operation"""
    options = Options()
    if headless:
        """Options required to run chrome in headless mode"""
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
    
    """ Block images to improve load speeds """
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=options, executable_path=executable_path)
    driver.implicitly_wait(4)
    driver.wait = WebDriverWait(driver, 2)
    
    return driver

def extract_user_data(driver):
    user_data = []
    timeseries = driver.find_elements_by_class_name("c3-event-rect")
    for point in timeseries:
        hover = ActionChains(driver).move_to_element(point)
        hover.perform()
        
        tooltip = driver.find_element_by_class_name("insight-table")
        
        data_point = tooltip.text.split("\n")
        data_point[1] = int(data_point[1].replace("Submissions ", ""))
        user_data.append(data_point)
    
    return user_data

def save_user_data(user_url, user_data, filepath=None):
    username = user_url.replace("https://bugcrowd.com/", "").lower()
    
    if username == None:
        print("Error: No user name found, file not saved (" + user_url +")")
    
    """"The links are saved in the form bc_user_list_DD_MM_YYYY"""
    now = datetime.now()
    date_string = "_" + now.strftime("%d") + "_" + now.strftime("%m") + "_" + now.strftime("%Y")
    if filepath is not None:
        filename = filepath + username + date_string + ".csv"
    else:
        filename = username + date_string + ".csv"
    df = pd.DataFrame(user_data, columns=["Date","Submissions"])
    df.to_csv(filename, sep=',',index=False)

def load_user_urls(filename):
    df = pd.read_csv(filename)
    return df

def restart_driver(driver):
    driver.quit()
    chromedriver_path = "C:\\ChromeDriver\\chromedriver.exe"
    new_driver = create_webdriver(chromedriver_path, headless=True)
    return new_driver

def batch_collection_wrapper(driver, url_df, batch_sz=100, offset=0):
    num_batches = int(url_df.shape[0] / batch_sz) + 1
    
    for batch in range(num_batches - offset):
        print("Starting batch {:3.0f} of {:3.0f}".format(batch, num_batches - offset - 1))
        lower_index = (batch + offset) * batch_sz
        upper_index = (batch + offset + 1) * batch_sz
        iter_df = url_df.iloc[lower_index:upper_index]
        iter_df = iter_df.to_numpy()
        for url in iter_df:
            #Collect and save URL
            url = url[0]  
            print(url)
            driver.get(url)
            data = extract_user_data(driver)
            save_user_data(url, data, "user_data/")
            
        #Need to implement restarts
        driver = restart_driver(driver)

"""
Initiate webdriver
N.B. This requires chromedriver to be downloaded
https://chromedriver.chromium.org/downloads
"""
chromedriver_path = "C:\\ChromeDriver\\chromedriver.exe"
driver = create_webdriver(chromedriver_path, headless=True)

url_df = load_user_urls("bc_user_list_17_01_2021.csv")
batch_collection_wrapper(driver, url_df, offset=0)

driver.quit()