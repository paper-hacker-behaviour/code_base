# -*- coding: utf-8 -*-
"""
Collect the links to all public previously-active users on bugcrowd
"""

from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import time
import pandas as pd
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

def load_programme_urls(filename):
    df = pd.read_csv(filename)
    return df

def generate_hof_urls(df):
    """"Append /hall-of-fame to each url"""
    df["URL"] = df["URL"] + "/hall-of-fame"


def extract_hof_data(driver):
    element_username = driver.find_elements_by_class_name("bc-userblock")
    element_points = driver.find_elements_by_class_name("bc-table__cell-right.rp-leaderboard__points")
    
    username = [x.text for x in element_username]
    points = [int(x.text) for x in element_points]
    
    urls = [re.search(r'href=\"(.*?)\"', x.get_attribute('outerHTML')).group(1) if re.search(r'href=\"(.*?)\"', x.get_attribute('outerHTML')) else "" for x in element_username]
    
    data = {'usernames':username,'points':points, 'urls':urls}
    df = pd.DataFrame(data)
    return df

def compute_programme_statistics(hof_df):
    total_users = hof_df.shape[0]
    
    private_users = [1 if "Private user" in x else 0 for x in hof_df["usernames"]]
    private_users = sum(private_users)
    public_users = total_users - private_users
    
    return [total_users, private_users, public_users, hof_df["points"].sum(), hof_df["points"].mean(), hof_df["points"].std()]


def collect_hof_data(df, driver):
    ret_df = pd.DataFrame()
    ret_df["URL"] = df["URL"]
    ret_df["total"] = 0
    ret_df["private"] = 0
    ret_df["public"] = 0
    ret_df["points"] = 0
    ret_df["mean"] = 0.0
    ret_df["std"] = 0.0
    
    url_df = pd.DataFrame()
    
    
    for i in range(df.shape[0]):
        driver.get(df.iloc[i,0])
        time.sleep(1.0)
        print("{:3.0f}/{:3.0f}".format(i, df.shape[0]))
        try:
            """"See if empty page"""
            driver.find_element_by_class_name("bc-blankstate__title")
        except:
            """Collect HOF data"""
            hof_df = extract_hof_data(driver)
            rv = compute_programme_statistics(hof_df)
            ret_df.iloc[i,1:7] = rv
            
            url_df = pd.concat([url_df,hof_df["urls"]], ignore_index=True, axis=1)
    
    return ret_df, url_df

def process_user_list(df):
    """Rearranges the data and removes duplicates"""
    url_df_stack = df.stack()
    url_df_stack.replace("", float("NaN"), inplace=True)
    url_df_stack = url_df_stack.dropna()
    url_df_stack = url_df_stack.drop_duplicates()
    
    return "https://bugcrowd.com" + url_df_stack

def save_url_data(url_list, filepath=None):
    """"The links are saved in the form bc_user_list_DD_MM_YYYY"""
    now = datetime.now()
    date_string = now.strftime("%d") + "_" + now.strftime("%m") + "_" + now.strftime("%Y")
    if filepath is not None:
        filename = filepath + "bc_user_list_" + date_string + ".csv"
    else:
        filename = "bc_user_list_" + date_string + ".csv"
    df = pd.DataFrame(url_list, columns=["URL"])
    df.to_csv(filename, sep=',',index=False)
    
def save_programme_data(df, filepath=None):
    """"The data is saved in the form bc_prog_data_DD_MM_YYYY"""
    now = datetime.now()
    date_string = now.strftime("%d") + "_" + now.strftime("%m") + "_" + now.strftime("%Y")
    if filepath is not None:
        filename = filepath + "bc_prog_data_" + date_string + ".csv"
    else:
        filename = "bc_prog_data_" + date_string + ".pkl"
    df.to_pickle(filename)

"""
Initiate webdriver
N.B. This requires chromedriver to be downloaded
https://chromedriver.chromium.org/downloads
"""
chromedriver_path = "C:\\ChromeDriver\\chromedriver.exe"
driver = create_webdriver(chromedriver_path, headless=True)

""" Load programme urls from CSV and generate hof page url """
df = load_programme_urls("bc_prog_list_17_01_2021.csv")
generate_hof_urls(df)

""" Scrape all hall-of-frame data """
ret_df, url_df = collect_hof_data(df, driver)

""" Find the unique urls of all users and save """
unique_user_url_df = process_user_list(url_df)
save_url_data(unique_user_url_df)

"""" Save the programme data for later analysis """
save_programme_data(ret_df)

driver.quit()
