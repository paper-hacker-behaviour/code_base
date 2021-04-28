# -*- coding: utf-8 -*-
"""
Collect the links to all programmes on Bugcrowd
"""
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
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
    
    driver = webdriver.Chrome(options=options, executable_path=executable_path)
    
    return driver


def load_all_list_elements(driver):
    """Automatic script to click on the 'Load more results' button"""
    loading = 1
    while loading:
        time.sleep(3.5)
        try:
            driver.find_element_by_class_name("rp-program-list__load-more__btn.bc-btn.bc-btn--small.bc-btn--tertiary").click()
        except:
            loading = 0

def remove_unwanted_pages(page_list):
    """The programme page contains many (upwards of 80) pages that do not contain information"""
    """Remove all the 'teaser' pages"""
    reduced_list = [x for x in page_list if not "programs/teasers" in x]
    
    """Remove any internal bugcrowd pages"""
    reduced_list = [x for x in reduced_list if not "www." in x]
    return reduced_list

def extract_programme_links(driver):
    """For every programme, the hyperlink to the main page is contained within the panel__title"""
    content = driver.find_elements_by_class_name("bc-panel__title")
    link_postfix = [re.search(r'\"(.*?)\"', x.get_attribute('innerHTML')).group(1) for x in content]
    link_postfix = remove_unwanted_pages(link_postfix)
    """"All links are put into the form https://bucrowd.com/XXXX"""
    links = ["https://bugcrowd.com" + x for x in link_postfix]
    return links

def save_data(url_list, filepath=None):
    """"The links are saved in the form bc_prog_list_DD_MM_YYYY"""
    now = datetime.now()
    date_string = now.strftime("%d") + "_" + now.strftime("%m") + "_" + now.strftime("%Y")
    if filepath is not None:
        filename = filepath + "bc_prog_list_" + date_string + ".csv"
    else:
        filename = "bc_prog_list_" + date_string + ".csv"
    df = pd.DataFrame(url_list, columns=["URL"])
    df.to_csv(filename, sep=',',index=False)




"""
URL of the Bugcrowd programme page
"""
base_url = "https://bugcrowd.com/programs"

"""
Initiate webdriver
N.B. This requires chromedriver to be downloaded
https://chromedriver.chromium.org/downloads
"""
chromedriver_path = "C:\\ChromeDriver\\chromedriver.exe"

driver = create_webdriver(chromedriver_path, headless=True)
driver.get(base_url)
load_all_list_elements(driver)
programme_links = extract_programme_links(driver)
save_data(programme_links)
driver.quit()


