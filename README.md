# Behavioural analysis of hackers

This anonymised code repository accompanies the working paper concerning the behaviour of hackers on bug bounty platforms. This description will later be updated with a link to the published paper.

The code provided allows individuals to verify the findings of the paper by following the data collection and processing steps used during research - Please refer to the section below for usage instructions.


## Usage

The repository contains 9 Python files that must be sequentially executed in order to replicate the process outlined in the paper. The scripts should be executed in the following order:

1. generate_programme_list
2. generate_user_list
3. collect_user_data (very slow)
4. sort_user_timebase
5. generate_inactivity_distribution
6. classify_user_behaviour
7. estimate_hmm_parameters
8. find_cluster_parameters
9. validation

## Requirements
Compatible with Python 3.6 and 3.8, the following libraries are required:
* Selenium (with chromedriver)
* time
* pandas
* numpy
* re
* datetime
* glob
* shutil
* os
* matplotlib
* sklearn

## License
[MIT](https://choosealicense.com/licenses/mit/)