import argparse
import os
from Relevance_Batched_updated import *

years = list(range(2020,2024)) # the range of years to be tested
issue = 'sexuality'

dir_path = os.path.dirname(os.path.realpath(__file__))+"/Output/{}/".format(issue)
file_list = []
for i in os.listdir(dir_path):
    for j in years:
        if str(j) in i and "lang_filtered.csv" in i:
            file_list.append(i)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--array', type = int)
    args = argparser.parse_args()

relevance_filtering(file_=dir_path+file_list[args.array])