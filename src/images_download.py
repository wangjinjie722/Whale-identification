import json
import urllib.request
from multiprocessing import Pool
import time


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

training_raw = '../data/raw/imaterialist-product-2019/train.json'
training_repo = '../data/training_set'

class load_and_download:
    def __init__(self, json_str, repo):
        self.json_str = json_str
        #self.raw = raw
        self.repo = repo

        print('gotch u')

    def load(self):
        
        with open(self.json_str, "rb") as file:
            self.load_dict = json.load(file)
        return self.load_dict

    @staticmethod
    def download(case_repo):

        case, repo = case_repo[0], case_repo[1]

        try:
            urllib.request.urlretrieve(case['url'], repo + '/' + case['id'])
        except:
            print('404')



if __name__ == '__main__':

    training = load_and_download(training_raw, training_repo)
    training_dict = training.load()

    case_input = training_dict['images']
    repo_input = [training_repo for i in range(len(case_input))]
    # pool = Pool()
    # print(type(input))
    # pool.map(load_and_download.download, input)
    # pool.close()
    # pool.join()
    count = 0
    for case_repo in zip(case_input,repo_input):
        training.download(case_repo)
        
        if count > 5:
            break



