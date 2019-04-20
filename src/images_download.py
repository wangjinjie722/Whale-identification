import json
import urllib
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

        file = open(self.json_str, "rb")
        self.load_dict = json.load(file)
        return self.load_dict

    def download(self, case):

        try:
            urllib.request.urlretrieve(case['url'], training_repo + '/' + case['id'])
        except:
            print('404')



if __name__ == '__main__':

    training = load_and_download(training_raw, training_repo)
    training_dict = training.load()

    input = training_dict['images']
    t_use = time.perf_counter()
    # pool = Pool()
    # print(type(input))
    # pool.map(load_and_download.download, input)
    # pool.close()
    # pool.join()
    i = 0
    for case in input:
        try:
            urllib.request.urlretrieve(case['url'], training_repo + '/' + case['id'])
        except:
            print('404')
        print(i)
        i += 1
    print('runtime = ', t_use)



