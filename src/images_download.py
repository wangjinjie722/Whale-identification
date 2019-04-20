import ssl
import json
import tqdm
import time
import urllib.request
from multiprocessing import Pool


ssl._create_default_https_context = ssl._create_unverified_context

training_raw = '../data/raw/imaterialist-product-2019/train.json'
training_repo = '../data/training_set'


class Load_and_download:
    def __init__(self, json_str, repo):
        self.json_str = json_str
        # self.raw = raw
        # self.repo = repo

        print('gotch u')

    def load(self):
        '''
        load dict from json
        '''
        with open(self.json_str, "rb") as file:
            self.load_dict = json.load(file)
        return self.load_dict

    @staticmethod
    def download(case_repo):
        '''
        download images
        input: case_repo, zip of case(dict) and repo(list)
        '''
        case, repo = case_repo[0], case_repo[1]

        try:
            urllib.request.urlretrieve(case['url'], repo + '/' + case['id'])
        except:
            print(' \n404: {}\n'.format(case))


if __name__ == '__main__':

    training = Load_and_download(training_raw, training_repo)
    training_dict = training.load()

    # perpare for download
    case_input = training_dict['images']
    length = len(case_input)
    repo_input = [training_repo for i in range(length)]

    # download using multiprocessing
    pool = Pool()
    print('downloading')
    t_1 = time.clock()
    # pool.map(load_and_download.download, zip(case_input, repo_input))
    # pool.close()
    # pool.join()
    for _ in tqdm.tqdm(pool.imap_unordered(training.download, zip(case_input, repo_input)), total=length):
        pass

    pool.close()
    pool.join()
    print('elapsed time: ', time.clock() - t_1)
    print('download complete')
