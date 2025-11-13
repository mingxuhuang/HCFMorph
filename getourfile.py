import json
import os
import random

from itertools import combinations

def generate_pairs(original_list):
    return list(combinations(original_list, 2))
def generate_pairs_one(original_list):
    return [(original_list[0], element) for element in original_list[1:]]

def getpath(name = 'train'):
    imglist = set(os.listdir(r'D:\mx\CT_Re\data\CT_resize'))
    with open(r"E:\mx\multi_Re\data.json",'r',encoding='utf-8') as load_f:
        load_dict = json.load(load_f)

    alljson = set(load_dict['train']+load_dict['val']+load_dict['test'])
    isuba = list(imglist-alljson)
    asubi = list(alljson-imglist)

    for value in asubi:
        if value in load_dict['train']:
            load_dict['train'].remove(value)
        elif value in load_dict['val']:
            load_dict['val'].remove(value)
        else:
            load_dict['test'].remove(value)

    load_dict['train'].extend(isuba)

    allfile = []
    for i in load_dict[name]:
        allfile.append(i)
    return allfile

def getfile(filedir,mode = 'train',sample = None):
    path = r'D:\mx\CT_Re\data\CT_resize'
    # a = 0
    filepath = []
    for file in filedir:
        filename = [i for i in os.listdir(os.path.join(path,file)) if 'CT' in i]
        filename.sort(key=lambda x: int(x[3]))
        filename = [os.path.join(path,file,j) for j in filename]
        # a = a + len(filename)
        if mode == 'train':
            list_pair = generate_pairs(filename)
            if sample is not None:
                random.shuffle(list_pair)
                if len(list_pair) > sample:
                    list_pair = random.sample(list_pair, k=sample)
        else:
            list_pair = generate_pairs_one(filename)

        filepath.extend(list_pair)
    # print(a)
    return filepath
# if __name__ == '__main__':
#     filedir = getpath('val')
#     aa = getfile(filedir,'val')
#     a = 0
def getdirfile(mode = 'train',sample=None):
    filedir = getpath(mode)
    file = getfile(filedir,mode,sample)
    # ff = []
    # for c in file:
    #     ff.append(c[0].split('\\')[-2])
    # ff = set(ff)-set(filedir)
    return file


# print(len(getdirfile(sample=5)))
# print(len(getdirfile('val')))
# print(len(getdirfile('test')))