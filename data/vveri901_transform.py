import os
import numpy as np

from os import listdir
from os.path import isfile, join

train_file = './train.txt'
test_file = './gallery.txt'
query_file = './query.txt'

output_train_img_path = './bounding_box_train/'
output_test_img_path = './bounding_box_test/'
output_query_img_path = './query/'

def file_parsing(filepath, output_path, output_list=False):
    files = open(filepath, 'r')
    file_list = []
    _file_cnt = 0
    for _file in files:
        _parse = _file.split(' ')
        import pdb
        pdb.set_trace()
        _parsing = _parse[0].split('/')[2:]
        _vid = '%d' % (int(_parsing[0]))
        _cid = _parsing[1].split('_')[1]
        _cid = int(_cid[1:])
        path_from =  './'+_parsing[0]+'/'+_parsing[1].split('\n')[0]
        onlyfiles = [f for f in listdir(path_from) if isfile(join(path_from, f))]
        for _f in onlyfiles:
            org_file = path_from+'/'+_f
            transformed_file = str(_vid).zfill(5) + '_c' + str(_cid).zfill(5) + '_' + str(_file_cnt).zfill(8) + '.png'
            transformed_path = output_path + transformed_file
            _command = 'cp %s %s' % (org_file, transformed_path)
            print(_command)
            os.system(_command)
            file_list.append(transformed_file)
            _file_cnt +=1
    if output_list == True:
        return file_list


def duplication_check(query_list, test_list):
    for _qf in query_list:
        if _qf in test_list:
            print(_qf + ' is duplicated')
            _command = 'rm %s' % (output_query_img_path + query_list)
            print(_command)
            os.system(_command)


if __name__ == '__main__':
    # for Training


    if not os.path.exists(output_train_img_path):
        os.makedirs(output_train_img_path)
        print('make directory - %s'%(output_train_img_path))
    if not os.path.exists(output_test_img_path):
        os.makedirs(output_test_img_path)
        print('make directory - %s'%(output_test_img_path))
    if not os.path.exists(output_query_img_path):
        os.makedirs(output_query_img_path)
        print('make directory - %s'%(output_query_img_path))



    print('Preparing Trainig dataaset for Video-Vehile Re-iD 991 dataset')
    file_parsing(train_file, output_train_img_path)
    # for Test

    print('Preparing large test dataaset for Veri-Wild')
    test_file_list = file_parsing(test_file, output_test_img_path, output_list=True)
    # for Test Query
    print('Preparing large query dataaset for Veri-Wild')
    query_file_list = file_parsing(query_file, output_query_img_path, output_list=True)


    # Duplicated file check
    # duplication_check(query_file_list,test_file_list)
