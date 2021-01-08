# -*- coding: utf-8 -*-
import json
import sys
import time
import os
from progressbar import ProgressBar
from shutil import copyfile


def make_dirs(target_folder):
    for j in ['train','val']:
        for i in ['是-有车损','是-没有车损','否-有车损','否-没有车损','零件-有车损','零件-没有车损']:
            folder = os.path.join(target_folder,j,i)
            os.makedirs(folder, exist_ok=True)

def prepare(labelfile,target_folder):
    with open(labelfile, "r") as f:
        annotations = json.load(f, encoding='unicode-escape')

    annotations_values = list(annotations.values())  # don't need the dict keys
    files = list(annotations.keys())
    annotations_all = [a['file_attributes'] for a in annotations_values if a['file_attributes']]
    #make sure the file length is the same as annotations
    assert (len(files)==len(annotations_all))

    #a['是', '否', '零件']
    #b['有车损', '没有车损']
    pbar = ProgressBar().start()
    total = len(files)
    base_path = '/Users/liujunyi/Desktop/spottag/customer/联保/data'
    make_dirs(target_folder)

    split = int(total*0.7)

    for i in range(total):
        org_file_path = os.path.join(base_path,'/'.join(files[i].split('/')[3:]))
        file_name = '--'.join(files[i].split('/')[4:])
        if i<split:
            flag = 'train'
        else:
            flag = 'val'

        if annotations_all[i]['A-车']=='是' and annotations_all[i]['B-损']=='有车损':
            copyfile(org_file_path, os.path.join(target_folder,flag,'是-有车损',file_name))
        elif annotations_all[i]['A-车']=='是' and annotations_all[i]['B-损']=='没有车损':
            copyfile(org_file_path, os.path.join(target_folder,flag,'是-没有车损',file_name))
        elif annotations_all[i]['A-车']=='否' and annotations_all[i]['B-损']=='有车损':
            copyfile(org_file_path, os.path.join(target_folder,flag,'否-有车损',file_name))
        elif annotations_all[i]['A-车']=='否' and annotations_all[i]['B-损']=='没有车损':
            copyfile(org_file_path, os.path.join(target_folder,flag,'否-没有车损',file_name))
        elif annotations_all[i]['A-车']=='零件' and annotations_all[i]['B-损']=='有车损':
            copyfile(org_file_path, os.path.join(target_folder,flag,'零件-有车损',file_name))
        elif annotations_all[i]['A-车']=='零件' and annotations_all[i]['B-损']=='没有车损':
            copyfile(org_file_path, os.path.join(target_folder,flag,'零件-没有车损',file_name))
        else:
            print (annotations_all[i])
        pbar.update(int((i / (total - 1)) * 100))

    pbar.finish()


if __name__ == '__main__':
    print (sys.getdefaultencoding())
    target_folder = './res'
    prepare('./test.json',target_folder)
    #make_dirs(target_folder)
