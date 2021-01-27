# -*- coding: utf-8 -*-
import json
import sys
import time
import os
from progressbar import ProgressBar
from shutil import copyfile


def make_dirs(target_folder):
    for i in ['label','pic']:
        folder = os.path.join(target_folder,i)
        os.makedirs(folder, exist_ok=True)

def prepare(labelfile,target_folder):
    with open(labelfile, "r") as f:
        annotations = json.load(f, encoding='unicode-escape')

    annotations_values = list(annotations.values())[1]  # don't need the dict keys

    #print (annotations_values)
    print (len(annotations_values))
    key_ls = list(annotations_values.keys())
    make_dirs(target_folder)
    base_path = '/Users/liujunyi/Desktop/spottag/customer/联保/data'

    for i in range(len(annotations_values)):
        try:
            if len(annotations_values[key_ls[i]]['regions'])>0:
                print (annotations_values[key_ls[i]])
                file_name = annotations_values[key_ls[i]]['filename']
                target_file_name = 'data'+str(i)+'.jpg'
                org_file_path = os.path.join(base_path, file_name)
                copyfile(org_file_path, os.path.join(target_folder, 'pic', target_file_name))

                json_name = os.path.join(target_folder,'label','data'+str(i)+'.json')
                with open(json_name, 'w') as outfile:
                    json.dump(annotations_values[key_ls[i]], outfile)
        except:
            continue

def prepare_modelc(labelfile,target_folder):
    with open(labelfile, "r") as f:
        annotations = json.load(f, encoding='unicode-escape')
    annotations_values = list(annotations.values()) # don't need the dict keys
    print (annotations_values[0])
    print (len(annotations_values))
    (filepath, tempfilename) = os.path.split(labelfile)
    print ("<<< file path: ", filepath)
    make_dirs(target_folder)

    for i in range(len(annotations_values)):
        try:
            if len(annotations_values[i]['regions'])>0:
                tempfilename = annotations_values[i]['filename']
                (filename, extension) = os.path.splitext(tempfilename)

                json_name = os.path.join(target_folder,'label',filename+'.json')
                with open(json_name, 'w') as outfile:
                    json.dump(annotations_values[i], outfile)
        except:
            continue

def prepare_modelc(labelfile,target_folder):
    with open(labelfile, "r") as f:
        annotations = json.load(f, encoding='unicode-escape')
    annotations_values = list(annotations.values()) # don't need the dict keys
    print (annotations_values[0])
    print (len(annotations_values))
    (filepath, tempfilename) = os.path.split(labelfile)
    print ("<<< file path: ", filepath)
    make_dirs(target_folder)

    for i in range(len(annotations_values)):
        try:
            if len(annotations_values[i]['regions'])>0:
                tempfilename = annotations_values[i]['filename']
                (filename, extension) = os.path.splitext(tempfilename)

                json_name = os.path.join(target_folder,'label',filename+'.json')
                with open(json_name, 'w') as outfile:
                    json.dump(annotations_values[i], outfile)
        except:
            continue

def prepare_modelb(labelfile,target_folder):
    with open(labelfile, "r") as f:
        annotations = json.load(f, encoding='unicode-escape')
    annotations_values = list(annotations.values()) # don't need the dict keys
    print (annotations_values[0])
    print (len(annotations_values))
    #(filepath, tempfilename) = os.path.split(labelfile)
    #print ("<<< file path: ", filepath)
    make_dirs(target_folder)

    for i in range(len(annotations_values)):
        try:
            if len(annotations_values[i]['regions'])>0:
                labelfile = annotations_values[i]['filename']
                (filepath, tempfilename) = os.path.split(labelfile)
                (filename, extension) = os.path.splitext(tempfilename)
                json_name = os.path.join(target_folder,'label',filename+'.json')
                with open(json_name, 'w') as outfile:
                    json.dump(annotations_values[i], outfile)
        except:
            continue

if __name__ == '__main__':
    print (sys.getdefaultencoding())
    target_folder = './res'
    #origin 1000
    #prepare('/Users/liujunyi/Desktop/spottag/customer/联保/1000b-tag.json',target_folder)
    #modelb
    prepare_modelb('/Users/liujunyi/Desktop/spottag/customer/联保/label-2/20210125_model_b_dataset/new_label_output.json',
                   '/Users/liujunyi/Desktop/spottag/customer/联保/label-2/20210125_model_b_dataset')

    #model c
    #prepare_modelc('/Users/liujunyi/Desktop/spottag/customer/联保/label-2/20210125_image_segmentation/1.json','/Users/liujunyi/Desktop/spottag/customer/联保/label-2/20210125_image_segmentation')
    #make_dirs(target_folder)
