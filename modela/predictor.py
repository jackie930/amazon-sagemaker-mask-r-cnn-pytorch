# -*- coding: utf-8 -*-
import sys
import os
import argparse
import logging
import warnings
import io
from torchvision import datasets, models, transforms
import json
import boto3
import torch.nn as nn

import warnings
import numpy as np

from PIL import Image
import itertools
import cv2
import skimage.io
import torch
import GPUtil

warnings.filterwarnings("ignore", category=FutureWarning)

import sys

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import torchvision
import PIL
import codecs

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import flask

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    # parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

    data = flask.request.data.decode('utf-8')
    data = json.loads(data)

    bucket = data['bucket']
    image_uri = data['image_uri']

    download_file_name = image_uri.split('/')[-1]
    print ("<<<<download_file_name ", download_file_name)
    #download_file_name = './test.jpg'
    s3_client.download_file(bucket, image_uri, download_file_name)
    print('Download finished!')
    # inference and send result to RDS and SQS

    print('Start to inference:')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model.load_state_dict(copy.deepcopy(torch.load("./model.pth", device)))
    model = model.to(device)
    model.eval()

    trans2 = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = PIL.Image.open(download_file_name)
    t1 = trans2(image)
    t1 = t1.to(device, dtype=torch.float)

    t2 = t1.reshape((1, 3, 224, 224))
    output = model(t2)
    idx = torch.argmax(output)

    classes = classes = ['否', '是', '零件']
    label = classes [idx]

    # output is a list of dict, containing the postprocessed predictions
    result = {
        'label': label
    }

    print ('<<<< result: ', result)

    inference_result = {
        'result': result
    }
    _payload = json.dumps(inference_result, ensure_ascii=False)
    #show gpu utli
    GPUtil.showUtilization()
    #release gpu memory
    torch.cuda.empty_cache()
    GPUtil.showUtilization()


    return flask.Response(response=_payload, status=200, mimetype='application/json')