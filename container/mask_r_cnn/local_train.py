#!/usr/bin/env python3

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function
import os
from helper import *
from engine import evaluate
import argparse
from torch.utils.tensorboard import SummaryWriter

def init_args():
    """
    参数初始化
    :return: None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--root_train_data', type=str,
                        help='Directory containing train data')
    parser.add_argument('-m', '--model_type', type=str,
                        help='model type, can be modelb/modelc/modelc_sub')
    parser.add_argument('-n', '--num_epochs', type=int,
                        help='train epoch')
    parser.add_argument('-s', '--save_path', type=str,
                        help='Path to logs and ckpt models')

    return parser.parse_args()

# The function to execute the training.
def train(root_train_data,model_type, num_epochs,save_path):
    #preprocess
    print('Starting preprocessing')
    try:
        preprocess(root_train_data)
    except:
        pass

    print('Starting the training.')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # num_classes = int(trainingParams['num_classes'])
    if model_type=='modelb':
        num_classes = 30 + 1
        dataset = LianbaoDataset(root_train_data, get_transform(train=True))
        dataset_test = LianbaoDataset(root_train_data, get_transform(train=False))
    elif model_type=='modelc':
        num_classes = 50 + 1
        dataset = LianbaoDatasetModelC(root_train_data, get_transform(train=True))
        dataset_test = LianbaoDatasetModelC(root_train_data, get_transform(train=False))
    elif model_type=='modelc_sub':
        num_classes = 5 + 1
        dataset = LianbaoDatasetModelC_sub(root_train_data, get_transform(train=True))
        dataset_test = LianbaoDatasetModelC_sub(root_train_data, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                              collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                        momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=3,
                                           gamma=0.1)

    # let's train it for 10 epochs

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)

    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    # save the model
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'mask_rcnn_model_saved_test'))

    print('Training complete.')

def get_file(x):
    (filepath, tempfilename) = os.path.split(x)
    (filename, extension) = os.path.splitext(tempfilename)
    return filename

def preprocess(root):
    '''preprocess to make sure the label files and img files match'''
    imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
    labels = list(sorted(os.listdir(os.path.join(root, "label"))))
    img_ls = [get_file(i) for i in imgs]
    label_ls = [get_file(i) for i in labels]
    if len(imgs)!=len(labels):
        imgs_idx_to_move = [i for i in img_ls if i not in label_ls]
        labels_idx_to_move = [i for i in label_ls if i not in img_ls]
        imgs_to_move = [imgs[img_ls.index(i)] for i in imgs_idx_to_move]
        labels_to_move = [labels[label_ls.index(i)] for i in labels_idx_to_move]
        #move
        for i in imgs_to_move:
            try:
                os.remove(os.path.join(root,'pic',i))
                print ("successfully removed file : ", os.path.join(root,'pic',i))
            except:
                continue
        for j in labels_to_move:
            try:
                os.remove(os.path.join(root,'label',j))
                print("successfully removed file : ", os.path.join(root,'label',j))
            except:
                continue
    else:
        exit()

if __name__ == '__main__':
    # init args
    args = init_args()
    print('start------------------------------------ train ')
    train(root_train_data=args.root_train_data,
          model_type=args.model_type,
          num_epochs=args.num_epochs,
          save_path=args.save_path)





