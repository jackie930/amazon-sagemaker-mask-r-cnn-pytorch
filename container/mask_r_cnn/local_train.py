#!/usr/bin/env python3

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function
from helper import *
from engine import evaluate

# The function to execute the training.
def train(root_train_data):
    print('Starting the training.')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # num_classes = int(trainingParams['num_classes'])
    num_classes = 30 + 1
    dataset = LianbaoDataset(root_train_data, get_transform(train=True))
    dataset_test = LianbaoDataset(root_train_data, get_transform(train=False))

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
    optimizer = torch.optim.SGD(params, lr=0.005,
                        momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=3,
                                           gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 2

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
    torch.save(model.state_dict(), os.path.join('./save_model', 'mask_rcnn_model_saved'))

    print('Training complete.')

if __name__ == '__main__':
    train('../../modelb')


