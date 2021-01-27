import os
import json
import pickle
import sys
import traceback
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data
import torchvision
import transforms as T
from engine import train_one_epoch
import utils
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class LianbaoDataset(torch.utils.data.Dataset):
    # assume the each contains one label file for one pic, inside two folders with same name
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "label"))))
        if '.ipynb_checkpoints' in self.imgs:
            self.imgs.remove('.ipynb_checkpoints')
        if '.ipynb_checkpoints' in self.labels:
            self.labels.remove('.ipynb_checkpoints')

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "pic", self.imgs[idx])
        label_path = os.path.join(self.root, "label", self.labels[idx])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        annotations = json.load(open(label_path))
        # annotations = list(annotations.values())  # don't need the dict keys
        # annotations = [a for a in annotations if a['regions']]

        # only include polygons
        objs = [i for i in annotations['regions'] if i['shape_attributes']['name'] == 'polygon']

        class_dict = {'刮擦-轻度损伤': 20, '刮擦-中度损伤': 1, '刮擦-重度损伤': 2, '刮擦-换件': 3, \
                      '开裂-轻度损伤': 4, '开裂-中度损伤': 5, '开裂-重度损伤': 6, '开裂-换件': 7, \
                      '褶皱-轻度损伤': 8, '褶皱-中度损伤': 9, '褶皱-重度损伤': 10, '褶皱-换件': 11, \
                      '穿孔-轻度损伤': 12, '穿孔-中度损伤': 13, '穿孔-重度损伤': 14, '穿孔-换件': 15, \
                      '凹陷-轻度损伤': 16, '凹陷-中度损伤': 17, '凹陷-重度损伤': 18, '凹陷-换件': 19, \
                      '其他-其他': 21, '其他-换件': 22, '其他-轻度损伤': 23, '其他-中度损伤': 24, '其他-重度损伤': 25, \
                      '刮擦-其他': 26, '开裂-其他': 27, '褶皱-其他': 30, '穿孔-其他': 28, '凹陷-其他': 29}
        # 获取每个mask的边界框坐标
        num_objs = len(objs)
        boxes = []
        classes = []
        masks = []
        for i in range(num_objs):
            try:
                xmin = np.min(objs[i]['shape_attributes']['all_points_x'])
                xmax = np.max(objs[i]['shape_attributes']['all_points_x'])
                ymin = np.min(objs[i]['shape_attributes']['all_points_y'])
                ymax = np.max(objs[i]['shape_attributes']['all_points_y'])
                boxes.append([xmin, ymin, xmax, ymax])
                # 备注：类别列表 -
                # D-损伤类型：【刮擦，开裂，褶皱，穿孔，其他】
                # E-损伤程度： 【轻度损伤，中度损伤，重度损伤，其他】
                d_type = objs[i]['region_attributes']['D-损伤类型']
                e_type = objs[i]['region_attributes']['E-损伤程度']
                if e_type == '轻度':
                    e_type = '轻度损伤'
                if e_type == '中度':
                    e_type = '中度损伤'
                if e_type == '重度':
                    e_type = '重度损伤'
                classes.append(d_type + '-' + e_type)

                # convert masks
                # Create an empty mask and then fill in the polygons
                mask = np.zeros([width, height])
                a3 = vim_to_labels(objs[i]['shape_attributes'])
                cv2.fillPoly(mask, a3, 1)
                masks.append(mask)
            except:
                print("annotations: ", annotations)
                print("<<< objs", i, objs[i])

        # 将所有转换为torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([class_dict[i] for i in classes], dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class LianbaoDatasetModelC(torch.utils.data.Dataset):
    # assume the each contains one label file for one pic, inside two folders with same name
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "label"))))
        if '.ipynb_checkpoints' in self.imgs:
            self.imgs.remove('.ipynb_checkpoints')
        if '.ipynb_checkpoints' in self.labels:
            self.labels.remove('.ipynb_checkpoints')

    def __getitem__(self, idx):
        #print ("<<< idx", idx)
        img_path = os.path.join(self.root, "pic", self.imgs[idx])
        #print ("<<< img_path ", img_path )
        label_path = os.path.join(self.root, "label", self.labels[idx])
        #print("<<< label_path ", label_path)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        annotations = json.load(open(label_path))
        # annotations = list(annotations.values())  # don't need the dict keys
        # annotations = [a for a in annotations if a['regions']]

        # only include polygons
        objs = [i for i in annotations['regions'] if i['shape_attributes']['name'] == 'polygon']

        errors = ['A-1-前保险杠',
                  'A-2-前保险杠格栅（下）',
                  'A-3-前大灯（右）',
                  'A-4-前大灯（左）',
                  'A-5-中网',
                  'B-6-后保险杠',
                  'B-7-后保险杠装饰灯（右）',
                  'B-8-后保险杠装饰灯（左）',
                  'B-9-尾灯',
                  'B-10-尾灯（右）',
                  'B-11-尾灯（左）',
                  'B-12-内尾灯（左）',
                  'B-13-内尾灯（右）',
                  'B-14-外尾灯（左）',
                  'B-15-外尾灯（右）',
                  'C-16-前挡风玻璃',
                  'D-17-后挡风玻璃',
                  'E-18-车顶',
                  'F-19-引擎盖',
                  'G-20-行李箱盖',
                  'H-21-钢圈',
                  'I-22-前叶子板（左）',
                  'I-23-前轮眉（左）',
                  'J-24-前叶子板（右）',
                  'J-25-前轮眉（右）',
                  'K-26-底大边（左）',
                  'K-27-前立柱（左）',
                  'K-28-后立柱（左）',
                  'L-29-底大边（右）',
                  'L-30-前立柱（右）',
                  'L-31-后立柱（右）',
                  'M-32-后叶子板（左）',
                  'M-33-后轮眉（左）',
                  'N-34-后叶子板（右）',
                  'N-35-后轮眉（右）',
                  'O-36-前门（左）',
                  'O-37-前门玻璃（左）',
                  'O-38-前门外拉手（左）',
                  'P-39-前门（右）',
                  'P-40-前门玻璃（右）',
                  'P-41-前门外拉手（右）',
                  'Q-42-后门（左）',
                  'Q-43-后门玻璃（左）',
                  'Q-44-后门外拉手（左）',
                  'R-45-后门（右）',
                  'R-46-后门玻璃（右）',
                  'R-47-后门外拉手（右）',
                  'S-48-倒车镜（左）',
                  'T-49-倒车镜（右）',
                  'Z-99-其它']

        class_dict = {}
        for i in errors:
            class_dict[i] = errors.index(i) + 1

        # 获取每个mask的边界框坐标
        num_objs = len(objs)
        boxes = []
        classes = []
        masks = []
        for i in range(num_objs):
            try:
                xmin = np.min(objs[i]['shape_attributes']['all_points_x'])
                xmax = np.max(objs[i]['shape_attributes']['all_points_x'])
                ymin = np.min(objs[i]['shape_attributes']['all_points_y'])
                ymax = np.max(objs[i]['shape_attributes']['all_points_y'])
                boxes.append([xmin, ymin, xmax, ymax])
                #assert box shape
                c_type = objs[i]['region_attributes']['C-外观零部件']
                # error fix
                c_type = error_fix(c_type)
                classes.append(c_type)

                # convert masks
                # Create an empty mask and then fill in the polygons
                mask = np.zeros([width, height])
                a3 = vim_to_labels(objs[i]['shape_attributes'])
                cv2.fillPoly(mask, a3, 1)
                masks.append(mask)
            except:
                print("annotations: ", annotations)
                print("<<< objs", i, objs[i])

        # 将所有转换为torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([class_dict[i] for i in classes], dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # make sure below happends to avoid nan box loss, refer to https://github.com/pytorch/vision/issues/997
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def error_fix(x):
    if x =='A-2-前保险杠隔栅（下）':
        return "A-2-前保险杠格栅（下）"
    else:
        return x

def vim_to_labels(obj):
    res = []
    xs = obj['all_points_x']
    ys = obj['all_points_y']
    for i in range(len(xs)):
        res.append([xs[i], ys[i]])

    return (np.array([res], dtype=np.int32))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

