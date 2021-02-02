+++
title = " 模型训练"
weight = 0402
chapter = true
pre = "<b>4.2 </b>"
+++

这里，我们来介绍一下如何使用sagemaker训练一个目标分割型。共分为以下几步：

* [下载数据]((#下载代码))
* [下载代码]((#下载数据))
* [本地训练](#训练)
* [模型训练日志](#模型训练日志)
* [查看结果](#查看结果)

### 下载代码
```sh
cd SagemMaker && git clone https://github.com/jackie930/amazon-sagemaker-mask-r-cnn-pytorch.git
```

### 下载数据
```sh
cd amazon-sagemaker-mask-r-cnn-pytorch
mkdir data && cd data

#download file (todo: change access)
wget s3://lianbao-mask-rcnn/modelb/modelc.zip

#untar 
tar -zxvf modelc.zip
```

运行后，你可以看到对应的文件目录

```
-|--label
 |--pic
 |--jingbiao.json
```

### 本地训练测试

模型c

```sh
source activate pytorch_p36
pip install pycocotools tensorboard
cd container/mask_r_cnn
python local_train.py \
  --root_train_data='../../data/modelc' \
  --model_type='modelc_sub' \
  --num_epochs=2 \
  --save_path='../../models/modelc'
```

### 训练日志

![](train-progress.png)

### 训练结果

模型产生的结果目录中，可以看到模型在测试集上的表现

IoU metric: bbox | 结果|
-|-|
 Average Precision  (AP) @[ IoU=0.50:0.95  area=   all maxDets=100 ] |0.464 |
 Average Precision  (AP) @[ IoU=0.50       area=   all  maxDets=100 ] | 0.646 |
 Average Precision  (AP) @[ IoU=0.75       area=   all  maxDets=100 ] | 0.526 |
 Average Precision  (AP) @[ IoU=0.50:0.95  area= small  maxDets=100 ] | 0.118 |
 Average Precision  (AP) @[ IoU=0.50:0.95  area=medium  maxDets=100 ] | 0.261 |
 Average Precision  (AP) @[ IoU=0.50:0.95  area= large  maxDets=100 ] | 0.572 |
 Average Recall     (AR) @[ IoU=0.50:0.95  area=   all  maxDets=  1 ] | 0.481 |
 Average Recall     (AR) @[ IoU=0.50:0.95  area=   all  maxDets= 10 ] | 0.658 |
 Average Recall     (AR) @[ IoU=0.50:0.95  area=   all  maxDets=100 ] | 0.658 |
 Average Recall     (AR) @[ IoU=0.50:0.95  area= small  maxDets=100 ] | 0.188 |
 Average Recall     (AR) @[ IoU=0.50:0.95  area=medium  maxDets=100 ] | 0.470 |
 Average Recall     (AR) @[ IoU=0.50:0.95  area= large  maxDets=100 ] | 0.780 |
