+++
title = " 模型训练"
weight = 0402
chapter = true
pre = "<b>4.2 </b>"
+++

这里，我们来介绍一下如何使用sagemaker训练一个情感分类模型。共分为以下几步：

* [下载数据]((#下载数据))
* [本地训练](#本地训练)
* [SageMaker训练，部署](#SageMaker训练，部署)
    * [构建ecr镜像](#构建ecr镜像)
    * [模型训练](#模型训练)
    * [查看结果](#查看结果)
    * [模型部署](#模型部署)
* [测试调用](#测试调用)

### 下载数据
```sh
#download data/model files/JupyterNotebook
wget https://spot-bot-asset.s3.amazonaws.com/spot-workshop-2020/demo2.tar.gz

#untar 
tar -zxvf demo2.tar.gz
```

运行后，你可以看到对应的文件目录

```
-|--bert
 |--data
 |--build_and_push.sh
 |--DockerFile
 |--train.ipynb
```

### 本地训练测试

模型b

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

### 模型部署
```python
predictor = estimator.deploy(1, instance_type='ml.m5.large', endpoint_name='bert-sentiment')
```
运行后，会看到生成了对应的endpoint

### 测试调用

![](test.png)