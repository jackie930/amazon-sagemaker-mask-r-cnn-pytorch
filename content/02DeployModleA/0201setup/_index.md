+++
title = "环境搭建和验证"
weight = 0201
chapter = true
pre = "<b>2.1 </b>"
+++

#### EC2 环境准备

要完成本章节操作步骤，您需要准备一台EC2 实例：

- AMI : Deep Learning AMI (Ubuntu 18.04) Version 39.0 - ami-08773c85de0140def
- 实例类型： g4dn.xlarge  （4C/16G）
- 存储: 150G 



#### 获取模型代码
登录已经启动的 EC2 实例，创建工作目录，并下载代码
```
source activate pytorch_p36
mkdir workshop
cd workshop
git clone https://github.com/jackie930/amazon-sagemaker-mask-r-cnn-pytorch
```

#### 配置AK/SK
使用“aws configure” 命令配置AK/SK
```
aws configure
AWS Access Key ID [None]: ********************
AWS Secret Access Key [None]: ****************************************
Default region name [None]: cn-northwest-1
Default output format [None]:
```
**注意** 请将“*******” 替换为您AWS 帐号的Access Key ID 和 AWS Secret Access Key 


#### 检查模型文件
检查一下两个模型文件(.pth) 是否已经存在
```
ls -la *pth
-rw-r--r-- 1 ubuntu ubuntu 178090079 Feb  2 23:31 maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
-rw-r--r-- 1 ubuntu ubuntu  44793465 Feb  2 23:25 model.pth
```

#### 修改Dockerfile
如需要使用本地GPU EC2实例部署和使用模型A，需要修改Dockerfile

1. 修改gpu基础镜像：
```
ARG REGISTRY_URI
# FROM ${REGISTRY_URI}/pytorch-inference:1.5.0-cpu-py36-ubuntu16.04
FROM ${REGISTRY_URI}/pytorch-inference:1.5.0-gpu-py3
```

2. 增加以下安装命令：
```
 RUN pip install torchvision GPUtil  -i https://opentuna.cn/pypi/web/simple
```

3. 修改复制内容
```
#COPY * /opt/program/
COPY maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth  /root/.cache/torch/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
COPY model.pth /opt/program/
COPY predictor.py /opt/program/
COPY serve.py /opt/program/
COPY wsgi.py /opt/program/
COPY nginx.conf /opt/program/
WORKDIR /opt/program
```

#### 修改模型代码
```
import torchvision
import PIL
import codecs
import copy
import re
import base64
from io import BytesIO
```


```
    data = flask.request.data.decode('utf-8')
    # data = json.loads(data)

    # bucket = data['bucket']
    # image_uri = data['image_uri']
    
    img_data = data

    # download_file_name = image_uri.split('/')[-1]
    # print ("<<<<download_file_name ", download_file_name)
    # #download_file_name = './test.jpg'
    # s3_client.download_file(bucket, image_uri, download_file_name)
    # print('Download finished!')
    # inference and send result to RDS and SQS

    print('Start to inference:')

    if torch.cuda.is_available():
        print("==> Using cuda")
    else:
        print("==> Using cpu")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

```
    input_size = 224

    trans2 = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # image = PIL.Image.open(download_file_name)
    img_data = data
    image = base64_to_image(img_data)
```

```
def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = PIL.Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img
```

#### Build Docker image
```
./build_local.sh
```
执行成功后检查image 并启动容器
```
docker images
REPOSITORY                                                               TAG                          IMAGE ID       CREATED        SIZE
car-filter                                                               latest                       44e85c202503   15 hours ago   9.54GB
```
启动容器
```
nvidia-docker  run -d -p 8080:8080 --env AWS_DEFAULT_REGION=cn-northwest-1 car-filter
```


#### 数据准备
将模拟测试数据上传到 EC2 "sample_20210202" 目录
```
find sample_20210202/
sample_20210202/
sample_20210202/标的定损-本田
sample_20210202/标的定损-本田/现场查勘照片
sample_20210202/标的定损-本田/现场查勘照片/外观照片 (10)29.jpg
sample_20210202/标的定损-本田/现场查勘照片/内杠损失 (1)16.jpg
sample_20210202/标的定损-本田/现场查勘照片/人车合影19.jpg
....
sample_20210202/三者定损-江铃江特
sample_20210202/三者定损-江铃江特/现场查勘照片
sample_20210202/三者定损-江铃江特/现场查勘照片/现场查勘照片15.jpg
sample_20210202/三者定损-江铃江特/现场查勘照片/现场查勘照片10.jpg
sample_20210202/三者定损-江铃江特/现场查勘照片/现场查勘照片14.jpg
...
```

创建输出目录 “output”
```
mkdir output
python create_output.py -input sample_20210202/ -output output/

ls output/
out_car  out_err1  out_err2  out_no_car  out_not_pic  out_parts


```


#### 测试调用模型
```
python run_filter.py  -input "sample_20210202/" -output "output" -api "http://localhost:8080/invocations"
```


#### 导出 / 导入 image
执行以下命令导出image
```
docker save car-filter -o car-filter.tar
```
将tar 文件复制到另一台服务器，执行以下命令，装载镜像
```
docker load < car-filter.tar
```

