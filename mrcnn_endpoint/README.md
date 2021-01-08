# Maskrcnn-SageMaker
Deploy `MaskRCNN` (torch-vision)  on `Amazon SageMaker`

## Maskrcnn
`Mask R-CNN` 是一个两阶段的框架，第一个阶段扫描图像并生成提议（proposals，即有可能包含一个目标的区域），第二阶段分类提议并生成边界框和掩码。Mask R-CNN 扩展自 Faster R-CNN，由同一作者提出。Faster R-CNN 是一个流行的目标检测框架，Mask R-CNN 将其扩展为实例分割框架。
[source code](https://github.com/matterport/Mask_RCNN)

## Features

- [x] **Use `MaskRCNN` pretrained model (coco) to deploy on `Amazon SageMaker`**
- [x] **support gpu inference**

## Quick Start

---------------

# build the environment and test locally

server
~~~~shell script
sh build_and_push.sh
#if test locally, run below
docker run -v -d -p 8080:8080 mrcnn-car
~~~~

client
~~~~shell script
import requests
import json

url='http://localhost:8080/invocations'

bucket = 'spot-bot-asset'
image_uri = 'end/car.jpg'
test_data = {
    'bucket' : bucket,
    'image_uri' : image_uri,
    'content_type': "application/json",
}
payload = json.dumps(test_data)


r = requests.post(url,data=payload)

#show result
print (r.text)
CPU times: user 763 ms, sys: 125 ms, total: 888 ms
Wall time: 795 ms
~~~~

result
~~~~
{"result": {"label": "withcar"}}
~~~~

# build endpoint
~~~~
cd mrcnn_endpoint
python create_endpoint.py
~~~~

# use endpoint
~~~~ python
def infer(input_image):
    from boto3.session import Session
    import json

    bucket = 'lianbao-mask-rcnn'
    image_uri = input_image
    test_data = {
        'bucket' : bucket,
        'image_uri' : image_uri,
        'content_type': "application/json",
    }
    payload = json.dumps(test_data)


    session = Session()

    runtime = session.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName='mrcnn-car',
        ContentType="application/json",
        Body=payload)

    result = json.loads(response["Body"].read())
    print (result)
    
infer('test.jpg')
~~~~

multi-process

~~~~ python
%%time 
import multiprocessing as mul

pool = mul.Pool(5)
rel = pool.map(infer, ['end/test2.jpg','end/test.jpg','end/test2.jpg','end/test2.jpg','end/test2.jpg','end/test2.jpg'])
print(rel)
~~~~