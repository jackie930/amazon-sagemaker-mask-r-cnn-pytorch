+++
title = "创建SageMaker笔记本实例"
weight = 0401
chapter = true
pre = "<b>4.1 </b>"
+++

### 创建笔记本实例
SageMaker 提供无需设置的托管Jupyter Notebook，因此您可以立即开始处理您的训练数据集。只需在 SageMaker 控制台中单击几下，您就可以创建完全托管的笔记本实例，预先加载了用于机器学习的有用库。您只需添加您的数据。

首先，您将创建将在整个研讨会中使用的 Amazon S3 存储桶。然后，您将创建一个 SageMaker 笔记本实例，该实例将用于其他研讨会模块。

#### 创建 S3 存储桶
SageMaker 通常使用 S3 作为数据和模型工件的存储。在此步骤中，您将为此目的创建 S3 存储桶。要开始，请登录 AWS 管理控制台 https://console.amazonaws.cn/。

请记住，您的存储桶名称必须在所有区域和客户中具有全球唯一性。我们建议使用像 `spot-bot-exampledata-cn-northwest-1-123456789012` 这样的名字（这里的"123456789012"代表您的账户ID）。如果您收到存储桶名称已存在的错误信息，请尝试添加其他数字或字符，直到找到未使用的名称。

1. 在 AWS 管理控制台中，选择服务，然后在存储下选择 S3。

2. 选择创建存储桶

3. 为您的存储桶提供全局唯一的名称，例如 “spot-bot-exampledata-cn-northwest-1-123456789012”。(请参考使用机器人章节，【使用 cloudformation 创建 EC2 并创建 S3 Bucket】)

4. 从下拉列表中选择您选择用于此研讨会的区域。

5. 在对话框左下角选择创建，而不选择要从中复制设置的存储桶。

#### 启动笔记本实例

1. 在 AWS 管理控制台的右上角，确认您位于所需的 AWS 区域。选择由西云数据运营的AWS(宁夏)区域或由光环新网运营的AWS(北京)区域。

2. 点击所有服务列表中的亚马逊 SageMaker。这将带您访问亚马逊 SageMaker 控制台主页。

![控制台中的服务](sm-console-services.png)

3. 要创建新的笔记本实例，请转到笔记本实例，然后单击浏览器窗口顶部的创建笔记本实例按钮。

![笔记本实例](sm-notebook-instances.png)

4. 在笔记本实例名称文本框中键入mldev，然后选择 ml.t3.medium 作为笔记本实例类型。

![创建笔记本实例](sm-notebook-settings.png)

5. 选择名称为SpotBot的VPC，并选择名称为SpotBot-PublicSubnet2的子网。

![创建笔记本实例](sm-notebook-settings-2.png)

6. 对于 IAM 角色，选择创建新角色，然后在生成的弹出模式中，选择您指定的 S3 存储桶下的任意 S3 存储桶。单击创建角色。

![创建 IAM 角色](sm-role-popup.png)

7. 在IAM中选择角色，在角色列表中选择刚刚创建的SageMaker-ExecutionRole，并添加AmazonEC2ContainerRegistryFullAccess Policy。

![添加ECR访问权限](sm-role-ecr.png)

#### 访问笔记本电脑实例

1. 等待服务器状态更改为 InService。这将需要几分钟，可能最多 10 分钟，但可能更少。

![访问笔记本](sm-open-notebook.png)

2. 单击 “打开”。您现在将看到您的笔记本实例的 Jupyter 主页。

![打开笔记本](sm-jupyter-homepage.png)
