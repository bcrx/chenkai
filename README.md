# 基于CPU-GPU异构混合编程的遥感数据时空融合程序，包括STDFA，STARFM，CDSTARFM三种模型
Version 1.0

## 上手指南
以下指南将帮助你在本地机器上安装和运行该项目，进行开发和测试。关于如何将该项目部署到在线环境，请参考部署小节。<br>

### 1、windows10下环境部署步骤
#### 1.1 Python安装
建议安装Anaconda，默认安装，全英文路径。anaconda下载网址如下：<br>
https://www.anaconda.com/download/<br>

#### 1.2 GDAL安装
下载Python对应版本的gdal安装文件，比如GDAL‑2.4.1‑cp36‑cp36m‑win_amd64.whl。下载地址如下：<br>
https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal<br>
打开命令提示符输入一下命令<br>
python -m pip install GDAL‑2.4.1‑cp36‑cp36m‑win_amd64.whl<br>
检测是否安装成功：<br>
打开命令行，输入python，进入python运行环境，输入import gdal，不报错，表示安装成功。<br>

#### 1.3 MRT（MODIS Reprojection Tool）安装
链接：https://pan.baidu.com/s/1ANu9xt6C_aRGRzI7HMyIpA 提取码：chbe <br>
由于安装MRT需要Java环境，首先需要安装Java，下载路径如下：<br>
https://www.java.com/zh_CN/download/manual.jsp<br>
Java安装并配置好环境变量后，解压双击mrt_install.bat进行安装，按照提示进行。<br>
检测是否安装成功:<br>
MRT安装路径下.\bin，双击ModisTool.bat，出现MRT处理界面，表示成功。<br>

#### 1.4 cuda安装
首先检查计算机显卡型号，是否支持cuda，支持哪个cuda版本，然后下载对应版本的cuda，默认安装。Cuda下载路径如下:<br>
https://developer.nvidia.com/cuda-downloads<br>
可参考官方文档https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html<br>
检测是否安装成功:<br>
以管理员模式打开cmd，输入nvcc -V，出现版本信息表示安装成功。<br>

## 程序运行方式

### 输入数据要求
Landsat输入数据要求：Landsat5,7,8标准数据集<br>
MODIS输入数据要求：MODGQ09标准数据集<br>

### 参数含义
各个参数含义查看帮助文档或者打开cmd分别运行python STDFA.py --help，python STARFM.py --help，python CDSTARFM.py --help<br>

### 程序运行
STDFA：修改参数后运行STDFA.bat<br>
STDFA：修改参数后运行STARFM.bat<br>
STDFA：修改参数后运行CDSTARFM.bat<br>
说明：修改参数为帮助文档中的必选参数<br>
