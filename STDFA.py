'''
2018.6.10
version 1.0
STDFA时空数据融合GPU并行加速程序
作者：西南交通大学地球科学与环境工程学院陈凯

'''

import gdal
import numpy
import sys
from gdalconst import *

import struct
import copy
import os
import shutil
import math
import time

import threading
import gc #内存管理
import random

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


class Fusion:
    #初始化
    def __init__(self,root,class_count = 10,max_iter = 10,band_number = 3,window_size = 7):
        #设置根目录,分类数，最大迭代数，波段，窗口大小
        self.root = root
        self.class_count = class_count
        self.band_number = band_number
        self.window_size = window_size
        self.max_iter = max_iter
        os.chdir(root)

    #读取影像
    def read_image(self,path):
        print('正在读取影像%s...'%path)
        image = gdal.Open(path)
        if image is None:
            print('影像不存在或损坏，请重新选择影像路径!')
            return 0
        print('影像读取完毕!')
        return image

    #读取影像数据
    def read_data(self,image,band):
        print('正在读取数据...')
        band = image.GetRasterBand(band)
        data = band.ReadAsArray()
        print('数据读取完毕!')
        return data

    #landsat反射率计算
    def cal_ref_landsat5(self,metadata_name,data,band_number):
        metadata = open(metadata_name)

        str_metadata = metadata.read()

        metadata.close()

        if(not metadata):
            print("缺少头文件,请重新选择影像")
            return numpy.array([0])

        band_name = ['blue','green','red','NIR','SWIR1','LWIR','SWIR2']
        print('开始计算landsat5的%s波段反射率...'%band_name[band_number-1])
        
        #太阳天顶角
        sun_azimuth_index = str_metadata.find('SUN_ELEVATION')
        sun_elevation = float(str_metadata[sun_azimuth_index+16:sun_azimuth_index+26]) #保留7位小数
        print('太阳高度角:',sun_elevation)
        sun_zenith = (90 - sun_elevation )/180*math.pi

        #大气顶层太阳辐照度
        ESUN = [1957,1826,1554,1036,215,80.96]

        #日地距离
        temp_path = os.path.split(metadata_name)[-1]
        if(int(temp_path[10:14])>2006):
            time_index = str_metadata.find('ACQUISITION_DATE')
            time = str_metadata[time_index+19:time_index+29]
        else:
            time_index = str_metadata.find('DATE_ACQUIRED')
            time = str_metadata[time_index+16:time_index+26]                

        year = int(time[0:4])
        month = int(time[5:7])
        day = int(time[8:10])
        print('影像成像日期：%d年%d月%d日'%(year,month,day))
        
        JD = 1721103.5 + int(365.25*year)+int(30.6*month+0.5)+day
        d = 1+0.01674*math.cos(0.9856*(JD-2451545)*math.pi/180/36525)
        print('日地距离：',d)

        #MIN_MAX_RADIANCE,MIN_MAX_PIXEL_VALUE
        Lmax = [193.000,365.000,264.000,221.000,30.200,15.303,16.500]
        Lmin = [-1.520,-2.840,-1.170,-1.510,-0.370,1.238,-0.150]
        Qcal_max = 255.0
        Qcal_min = 0.0
        gain = [0.7628,1.4425,1.0399,0.8726,0.1199,0.0551,0.0653]

        
        #反射率计算
        temp = math.pi*d*d/(ESUN[band_number-1]*math.cos(sun_zenith))

        reflectance = copy.deepcopy(data).astype('float32')

        [m,n] = data.shape

        #反射率GPU计算
        #定义核函数
        mod = SourceModule("""
        #include <math.h>
        __global__ void func(float *reflectance, float *data, float temp,float gain,float Lmin,int m,int n)
        {
          const int i = blockIdx.x * blockDim.x + threadIdx.x;

          //reflectance[0,0] = 1.0;
          if (i > (m*n))
          {
            return;
          }
          float temp_data = data[i];
          
          if(!temp_data){reflectance[i] = 0.0;}
          else{
                    float L = gain*temp_data+Lmin;
                    reflectance[i] = temp*L;
                    if(reflectance[i]<0){
                        reflectance[i] = 0.001;
                        }}

        }
        """)

        func = mod.get_function("func")
        [m,n] = data.shape
        reflectance = copy.deepcopy(data).astype('float32').reshape(1,m*n)
        data = data.astype(numpy.float32).reshape(1,m*n)
        
        m = numpy.int32(m)
        n = numpy.int32(n)
       
        temp = numpy.float32(temp)
        gain = numpy.float32(gain[band_number-1])
        Lmin = numpy.float32(Lmin[band_number-1])
        nTheads = 128
        nBlocks = int( ( m*n + nTheads - 1 ) / nTheads )#保持多一个块
        
        func(
                drv.InOut(reflectance), drv.In(data),temp,gain,Lmin,m,n,
                block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        print(data[0,0],reflectance[0,0],m,n)
        reflectance = reflectance.reshape(m,n)


        print('landsat5的%s波段反射率计算结束!'%band_name[band_number-1])

        return reflectance

    def cal_ref_landsat7(self,metadata_name,data,band_number):
        metadata = open(metadata_name)

        str_metadata = metadata.read()

        metadata.close()

        if(not metadata):
            print("缺少头文件,请重新选择影像")
            return numpy.array([0])

        band_name = ['blue','green','red','NIR','SWIR1','TIR','SWIR2']
            
        print('开始计算landsat7的%s波段反射率...'%band_name[band_number-1])

            
        #太阳天顶角
        sun_elevation_index = str_metadata.find('SUN_ELEVATION')
        sun_elevation = float(str_metadata[sun_elevation_index+16:sun_elevation_index+28]) #保留8位小数
        sun_zenith = (90-sun_elevation)/180*math.pi
        print('太阳天顶角:',sun_zenith)

        #反射率调整因子
        reflectance_mult_index = str_metadata.find('REFLECTANCE_MULT_BAND_'+str(band_number))
        reflectance_mult = float(str_metadata[reflectance_mult_index+26:reflectance_mult_index+36])
        print('%s波段的反射率增益:'%(band_name[band_number-1]),reflectance_mult)

        #反射率调整参数
        reflectance_add_index = str_metadata.find('REFLECTANCE_ADD_BAND_'+str(band_number))
        reflectance_add = float(str_metadata[reflectance_add_index+25:reflectance_add_index+34])
        print('%s波段的反射率偏移:'%(band_name[band_number-1]),reflectance_add)
        

        #反射率GPU计算
        #定义核函数
        mod = SourceModule("""
        #include <math.h>
        __global__ void func(float *reflectance, float *data, float reflectance_mult,float reflectance_add,float sun_zenith,int m,int n)
        {
          const int i = blockIdx.x * blockDim.x + threadIdx.x;

          if (i > m*n)
          {
            return;
          }
          float temp_data = data[i];
          if(!temp_data){reflectance[i] = 0.0;}
          else{
                    reflectance[i] = (reflectance_mult*temp_data+reflectance_add)/sun_zenith;
                    if(reflectance[i]<0){
                        reflectance[i] = 0.001;
                        }}
          // a[i] = a[i] + b[i];
        }
        """)

        func = mod.get_function("func")
        [m,n] = data.shape
        reflectance = copy.deepcopy(data).astype('float32').reshape(1,m*n)
        data = data.astype(numpy.float32).reshape(1,m*n)
        
        m = numpy.int32(m)
        n = numpy.int32(n)
        reflectance_mult = numpy.float32(reflectance_mult)
        reflectance_add = numpy.float32(reflectance_add)
        sun_zenith = numpy.float32(sun_zenith)
        
        nTheads = 256
        nBlocks = int( ( m*n + nTheads - 1 ) / nTheads )#保持多一个块

        func(
                drv.InOut(reflectance), drv.In(data),reflectance_mult,reflectance_add,math.cos(sun_zenith),m,n,
                block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        reflectance = reflectance.reshape(m,n)

        print('\nlandsat7的%s波段反射率计算结束!'%band_name[band_number-1])

        return reflectance

    def cal_ref_landsat8(self,metadata_name,data,band_number):
        metadata = open(metadata_name)

        str_metadata = metadata.read()

        metadata.close()

        if(not metadata):
            print("缺少头文件,请重新选择影像!")
            return numpy.array([0])
        band_name = ['coas','blue','green','red','NIR','SWIR1','SWIR2']
            
        print('开始计算landsat8的%s波段反射率...'%band_name[band_number-1])

        
        #太阳天顶角
        sun_elevation_index = str_metadata.find('SUN_ELEVATION')
        sun_elevation = float(str_metadata[sun_elevation_index+16:sun_elevation_index+28]) #保留8位小数
        sun_zenith = (90-sun_elevation)/180*math.pi
        print('太阳天顶角:',sun_zenith)

        #反射率调整因子
        reflectance_mult_index = str_metadata.find('REFLECTANCE_MULT_BAND_'+str(band_number))
        reflectance_mult = float(str_metadata[reflectance_mult_index+26:reflectance_mult_index+36])
        print('%s波段的反射率增益:'%(band_name[band_number-1]),reflectance_mult)

        #反射率调整参数
        reflectance_add_index = str_metadata.find('REFLECTANCE_ADD_BAND_'+str(band_number))
        reflectance_add = float(str_metadata[reflectance_add_index+25:reflectance_add_index+34])
        print('%s波段的反射率偏移:'%(band_name[band_number-1]),reflectance_add)
        

        #反射率计算
        mod = SourceModule("""
        #include <math.h>
        __global__ void func(float *reflectance, float *data, float reflectance_mult,float reflectance_add,float sun_zenith,int m,int n)
        {
          const int i = blockIdx.x * blockDim.x + threadIdx.x;

          if (i > m*n)
          {
            return;
          }
          float temp_data = data[i];
          if(!temp_data){reflectance[i] = 0.0;}
          else{
                    reflectance[i] = (reflectance_mult*temp_data+reflectance_add)/cos(sun_zenith);
                    if(reflectance[i]<0){
                        reflectance[i] = 0.001;
                        }}
          // a[i] = a[i] + b[i];
        }
        """)

        func = mod.get_function("func")
        [m,n] = data.shape
        reflectance = copy.deepcopy(data).astype('float32').reshape(1,m*n)
        data = data.astype(numpy.float32).reshape(1,m*n)
        
        m = numpy.int32(m)
        n = numpy.int32(n)
        reflectance_mult = numpy.float32(reflectance_mult)
        reflectance_add = numpy.float32(reflectance_add)
        sun_zenith = numpy.float32(sun_zenith)
        
        nTheads = 256
        nBlocks = int( ( m*n + nTheads - 1 ) / nTheads )#保持多一个块

        func(
                drv.InOut(reflectance), drv.In(data),reflectance_mult,reflectance_add,sun_zenith,m,n,
                block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        reflectance = reflectance.reshape(m,n)

        print('\nlandsat8的%s波段反射率计算结束!'%band_name[band_number-1])

        return reflectance

    def cal_reflectance(self,metadata_name,data,band_number,landsat_ID):
        if(5 == landsat_ID):
            reflectance = self.cal_ref_landsat5(metadata_name,data,band_number)
            return reflectance

        if(7 == landsat_ID):
            reflectance = self.cal_ref_landsat7(metadata_name,data,band_number)
            return reflectance

        if(8 == landsat_ID):
            reflectance = self.cal_ref_landsat8(metadata_name,data,band_number)
            return reflectance
        
    
    #将landsat_t1时刻的影像裁剪为scale倍数的大小，使能够与modis影像匹配
    def image_cut(self,image,scale):
        data = self.read_data(image,1)
        [m,n] = data.shape
        height = m - m%scale
        width = n - n%scale
        #计算角点坐标
        transform = image.GetGeoTransform()
        xmin = transform[0]
        xmax = xmin + width*transform[1]
        ymax = transform[3]
        ymin = ymax + height*transform[5]
        return [[xmin,ymin,xmax,ymax],data[0:height,0:width]]



     #提取丰度图,形参：分类图，返回值：丰度图二维列表

    def abundance_extract(self,class_image,landsat_resolusion,modis_resolusion):
        #格网宽高
        grid = int(modis_resolusion/landsat_resolusion)
        grid_height = grid
        grid_width = grid
        #格网数
        count = grid * grid

        [m,n] = class_image.shape

        class_image_height = int(m/grid)
        class_image_width = int(n/grid)
    
        abundance = [[0 for col in range(class_image_width)] for row in range(class_image_height)]
        abundance_temp = [[0 for col in range(class_image_width)] for row in range(class_image_height)]
        abundance_class = [[0 for col in range(class_image_width)] for row in range(class_image_height)]

        percent = int(class_image_height/20)

            
        for i in range(class_image_height):
            if(not i%percent):
                print('%d%%'%(int(i/percent*5)),end = ' ')
            for j in range(class_image_width):

                abundance_count = numpy.zeros(self.class_count) #丰度图计数
                for h in range(i*grid_height,(i+1)*grid_height):
                    for w in range(j*grid_width,(j+1)*grid_width):
                        if(int(class_image[h][w])<self.class_count):
##                        print(h,w,class_image[h][w])
                            abundance_count[int(class_image[h][w])] += 1
                abundance_percent = abundance_count/count
                abundance[i][j] = abundance_percent
                abundance_count = numpy.zeros(self.class_count)
            
        print('')
        return abundance

        #最小二乘Ax = b,求解x = (A.T*A).I*A.T*b,参数：丰度矩阵n*8，modis影像,data_0为landsat没有值属于的类别
    def least_squares(self,abundance_image,modis_image):

        [modis_height,modis_width] = modis_image.shape
##        print(class_index,data_0)
        
        total_m = modis_height*modis_width

        x = numpy.array([[0.0] for i in range(self.class_count)])

##        print(numpy.array(abundance_image).shape,numpy.array(abundance_image))
        abundance = (numpy.array(abundance_image).reshape(total_m,self.class_count)).tolist()
        
        #将modis反射率转化为n*1的矩阵,n为modis像素数
        modis_R = (modis_image.reshape(total_m,1)).tolist()
##        print(modis_R)

        #剔除有云的地方，不参与运算
        i = 0
        while(i<total_m):
            if(modis_R[i][0]>0.5):
                modis_R.pop(i)
                abundance.pop(i)
                #索引向前移动一位
                total_m-=1
            else:
                i+=1
        #还原变化的值
        total_m = modis_height*modis_width
        #约束方程所用值
        ref_limit = sum(sum(modis_image))/total_m

##        print(abundance,modis_R)
        
        X = numpy.matrix(abundance) #转化为矩阵

##        print(A[3000:3030])
        y = numpy.matrix(modis_R)
        #创建单位矩阵
        ones = numpy.zeros((self.class_count,self.class_count))
        for i in range(self.class_count):
            ones[i,i] = 1.0
        I = numpy.matrix(ones)
        
        #约束方程1,A1x>=0
        A1 = copy.deepcopy(I)
        c1 = numpy.matrix(numpy.zeros((self.class_count,1)))
        
        #约束方程2,,即Ax<=1
##        for i in range(self.class_count):
##            ones[i,i] = -1.0
##        A2 = numpy.matrix(ones)
        c2 = numpy.matrix(numpy.ones((self.class_count,1)))

        
####        print(A,b)
        XT = X.T
        XTX = XT*X
        XTy = XT*y
        
        if((numpy.linalg.det(XTX)) == 0.0):
            
##            print('bukeni')
            return x
        b = XTX.I*XTy
##        print('b',b)

        AA = []
        cc = []
        for i in range(len(A1)):
            if(b[i,0]<0):
                
                AA.append(A1[i].tolist()[0])
                cc.append([ref_limit])

        AA = numpy.matrix(AA).astype('float32')
        cc = numpy.matrix(cc).astype('float32')
        if(AA.any()):
            temp1 = AA*XTX.I*AA.T
            det = numpy.linalg.det(temp1)
##            print('det',det)
            if(det):
                temp2 = XTX.I
                AAT = AA.T
                temp1_I = temp1.I
                b = (I-temp2*AAT*temp1_I*AA)*temp2*XT*y+temp2*AAT*temp1_I*cc

        x = b
##        print(x)


##            x[i] = (A.T*A).I*A.T*b
##        print('x',x,modis_R.shape)

        return x
    #像元反射率计算r(c,tj) = mean_r(c,tj) - mean_r(c,ti) +r(c,ti),GPU
    def GPUcal_pixel_ref(self,landsat_reflect_ti,class_mean_difference):
        block_size_x,block_size_y = landsat_reflect_ti.shape
        total_pixel = block_size_x*block_size_y
        nTheads = 256
        nBlocks = int((total_pixel+nTheads-1)/nTheads)
        #print(total_pixel,nBlocks)
        mod = SourceModule('''
            //编写c++核函数
            __global__ void cal_ref(double *land_ref, double *class_mean_diff,double *new_landsat,int total)
            {
            	int count = blockIdx.x*blockDim.x + threadIdx.x;
            	
            	if (count > total) {
            		return;
            	}
             if (land_ref[count] <0.00001) {
                  new_landsat[count] = 0;
            		return;
            	}
            	
              new_landsat[count] = class_mean_diff[count] + land_ref[count]; //为了不出现黑点
            	
            }     
                           ''')
        new_ref = numpy.zeros((block_size_x,block_size_y))
        func_ref = mod.get_function("cal_ref")
        land_ref= landsat_reflect_ti.reshape(1,block_size_x*block_size_y).astype('float64')
        class_mean_difference= class_mean_difference.reshape(1,block_size_x*block_size_y).astype('float64')
        total = numpy.int32(total_pixel)
        new_landsat = new_ref.reshape(1,block_size_x*block_size_y).astype('float64')
        func_ref(
            drv.In(land_ref),drv.In(class_mean_difference),drv.InOut(new_landsat),total,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        new_ref = new_landsat.reshape(block_size_x,block_size_y)
        #print(new_landsat[0,9000:9050])
        return new_ref
    
        #均匀密度分割,GPU
    def GPUcal_class(self,landsat1,landsat2,GN,FL,FH):
        m,n = landsat1.shape
        total = m*n
        nTheads = 256
        nBlocks = int((total+nTheads-1)/nTheads)
        #print(total_pixel,nBlocks)
        mod1 = SourceModule('''
            //编写c++核函数
            #include<math.h>
            __global__ void cal_class(double *landsat1, double *landsat2, double *class_image,double gn,double fl,double fh,double total)
            {
            	int count = blockIdx.x*blockDim.x + threadIdx.x;
            	//class_image[count] = gn;
            	if (count > total) {
            		return;
            	}    
              if (landsat1[count] > 0.01 && landsat2[count] > 0.01) {
                  class_image[count] = int(gn*(landsat1[count]-landsat2[count]-0.5-fl)/(fh-fl+1));
                  //class_image[count] = int(gn*(landsat1[count]-landsat2[count]-0.5-fl)/(fh+1));
                  //class_image[count] = int((fh+1)*(landsat1[count]-landsat2[count]-fl-1)/gn+((fh)/10*pow(1.1/pow(gn,1/3),1/5)+0.4)*sin(2*3.1415*(landsat1[count]-landsat2[count]-fl-1)/gn)+0.5);
            		return;
            	} 
              else{
                  class_image[count] = gn;
                  return;
              }                           	
            }
                           ''')
        class_image = numpy.ones((m,n)).astype('int32')
        func_class = mod1.get_function("cal_class")
        landsat1= landsat1.reshape(1,total).astype('float64')
        landsat2= landsat2.reshape(1,total).astype('float64')
        class_image = class_image.reshape(1,total).astype('float64')
        gn = numpy.float64(GN)
        fl = numpy.float64(FL)
        fh = numpy.float64(FH)
        total = numpy.float64(total)
        #print(gn,fl,fh,total)
        #print(class_image[0,9000:9050])
        func_class(
            drv.In(landsat1),drv.In(landsat2),drv.InOut(class_image),gn,fl,fh,total,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        #print(class_image[0,29000000:29000050])
        class_image = class_image.reshape(m,n)
        return class_image
        
    #MRT转投影，location为范围，utm_zone为投影带，java路劲，MRT路劲
    def MRT(self,location,utm_zone,resolusion,modis_file_list,modis_band_number,save_path,java_path,MRT_path):
                
        x_min = location[0]
        y_min = location[1]
        x_max = location[2]
        y_max = location[3]

        #MRT参数编写

        modis_path_list = []
        for each in modis_file_list:
            temp_path = os.path.split(each)[0]
            if(temp_path not in modis_path_list):
                modis_path_list.append(temp_path)

        print(modis_path_list)
        for each_path in modis_path_list:
            print('当前目录%s'%os.getcwd())
            if(os.path.isdir(each_path)):
                #清除其它文件
                os.chdir(save_path)
                list_file = os.listdir()
                for each in list_file:
                    if('tif' not in each):
                        os.remove(each)
 
                write_prm_file = open(os.path.join(save_path,'MRT.prm'),'w')
                
                os.chdir(each_path)  
                list_modis = os.listdir()

                os.chdir(save_path)
                name = ''
                for modis_name in list_modis:
                    name = name+modis_name+' '
                input_name = r'D:\study\fusion test data\fusion_test_data\MRT\2007212-2007228\TmpMosaic.hdf'
                prm_context = '\
\n\
#The "INPUT_FILENAMES" field would be commented. If you want to load multiple input files please uncomment the "INPUT_FILENAMES" field and comment the"INPUT_FILENAME" field.\n\
#Also the "ORIG_SPECTRAL_SUBSET" field needs to be uncommented and changed to "SPECTRAL_SUBSET". The initial "SPECTRAL_SUBSET" field should be deleted.\n\
\n\
\n\
#INPUT_FILENAME = ( %s)\n\
\n\
INPUT_FILENAME = %s\n\
\n\
SPECTRAL_SUBSET = ( 1 )\n\
#ORIG_SPECTRAL_SUBSET = ( 0 0 1 0 0 )\n\
\n\
SPATIAL_SUBSET_TYPE = OUTPUT_PROJ_COORDS\n\
\n\
SPATIAL_SUBSET_UL_CORNER = ( %s %s )\n\
SPATIAL_SUBSET_LR_CORNER = ( %s %s )\n\
\n\
OUTPUT_FILENAME = D:\\study\\fusion test data\\fusion_test_data\\MRT\\2007212-2007228\\11.tif\n\
\n\
RESAMPLING_TYPE = NEAREST_NEIGHBOR\n\
\n\
OUTPUT_PROJECTION_TYPE = UTM\n\
\n\
OUTPUT_PROJECTION_PARAMETERS = ( \n\
0.0 0.0 0.0\n\
 0.0 0.0 0.0\n\
 0.0 0.0 0.0\n\
 0.0 0.0 0.0\n\
 0.0 0.0 0.0 )\n\
\n\
DATUM = WGS84\n\
\n\
UTM_ZONE = %d\n\
\n\
OUTPUT_PIXEL_SIZE = %d\n\
                '%(name,input_name,str(x_min),str(y_max),str(x_max),str(y_min),utm_zone,resolusion)

                write_prm_file.write(prm_context)
                    
                write_prm_file.close()
                
                #modis_temp_name = 'MOD09GQ.A'+os.path.split(each_path)[1]

                
                cmd = r'"%s\bin\java.exe" -jar "%s\bin\MRTBatch.jar" -d "%s" -p "%s\MRT.prm" -o "%s"'%(java_path,MRT_path,each_path,save_path,save_path)
                print(cmd)
                #直接用os.system(cmd)调用可能出错，调用失败
                temp_file = open('modis.bat','w')
                temp_file.writelines(cmd)
                temp_file.close()
                os.system('modis.bat')

                files = os.listdir()
                for file_temp in files:
                    if('mosaic.prm' in file_temp):
                        modis_temp_name = file_temp
                        break
                    
                print('正在转投影%s影像...'%modis_temp_name)
                if(modis_band_number == 1):
                    a = r'mrtmosaic -s "0 1 0" -i %s\%s -o %s\TmpMosaic.hdf'%(save_path,modis_temp_name,save_path)
                if(modis_band_number == 2):
                    a = r'mrtmosaic -s "0 0 1" -i %s\%s -o %s\TmpMosaic.hdf'%(save_path,modis_temp_name,save_path)
                temp_file = open('modis.bat','w')
                temp_file.writelines(a)
                temp_file.close()
                os.system('modis.bat')
                for file_temp in files:
                    if('resample.prm' in file_temp):
                        modis_temp_name = file_temp
                        break
                b = r'resample -p %s\%s'%(save_path,modis_temp_name)
                temp_file = open('modis.bat','w')
                temp_file.writelines(b)
                temp_file.close()
                os.system('modis.bat')
##                os.system(cmd)
##                print('正在进行%sMRT投影'%modis_temp_name)
##                os.system(a)
##                os.system(b)
#                temp_file = open('modis.bat','w')
#                temp_file.writelines(a+'\n')
#                temp_file.writelines(b)
                print(a)
                print(b)
#                temp_file.close()
#                os.system('modis.bat')
                
                print('%s影像转投影完成'%each_path)
        os.chdir(save_path)
        list_file = os.listdir()
        for each in list_file:
            if('tif' not in each):
                os.remove(each)
    
    def cal_mean(self,class_mean_difference,class_image,mean_difference_temp,total,length,nTheads,nBlocks,landsat_m,landsat_n):
        print('正在计算class_mean_difference')
         #class_mean_difference计算
        class_mean_difference_temp = class_mean_difference.reshape(1,total)
        class_image_temp = class_image.astype(numpy.int32).reshape(1,total)
        
        mod_mean = SourceModule("""
        #include<math.h>
        __global__ void func_mean(double *class_mean_difference,int *class_image,double *mean_difference,int total,int length)
        {
            const int i = blockIdx.x*blockDim.x+threadIdx.x;
            if(i>total){
                return;
            }
            class_mean_difference[i] = 1.64;
            if(class_image[i]<10)
            {

                class_mean_difference[i] = mean_difference[class_image[i]];//节省内存，class_image代表class_mean_difference
               // class_mean_difference[i] = 1.64;
                //tijk[i] = 1/abs(class_mean_difference[i])
            }
        }
        """
            )

        func_mean = mod_mean.get_function("func_mean")
        func_mean(
            drv.InOut(class_mean_difference_temp),drv.In(class_image_temp),drv.In(mean_difference_temp),total,length,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        class_mean_difference = class_mean_difference_temp.reshape(landsat_m,landsat_n)
        #tijk = tijk.reshape(landsat_m,landsat_n)
        return class_mean_difference
    
    #读取t1时刻landsat单波段影像与头文件
    def read_landsat(self,landsat_t1_path,band_number):
        os.chdir(landsat_t1_path)
        list_band_landsat = os.listdir()
        for landsat_file_name in list_band_landsat:
            #分离文件名与扩展名
            [file_name,file_expand] = os.path.splitext(landsat_file_name)
            #获得红或红外波段的影像
            if('B%d'%band_number in file_name and '.tif' == file_expand.lower()):
                print('读取%s'%landsat_file_name,"波段",band_number)
                landsat_image_t1 = self.read_image(landsat_file_name)
            #获得元数据的目录名字
            if('MTL' in file_name and '.txt'== file_expand.lower()):
                print('读取%s'%landsat_file_name)
                metadata_t1_filename = os.path.join(landsat_t1_path,landsat_file_name)

        return [landsat_image_t1,metadata_t1_filename]

     #以裁剪后landsat_t1时刻影像为基准裁剪
    def image_cut1(self,landsat_t1_image,cut_image,modis_re,land_re):
        '''
        landsat_t1_image,transform分别为DN值数组和地理信息
        cut_image为待裁剪影像，类型为gdal数据集
        '''
        landsat_t1_data = self.read_data(landsat_t1_image,1)
        transform = landsat_t1_image.GetGeoTransform()
        [m,n] = landsat_t1_data.shape
        xmin = transform[0]
        xmax = xmin + n*transform[1]
        ymax = transform[3]
        ymin = ymax + m*transform[5]
        print(xmin,xmax,ymin,ymax)

        cut_image_data = self.read_data(cut_image,1)
        cut_image_trans = cut_image.GetGeoTransform()
        [ci_m,ci_n] = cut_image_data.shape
        ci_xmin = cut_image_trans[0]
        ci_xmax = ci_xmin + ci_n*cut_image_trans[1]
        ci_ymax = cut_image_trans[3]
        ci_ymin = ci_ymax + ci_m*cut_image_trans[5]

        if(xmin<=ci_xmin):
            x_min = ci_xmin
        else:
            x_min = xmin

        if(ymin<=ci_ymin):
            y_min = ci_ymin
        else:
            y_min = ymin

        if(xmax<=ci_xmax):
            x_max = xmax
        else:
            x_max = ci_xmax

        if(ymax<=ci_ymax):
            y_max = ymax
        else:
            y_max = ci_ymax


        dx = x_max-x_min
        dy = y_max-y_min
        dx_8 = dx-dx%modis_re #取240的整倍数，使能够与modis对应
        dy_8 = dy-dy%modis_re
        x_max = x_min+dx_8
        y_max = y_min+dy_8
        print('11',dx,dy,dx_8,dy_8)
        if(int(dx_8%modis_re) is not 0 or int(dy_8%modis_re) is not 0):
            print('计算错误')
        print('影像范围',x_min,y_min,x_max,y_max)

        #共同的范围

        dx_min_1 = int((x_min-xmin)/land_re)
        dy_min_1 = int((y_min-ymin)/land_re)
        dx_max_1 = int((xmax-x_max)/land_re)
        dy_max_1 = int((ymax-y_max)/land_re)

        dx_min_n = int((x_min-ci_xmin)/land_re)
        dy_min_n = int((y_min-ci_ymin)/land_re)
        dx_max_n = int((ci_xmax-x_max)/land_re)
        dy_max_n = int((ci_ymax-y_max)/land_re)

        landsat_DN_t1 = landsat_t1_data[dy_max_1:(m-dy_min_1),dx_min_1:(n-dx_max_1)]
        landsat_DN_tn = cut_image_data[dy_max_n:(ci_m-dy_min_n),dx_min_n:(ci_n-dx_max_n)]
        
#        landsat_DN_t1 = landsat_t1_data[dx_min_1:(m-dx_max_1),dy_min_1:(n-dy_max_1)]
#        landsat_DN_tn = cut_image_data[dx_min_n:(ci_m-dx_max_n),dy_min_n:(ci_n-dy_max_n)]

        print('配准后landsatt1,tn时刻的影像大小:',landsat_DN_t1.shape,landsat_DN_tn.shape)
        return [[x_min,y_min,x_max,y_max],landsat_DN_t1,landsat_DN_tn]
    
    
    def save_image(self,array,drive,transform,projection,name,save_path):
        os.chdir(save_path)
        height,width = array.shape
        image = drive.Create(name,width,height,1,GDT_Float32)
        band = image.GetRasterBand(1)
        band.WriteArray(array)
        image.SetGeoTransform(transform)
        image.SetProjection(projection)
        image.FlushCache()


    #影像融合,参数：分类数，path依次表示landsat期初影像，landsat期末影像，时序的modis影像,band_number为第几波段,batch 是否批处理
    def fusion(self,landsat_t1_path,landsat_t2_path,MRT_save_path,modis_path_list,save_path,java_path,MRT_path,zhouqi = 16,landsat_resolusion = 30,modis_resolusion = 240,landsat_ID=5,block_size = 512,batch = False):
        if(batch):
            #设置根目录
            os.chdir(self.root)

        else:
            
            #读取t1时刻landsat影像，单波段执行
            landsat_image_t1,metadata_t1_filename = self.read_landsat(landsat_t1_path,self.band_number)
            #读取tn时刻landsat影像，单波段执行
            landsat_image_t2,metadata_t2_filename = self.read_landsat(landsat_t2_path,self.band_number)
            
            #landsat期初影像裁剪为8倍数,返回该影像的DN值与左下角与右上角坐标[x_min,y_min,x_max,y_max]
            [location,landsat_DN_t1,landsat_DN_t2] = self.image_cut1(landsat_image_t1,landsat_image_t2,modis_resolusion,landsat_resolusion)
            
            #计算影像的反射率
            print(landsat_DN_t1.max(),landsat_DN_t2.max())
            start_time = time.time()
            landsat_reflectance_t1 = self.cal_reflectance(metadata_t1_filename,landsat_DN_t1,self.band_number,landsat_ID)

            end_time = time.time()
            
            #print('抽查两个点的反射率',landsat_reflectance_t1[3000,3000],landsat_reflectance_t1[2000,3000])

            print('landsatt1影像反射率计算完成，所用时间:%0.2fs'%(end_time-start_time))
            #计算影像的反射率
            start_time = time.time()
            landsat_reflectance_t2 = self.cal_reflectance(metadata_t2_filename,landsat_DN_t2,self.band_number,landsat_ID)

            end_time = time.time()
            
            #print('抽查两个点的反射率',landsat_reflectance_t1[3000,3000],landsat_reflectance_t1[2000,3000])
            
            print('landsatt2影像反射率计算完成，所用时间:%0.2fs'%(end_time-start_time))

            
            m,n = landsat_reflectance_t1.shape
            diff = landsat_DN_t1.astype('int32') - landsat_DN_t2.astype('int32')

            a = Kmeansaddadd2d_sanjiao(self.class_count,self.max_iter,diff)
            
            class_image = a.kmeans() 

            print("landsat影像大小：",[m,n])
            print("分类影像nodata的类别：",self.class_count)
            print("landsat影像左下角坐标：",location[:2],'单位m')
            print("landsat影像右上角坐标：",location[2:],'单位m')
            
            #获得投影带号
            projection = landsat_image_t1.GetProjection()       
            utm_zone_index = projection.find('UTM zone')
            utm_zone = int(projection[utm_zone_index+9:utm_zone_index+11])
            
            print("landsatUTM投影带：",utm_zone)
            
            #红波段对应modis第一个影像，近红外对应第二个影像
            if(8 == landsat_ID):
                if (self.band_number == 4):
                    print('读取红波段影像')
                    modis_band_number = 1
                if(self.band_number == 5):
                    print('读取近红波段影像')
                    modis_band_number = 2
            else:
                if (self.band_number == 3):
                    print('读取红波段影像')
                    modis_band_number = 1
                if(self.band_number == 4):
                    print('读取近红波段影像')
                    modis_band_number = 2
            
            #MRT转投影
            start_time = time.time()
            #获取landsat影像的时间
            landsat_t1_time = os.path.split(landsat_t1_path)[1][9:16]
            time1_year = int(landsat_t1_time[0:4])  #期初影像的成像时间
            time1_day = int(landsat_t1_time[4:])
            time2_day = time1_day+zhouqi  #期末影像的成像时间,可超过365/366
            
            self.MRT(location,utm_zone,modis_resolusion,modis_path_list,modis_band_number,MRT_save_path,java_path,MRT_path)
            end_time = time.time()
            print('计算MRT转投影所用时间:%0.2fs'%(end_time-start_time))
            
            #提取丰度图，修改
            print('提取丰度图...')
            start_time = time.time()
            
            abundance_image_temp = self.abundance_extract(class_image,landsat_resolusion,modis_resolusion)
            abundance_image = numpy.array(abundance_image_temp)
            ##print(class_index[i][j])
            
            end_time = time.time()
            print('提取丰度图所用时间:%0.2fs'%(end_time-start_time))
                  
            
            os.chdir(MRT_save_path)
            list_dir = os.listdir()
            #读取期初landsat影像对应时间的modis影像
            start_time1 = time.time()
            for each in list_dir:
                if(str(time1_year)+str(time1_day) in each and 'sur_refl_b0%d'%modis_band_number in each and 'tif' == each[-3:].lower()):
                    print('读取%d时刻的modis影像%s'%(time1_day,each))
                    time1_modis_image = self.read_image(each)
                    time1_modis_reflectance = (self.read_data(time1_modis_image,1))/10000
                    print('正在计算%d时刻的类别平均反射率'%time1_day)
                            
                    time1_class_mean_reflectance = self.least_squares(abundance_image,time1_modis_reflectance)
                    if(not time1_class_mean_reflectance.any()):
                        print("%s影像本身无法计算类别平均反射率"%each)
                        print("请重新选择%d时间的MODIS影像"%time1_day)
                       # return 0
                    break
            print('计算类别平均反射率所用时间:%0.2fs'%(time.time()-start_time1))
            #开始计算融合反射率，参数初始化
            windows_size = 1
            #读取期初到期末时间内的modis影像
            
            #生成遥感影像数据集
            #transform and projection
            transform = (float(location[0]), landsat_resolusion, 0.0,float(location[3]),0.0 , -landsat_resolusion)
            proj = landsat_image_t1.GetProjection()
            
            #修改路径为MODIS储存路径
            os.chdir(MRT_save_path)
            list_dir = os.listdir()
            
            for each in list_dir:
                
                process = 0  #重置进度条为0
                
                os.chdir(MRT_save_path)
                landsat_reflectance = numpy.zeros((m,n))
                print('读取%d到%d时间内的modis影像'%(time1_day,time2_day))
                try:
                    modis_time2 = each[9:16]
                    modis_time2_year = int(modis_time2[0:4])
                    modis_time2_day = int(modis_time2[4:])
                    if(modis_time2_year>time1_year):
                        #闰年+366
                        if(time1_year%4):
                            modis_time2_day = modis_time2_day+365
                        else:
                            modis_time2_day = modis_time2_day+366
                        
                    print('modis成像时间:',str(modis_time2_year)+str(modis_time2_day))
                except:
                    print("出错")
                    continue
            ##            print(each,'MOD09GQ' in each,time1<int(each[9:16])<=time2 ,'sur_refl_b0%d'%modis_band_number in each ,'tif' == each[-3:].lower())
                if(time1_day<modis_time2_day<=time2_day and 'sur_refl_b0%d'%modis_band_number in each and 'tif' == each[-3:].lower()):
                  
            
                    start = time.time()
                   
                    print('读取%s时刻的modis影像%s'%(modis_time2,each))
                    times_modis_image = self.read_image(each)
                    times_modis_reflectance = (self.read_data(times_modis_image,1))/10000
            
                    #计算类别平均反射率
                    print('正在计算%s时刻的类别平均反射率'%modis_time2)
                    times_class_mean_reflectance = self.least_squares(abundance_image,times_modis_reflectance)
                    if(not times_class_mean_reflectance.any()):
                        print("%s影像质量不好，无法计算该影像类别平均反射率"%each)
                        continue
            
                    print('计算类别平均反射率所用时间:%0.2fs'%(time.time()-start))
                    landsat_m = m
                    landsat_n = n

                    #类别平均反射率之差
                    class_mean_difference = numpy.zeros((landsat_m,landsat_n))
                    mean_difference = times_class_mean_reflectance  - time1_class_mean_reflectance
            ##                print(mean_difference)
                             
                    total_pixel = landsat_m*landsat_n

                    mean_difference_temp = numpy.array(mean_difference)

                    total = numpy.int32(total_pixel)
                    length = numpy.int32(len(mean_difference))
                   # print(total,length)
            
                    nTheads = 256
                    nBlocks = int((total_pixel+nTheads-1)/nTheads)
                   
                    class_mean_difference = self.cal_mean(class_mean_difference,class_image,mean_difference_temp,total,length,nTheads,nBlocks,landsat_m,landsat_n)                    
                   
                    #分块处理,重叠三个像素（窗口大小减1）
                    b_m = int(m/(block_size-int(windows_size/2)))+1
                    b_n = int(n/(block_size-int(windows_size/2)))+1
                    over_size = windows_size-1 #重叠区域
                    half_size = int(over_size/2)  #去除黑边区域
                    block_landsat = [[0 for i in range(b_n)] for i in range(b_m)] #储存分块后影像
                    position_landsat = [[0 for i in range(b_n)] for i in range(b_m)]
                    drive = landsat_image_t1.GetDriver()
                    projection = landsat_image_t1.GetProjection()
                    #self.save_image(class_image,drive,transform,projection,'11.tif',save_path)
                    #self.save_image(class_mean_difference,drive,transform,projection,'22.tif',save_path)
                   #` self.save_image(landsat_reflectance_t1,drive,transform,projection,'landsat_reflectance_t1.tif',save_path)
                    #self.save_image(landsat_reflectance_t2,drive,transform,projection,'landsat_reflectance_t2.tif',save_path)
                    fusion_time = time.time()
                    for bi in range(b_m):
                        for bj in range(b_n):
                            if(bi < b_m-1 and bj < b_n-1):
                                #block_landsat[bi][bj] = landsat_reflectance_t1[bi*block_size:(bi+1)*block_size+3,bj*block_size:(bj+1)*block_size+3]
                                block_landsat[bi][bj] = self.GPUcal_pixel_ref(landsat_reflectance_t1[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:(bj+1)*block_size+over_size],
                                                                class_mean_difference[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:(bj+1)*block_size+over_size])
                                size_x,size_y = block_landsat[bi][bj].shape  
                                block_landsat[bi][bj] = block_landsat[bi][bj][half_size:size_x-half_size,half_size:size_y-half_size]
                                position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)
                                #self.save_image(block_landsat[bi][bj],drive,position_landsat[bi][bj],projection,'%d%d.tif'%(bi,bj),save_path)
                               # print(position_landsat[bi][bj])
                                #print(1,bi,bj,bi*block_size,(bi+1)*block_size+over_size,bj*block_size,(bj+1)*block_size+over_size,block_landsat[bi][bj].shape)
                            if(bi == b_m-1 and bj < b_n-1):
                                block_landsat[bi][bj] = self.GPUcal_pixel_ref(landsat_reflectance_t1[bi*block_size:,bj*block_size:(bj+1)*block_size+over_size],
                                                                class_mean_difference[bi*block_size:,bj*block_size:(bj+1)*block_size+over_size])
                                size_x,size_y = block_landsat[bi][bj].shape  
                                block_landsat[bi][bj] = block_landsat[bi][bj][half_size:size_x-half_size,half_size:size_y-half_size]
                                position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)
                                os.chdir(save_path)
                                #self.save_image(block_landsat[bi][bj],drive,position_landsat[bi][bj],projection,'%d%d.tif'%(bi,bj),save_path)
                               # print(position_landsat[bi][bj])
                               # print(2,bi,bj,bi*block_size,landsat_reflectance_t1.shape[0],bj*block_size,(bj+1)*block_size+over_size,block_landsat[bi][bj].shape)
                            if(bi < b_m-1 and bj == b_n-1):
                                block_landsat[bi][bj] = self.GPUcal_pixel_ref(landsat_reflectance_t1[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:],
                                                                class_mean_difference[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:])
                                size_x,size_y = block_landsat[bi][bj].shape  
                                block_landsat[bi][bj] = block_landsat[bi][bj][half_size:size_x-half_size,half_size:size_y-half_size]
                                position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)
                                os.chdir(save_path)
                                #self.save_image(block_landsat[bi][bj],drive,position_landsat[bi][bj],projection,'%d%d.tif'%(bi,bj),save_path)
                               # print(position_landsat[bi][bj])
                                #print(3,bi,bj,bi*block_size,(bi+1)*block_size+over_size,bj*block_size,landsat_reflectance_t1.shape[1],block_landsat[bi][bj].shape)
                            if(bi == b_m-1 and bj == b_n-1):
                                block_landsat[bi][bj] = self.GPUcal_pixel_ref(landsat_reflectance_t1[bi*block_size:,bj*block_size:],
                                                                class_mean_difference[bi*block_size:,bj*block_size:])
                                size_x,size_y = block_landsat[bi][bj].shape  
                                block_landsat[bi][bj] = block_landsat[bi][bj][half_size:size_x-half_size,half_size:size_y-half_size]
                                position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)
                                os.chdir(save_path)
                                #self.save_image(block_landsat[bi][bj],drive,position_landsat[bi][bj],projection,'%d%d.tif'%(bi,bj),save_path)
                                #print(position_landsat[bi][bj])
                               # print(4,bi,bj,bi*block_size,landsat_reflectance_t1.shape[0],bj*block_size,landsat_reflectance_t1.shape[1],block_landsat[bi][bj].shape)        
                     
                    #拼接,去除重叠部分，中间部分去掉边缘三个像素
                    print('融合时间：',time.time()-fusion_time)
                    new_landsat = numpy.zeros((block_size,0)) #初始化空白矩阵
                    block_landsat_new = [new_landsat for i in range(b_m)] #初始化横向拼接矩阵，储存横向拼接后影像，大小匹配
                    block_landsat_new[b_m-1] = numpy.zeros((block_landsat[b_m-1][0].shape[0],0))
                    for bi in range(b_m):            
                        for bj in range(b_n):
                            block_landsat_new[bi] = numpy.hstack((block_landsat_new[bi],block_landsat[bi][bj]))
                    for bi in range(1,b_m):
                        block_landsat_new[0] = numpy.vstack((block_landsat_new[0],block_landsat_new[bi]))
                    #position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)    
                    print(position_landsat[0][0],transform)
                    self.save_image(block_landsat_new[0],drive,position_landsat[0][0],projection,'STDFA_%s_b%d.tif'%(modis_time2,modis_band_number),save_path)
                    
                #      block_size = 512
              #      new_land = self.GPUcal_pixel_ref(landsat_reflectance_t1[0:block_size,0:block_size],class_image[0:block_size,0:block_size],class_mean_difference[0:block_size,0:block_size],weight_st[0:block_size,0:block_size])    


                    end = time.time()
                    print('计算总时长：',end-start)

#三角不等式加速
class Kmeansaddadd2d_sanjiao:
    #初始化聚类数，最大迭代数，二维距离
    def __init__(self,class_count,max_iteration,data1):
        self.class_count = class_count #初始化聚类数
        self.max_iteration = max_iteration #最大迭代数
        self.data1 = data1-data1.min()+1  #数据转化为正值
        print('kmeans聚类初始化...')
        print('聚类数：%d'%class_count)
        print('最大迭代数：%d'%max_iteration)
    
    #距离计算    
    def cal_distance(self,point1,point2):
        dim = len(point1)
        distance = 0
        for i in range(dim):
            distance = (float(point1[i])-float(point2[i]))**2+distance
        return distance
    #计算种子点
    def cal_center(self,data1,class_count):
        center = []
        start_time1 = time.time()
        #height,width = self.data[0].shape
#        data1 = numpy.random.randint(1,255,(7500,7000)).astype('int32')
#        data2 = numpy.random.randint(1,255,(7500,7000)).astype('int32')
        data1 = data1.astype('int32')

        #class_count = 10

        height,width = data1.shape
        #distance = numpy.array([0,0,0,0,0,0,0,0,0,0]).astype('float64')
        #data = [data1 for _ in range(3)]
        #初始化第一个种子点
        while True:
            m = random.randint(0,height-1)
            n = random.randint(0,width-1)
            state = True
#            for i in range(self.dim):
#                if(not self.data[i][m,n]):

            if(not data1[m,n]):
                state = False
            if(state):
                break
        
        mod_cal_dis = SourceModule(
                '''
                __global__ void cal_distance(int *data1,float *dis,double *center1,int total,int point){
                    const int count = blockIdx.x*blockDim.x+threadIdx.x;
                    if(count>total){return;}
                    if(data1[count]<0.0001){
                        dis[count] = 0;
                        return;
                    }
                    for(int i = 0;i<point;i++){
                        double temp = (data1[count]-center1[i])*(data1[count]-center1[i]);
                        if(temp<dis[count]){dis[count] = temp;}
                    }
                    
                    return;
                    
                }                   
                                   
                ''')
        cal_dis = mod_cal_dis.get_function("cal_distance")
        distance = numpy.ones((height*width)).astype('float64')*sys.maxsize #初始化距离
        total = numpy.int32(height*width)
        center1 = numpy.array([data1[m,n]])

        data1 = data1.reshape(1,total)

        nTheads = 256
        nBlocks = int((total+nTheads-1)/nTheads)
#        data1 = self.data1.reshape(1,total)
#        data2 = self.data2.reshape(1,total)
        #for i in range(self.class_count):
        for i in range(class_count-1):
            #args = args+'double *data%d,'%i
            #print(distance)
            #print(data[i])
            print('正在计算第%d个种子点'%(i+2))
            point = numpy.int32(i+1)
            cal_dis(drv.In(data1),drv.InOut(distance),drv.In(center1),total,point,
                    block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
            sum_dis = int(sum(distance))
            #print(sum_dis)
            #print(data1)
            #print(1,center1,center2)
            sum_dis_rand = random.randint(0,sum_dis)
            for j in range(total):
                sum_dis_rand -=distance[j]
                if(sum_dis_rand<0):
                    #j即为新的种子点
                    if(data1[0,j]):
                        center1 = center1.tolist()                    
                        center1.append(data1[0,j])
                        center1 = numpy.array(center1)

                   # print(2,center1,center2)
                        break
        print(center1)

        print('种子点计算完成')
        print('计算种子点所用时间:%0.2fs'%(time.time()-start_time1))
       

        return center1
     #计算类间最小距离
    def cal_dis_center(self,center1):
        print('计算类间最小距离')   
        class_count = self.class_count
        #class_count = center1.shape[0]
        distance_center = []
        min_dis = []
        for i in range(class_count):
            min_distance = sys.maxsize
            for j in range(class_count):
                if(i!=j):
                    distance_class = self.cal_distance([center1[i]],[center1[j]])
                    if(distance_class<min_distance):
                        min_distance = distance_class
                        class_index = j   #min_distance(Ci,Cj)
            min_dis.append(min_distance)
            distance_center.append(class_index)  #记载最小距离以及相应的类别数
        #print(distance_center,min_dis)
        min_dis = numpy.array(min_dis).astype('float64')
        distance_center = numpy.array(distance_center).astype('int32')  
        
        return min_dis,distance_center #分别为最小距离以及与哪一个中心点
    def kmeans(self):
        print('初始化聚类中心')
        data1 = self.data1

        #print(data1.shape)
        class_count = self.class_count
        max_iter = self.max_iteration
        center1 = self.cal_center(data1,class_count)
        
        min_dis,distance_center = self.cal_dis_center(center1)

        center1 = center1.astype('float64')

        
        #min_dis,distance_center = self.cal_dis_center(center1,center2)
        #计算
        mod_cal_dis2 = SourceModule(
            '''
            #define CENTER_POINT %d
            #define MINDIS %d
            __global__ void cal_distance2(int *data1,int *class_image,int *new_data1,double *center1,double *min_dis,int *center_dis,int total){
                const int count = blockIdx.x*blockDim.x+threadIdx.x;
                
                if(count>total){
                    return;
                }
                
                if(data1[count]<0.0001){
                    //class_image[count] = CENTER_POINT;
                    return;
                }
                double dis1 = MINDIS;
                //class_image[count] = 1;
                //return;
                
                int pre_class = class_image[count];   
                int i = 0;
                double temp_dis = 0;
                //如果点已分配到某一类中,不等式两边同时平方
                if(pre_class < 10){
                    double temp = (data1[count]-center1[pre_class])*(data1[count]-center1[pre_class]);
                    if(4*temp>min_dis[pre_class]){ 
                        dis1 = MINDIS;
                        for(int i = 0;i<CENTER_POINT;i++){
                            double temp_dis = (data1[count]-center1[i])*(data1[count]-center1[i]);
                            //new_data1[i*total+count] = min_dis[i];
                            if(temp_dis<dis1){ 
                                class_image[count] = i;
                                
                                dis1 = temp_dis;
                            }
                        }
                    }
                }
                //如果点未分配到某一类中
                else{
                    double dis1 = MINDIS;

                    for(int i = 0;i<CENTER_POINT;i++){
                        double temp = (data1[count]-center1[i])*(data1[count]-center1[i]);
                        if(temp<dis1){
                            class_image[count] = i;
                            
                            dis1 = temp;
                        }
    
                    }
                }  
                    
                int class_temp = int(class_image[count]);
                new_data1[class_temp*total+count] = data1[count];

                return;
                
            }                   
                               
            '''%(class_count,sys.maxsize))
        #距离计算
        height,width = data1.shape
        total = height*width
        data1 = data1.reshape(1,total)

        total = numpy.int32(height*width)
        class_image = (numpy.ones(total)*class_count).astype('int32')
        center1_temp = numpy.zeros(class_count)
        block_size = 256 #计算多少行
        b_m = height//block_size
        cal_dis = mod_cal_dis2.get_function('cal_distance2')
        nTheads = 256
        nBlocks = int((total+nTheads-1)/nTheads)

        for i in range(max_iter):
            juleitime = time.time()
            print('第%d次聚类'%(i+1))
            print(center1.astype('int32'))
   
            #print(class_image[:10])
            class_image_new = numpy.zeros(0)
            #cal_dis = mod_cal_dis2.get_function('cal_distance')
            #分块处理
            total = block_size*width
            center1_temp = numpy.zeros(class_count)
    
            for m in range(b_m):
                if(m<b_m-1):                    
                    new_data1 = numpy.zeros(total*class_count).astype('int32')
                    
                    data1_temp = data1[0][m*total:(m+1)*total].astype('int32')
                  
                    class_image_temp = class_image[m*total:(m+1)*total].astype('int32')
                    nTheads = 256
                    nBlocks = int((total+nTheads-1)/nTheads)
                    cal_dis(drv.In(data1_temp),drv.InOut(class_image_temp),
                        drv.InOut(new_data1),drv.In(center1.astype('float64')),
                        drv.In(min_dis),drv.In(distance_center),numpy.int32(total), 
                        block = (nTheads,1,1),grid = (nBlocks,1))
             #调整聚类中心
                    for j in range(class_count):
                        temp_data1 = new_data1[j*total:(j+1)*total].astype('float64')
                        
                        #print(sum(new_data1),sum(new_data2),sum(data1_temp),sum(data2_temp),center1_temp[j],sum(temp_data1),numpy.sum(temp_data1>0),sum(temp_data1)/numpy.sum(temp_data1>0))  
                        if(center1_temp[j]):
                            if(numpy.sum(temp_data1>0)):
                                center1_temp[j] = (center1_temp[j]+sum(temp_data1)/numpy.sum(temp_data1>0))/2
                                
                                #print(1,center1_temp[j],center2_temp[j])
                        else:
                            if(numpy.sum(temp_data1>0)):
                                center1_temp[j] = sum(temp_data1)/numpy.sum(temp_data1>0)
                                
                                #print(2,center1_temp[j],center2_temp[j])
                            else:
                                center1_temp[j] = 0
                                
                                #print(3,center1_temp[j],center2_temp[j])
                        #print(center1_temp[j],center2_temp[j])
                    #print(class_image_temp[:10])
                    class_image_new = numpy.hstack((class_image_new,class_image_temp)).astype('int32')
                else:
                    
                    data1_temp = data1[0][m*total:].astype('int32')
                    
                    class_image_temp = class_image[m*total:].astype('int32')
                    total = data1_temp.shape[0]
                    new_data1 = numpy.zeros(total*class_count).astype('int32')
                    
                    nTheads = 256
                    nBlocks = int((total+nTheads-1)/nTheads)
                    cal_dis(drv.In(data1_temp),drv.InOut(class_image_temp),
                        drv.InOut(new_data1),drv.In(center1.astype('float64')),
                        drv.In(min_dis),drv.In(distance_center),numpy.int32(total), 
                        block = (nTheads,1,1),grid = (nBlocks,1))
                    for j in range(class_count):
                        temp_data1 = new_data1[j*total:(j+1)*total].astype('float64')
                        
                        #print(center1_temp[j],sum(temp_data1),numpy.sum(temp_data1>0),sum(temp_data1)/numpy.sum(temp_data1>0)) 
                        if(numpy.sum(temp_data1>0)):
                            center1_temp[j] = (center1_temp[j]+sum(temp_data1)/numpy.sum(temp_data1>0))/2
                            
                    class_image_new = numpy.hstack((class_image_new,class_image_temp)).astype('int32')
            #print(center1_temp)
            #print(center2_temp)
            class_image = copy.deepcopy(class_image_new).astype('int32')
            if(numpy.sum(abs(center1-center1_temp)<1)>=class_count):
                break
            center1 = copy.deepcopy(center1_temp)
  
            #重新计算类间距离
            min_dis,distance_center = self.cal_dis_center(center1)
            print('第%d次聚类'%(i+1),'所用时间:%0.2f'%(time.time()-juleitime))

        print('聚类完成！')   
        return class_image.reshape((height,width))


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root',type=str,help = '根路径，必选')
parser.add_argument('--landsat_t1_path', type=str,help = 'landsat初期影像解压缩后文件夹路径，必选')
parser.add_argument('--landsat_t2_path', type=str,help = 'landsat初期影像解压缩后文件夹路径，必选')
parser.add_argument('--modis_path', type=str,help = 'MODIS影像文件夹路径,必选')
parser.add_argument('--MRT_save_path', type=str,help = 'MODIS转投影、裁剪与重采样后保存路径，必选')
parser.add_argument('--save_path', type=str,help = '合成影像保存路径，必选')
parser.add_argument('--java_path', type=str,help = 'Java安装路径，例如C:\Program Files\Java\jre1.8.0_144，必选')
parser.add_argument('--MRT_path', type=str,help = 'MRT安装路径，例如D:\Program Files\MRT，必选')
parser.add_argument('--landsat_ID', type=int,help = 'Landsat卫星型号，可选5,7,8，必选')
parser.add_argument('--band_number', type=int,help = '波段选择，Landsat5和7可选3,4，3为红波段，4为近红外波段，Landsat8可选4,5，4为红波段，5为近红外波段,，必选')
parser.add_argument('--zhouqi', type=int,default=16,help = 'Landsat间隔周期，可选，默认16')
parser.add_argument('--landsat_resolusion', type=int,default=30,help = 'Landsat分辨率，可选，默认30')
parser.add_argument('--modis_resolusion', type=int,default=240,help = 'MODIS重采样分辨率，可选，默认240')
parser.add_argument('--windows_size', type=int,default=15,help = '窗口大小，可选，默认15')
parser.add_argument('--class_count', type=int,default=512,help = '分类数，可选，默认10')
parser.add_argument('--max_iter', type=int,default=512,help = '最大迭代数，可选，默认10')
parser.add_argument('--block_size', type=int,default=512,help = '影像分块大小，可选，默认512')
args = parser.parse_args()
print(args)
os.chdir(args.modis_path)
modis_list = os.listdir()
modis_path_list = []
for each in modis_list:
    os.chdir(each)
    temp_path = os.getcwd()
    modis_each_path = os.listdir()
    for modis_each in modis_each_path:
        
        modis_path_list.append(os.path.join(temp_path,modis_each))
    os.chdir('..')
fusion = Fusion(args.root,class_count = args.class_count,max_iter = args.max_iter,band_number = args.band_number,window_size = args.windows_size)
time2 = time.time()
batch = False
fusion.fusion(args.landsat_t1_path,args.landsat_t2_path,args.MRT_save_path,modis_path_list,args.save_path,args.java_path,args.MRT_path,args.zhouqi,args.landsat_resolusion,args.modis_resolusion,args.landsat_ID,args.block_size,batch)
print('融合所用时间：%0.2f'%(time.time()-time2))










    
