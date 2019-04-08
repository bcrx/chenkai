'''
2018.6.10
version 1.0
CDSTARFM时空数据融合GPU并行加速程序
作者：西南交通大学地球科学与环境工程学院陈凯
'''

import gdal
import numpy
import sys
from gdalconst import *
from osr import SpatialReference
import struct
import copy
import os
import shutil
import math
import time
##import matplotlib.pyplot as pyplot
import threading
import gc #内存管理
import random

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


class Fusion:
    #初始化
    def __init__(self,root,class_count = 10,band_number = 3,window_size = 7):
        #设置根目录
        self.root = root
        self.class_count = class_count
        self.band_number = band_number
        self.window_size = window_size
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

#取回数据，不需要用inout，inout为创建内存副本
#        func = mod.get_function("doublify")
#        func(a_gpu, block=(4,4,1))
#        a_doubled = numpy.empty_like(reflectance)
#        cuda.memcpy_dtoh(a_doubled, reflectance)
#        print(a_doubled)
#        print(reflectance)
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
    def GPUcal_pixel_ref(self,landsat_reflect_ti,class_image,class_mean_difference,mul_st,landsat_resolution):
        block_size_x,block_size_y = landsat_reflect_ti.shape
        total_pixel = block_size_x*block_size_y
        nTheads = 256
        nBlocks = int((total_pixel+nTheads-1)/nTheads)
        #print(total_pixel,nBlocks)
        time1 = time.time()
        mod = SourceModule('''
            //编写c++核函数
            #define BLOCK_SIZE_X %d
            #define BLOCK_SIZE_Y %d
            #define WINDOW_SIZE %d
            #define LR %d
            //C++内无法调用math里面的函数（__host__函数），python运行GPU程序有时候可以，有时候不行，原因不明
            __device__ double sum(double *arr, int m, int n);//求和函数
            __device__ double new_sqrt(double a);//开方，
            __device__ double new_abs(double a);//绝对值
 
            __global__ void cal_ref(double *land_ref, int *class_image, double *class_mean_diff, double *mul_st, const int windows_size,double *new_landsat)
            {
            	int count = blockIdx.x*blockDim.x + threadIdx.x;
            	
            	if (count > BLOCK_SIZE_X*BLOCK_SIZE_Y) {
            		return;
            	}
             if (land_ref[count] <0.00001) {
                  new_landsat[count] = 0;
            		return;
            	}
            	
            	int total_size = windows_size*windows_size;
            
            	//计算行列号
            	//int *ij;
            	//ij = cal_ij(count, BLOCK_SIZE, BLOCK_SIZE);
            	int i, j;
            	i = int(count / BLOCK_SIZE_Y);
            	j = count - i * BLOCK_SIZE_Y;
            	int center = int(windows_size / 2);
            	//new_landsat[count] = land_ref[count];
            	//return;
            	if (i < center||i>(BLOCK_SIZE_X - center-1)||j<center||j>(BLOCK_SIZE_Y - center-1)) {
                new_landsat[count] = 0;
          		  return;
            	}
              
            	int arr_ij[WINDOW_SIZE*WINDOW_SIZE] = { 0 };//正确
            	for (int k = 0; k < windows_size; k++) {
            		for (int l = 0; l < windows_size; l++) {
            			arr_ij[k*windows_size + l] = (k+i- center)*BLOCK_SIZE_Y + (l+j- center);
            		}
            	}
            	
            	//double sum_pixel = sum(land_ref7);
            	
            	
            	//提取数组
            	double land_ref7[WINDOW_SIZE*WINDOW_SIZE] = { 0 };
            	double mul_st7[WINDOW_SIZE*WINDOW_SIZE] = { 0 };
            	double class_mean_diff7[WINDOW_SIZE*WINDOW_SIZE] = { 0 };
            	int class_image7[WINDOW_SIZE*WINDOW_SIZE] = { 0 };
            	for (int k = 0; k < WINDOW_SIZE*WINDOW_SIZE; k++) {
            		land_ref7[k] = land_ref[arr_ij[k]];
            		class_image7[k] = class_image[arr_ij[k]];
            		mul_st7[k] = mul_st[arr_ij[k]];
            		class_mean_diff7[k] = class_mean_diff[arr_ij[k]];
            	}
            	//求和
            	double sum_pixel = 0.0;//正确
            	//for (int k = 0; k < windows_size*windows_size; k++) {
            	//	sum_pixel = sum_pixel + land_ref[arr_ij[k]];
            	//}
            	sum_pixel = sum(land_ref7, WINDOW_SIZE, WINDOW_SIZE);
            	//求平均
            	double mean_pixel = sum_pixel / (windows_size*windows_size);//正确
            	
            	//类别判断
            	//sort(land_ref7, land_ref7+ WINDOW_SIZE*WINDOW_SIZE);
              int temp[%d] = {0};
              double length = 0.0;
              sum_pixel = 0.0;//正确
              for (int k = 0; k < windows_size*windows_size; k++) {
                  temp[class_image7[k]] = 1;
                  sum_pixel = sum_pixel+(land_ref7[k]-mean_pixel)*(land_ref7[k]-mean_pixel);
                  
              }
            	for (int l = 0; l < %d; l++) {
                  if(temp[l] == 1){length+=1;}    
              }
              //new_landsat[count] = length;
              //return;
            	//阈值
            	//sum_pixel = sum_pixel-mean_pixel*total_size;
            	double theta_b = new_sqrt(1.0 / (WINDOW_SIZE*WINDOW_SIZE)*sum_pixel) / length;//正确，阈值计算公式检查
            	
            	//权重
            	double weight_temp = 0;
            	for (int k = 0; k < WINDOW_SIZE*WINDOW_SIZE; k++) {
            		weight_temp = 0;
            		if (new_abs(land_ref7[k] - land_ref[count]) <= theta_b) {
            			int i = k / WINDOW_SIZE;
            			int j = k - i*WINDOW_SIZE;
            			weight_temp = 1 / (1 + new_sqrt((center - i)*(center - i)*LR*LR + (center - j)*(center - j)*LR*LR) / 750);
            			mul_st7[k] = mul_st7[k] * weight_temp;
            		}
            		else {
            			mul_st7[k] = 0;
            		}
            	}
            	//权重求和
            	double sum_weight = sum(mul_st7, WINDOW_SIZE, WINDOW_SIZE);
            	
            	double land_ref_new7[WINDOW_SIZE*WINDOW_SIZE] = { 0 };
            	//计算反射率
            	for (int k = 0; k < WINDOW_SIZE*WINDOW_SIZE; k++) {
            		if (mul_st7[k] != 0.0 && land_ref7[k] != 0) {
            			double temp_w = mul_st7[k]/sum_weight;
            			double temp_ref = class_mean_diff7[k] + land_ref7[k];
            			if (1 > temp_ref && temp_ref>0) {
            				land_ref_new7[k] = temp_w*temp_ref;
            			}
            			else {
            				land_ref_new7[k] = temp_w*land_ref7[k];
            			}
            
            		}
                  
            	}
            	new_landsat[count] = sum(land_ref_new7, WINDOW_SIZE, WINDOW_SIZE);
              if(new_landsat[count] == 0){new_landsat[count] = class_mean_diff[count] + land_ref[count];} //为了不出现黑点
            	
            }
            
            
            //开方函数
            __device__ double new_sqrt(double a) {
            	double x, y;
            	x = 0.0;
            	y = a / 2;
            	while (x != y)
            	{
            		x = y;
            		y = (x + a / x) / 2;
            	}
            	return x;
            }
            //绝对值
            __device__ double new_abs(double a) {
            	if (a >= 0) {
            		return a;
            	}
            	else {
            		return -a;
            	}	
            }
            //一维数组求和
            __device__ double sum(double *arr, int m, int n) {
            	double sum_arr = 0;
            	for (int i = 0; i < m*n; i++) {
            		sum_arr += arr[i];
            	}
            	return sum_arr;
            }
                           '''%(block_size_x,block_size_y,self.window_size,landsat_resolution,self.class_count,self.class_count))

        new_ref = numpy.zeros((block_size_x,block_size_y))

        func_ref = mod.get_function("cal_ref")

        land_ref= landsat_reflect_ti.reshape(1,block_size_x*block_size_y).astype('float64')
 
        class_mean_difference= class_mean_difference.reshape(1,block_size_x*block_size_y).astype('float64')

        class_image = class_image.reshape(1,block_size_x*block_size_y).astype('int32')
        #print(land_ref[0,9000:9050])

        mul_st = mul_st.reshape(1,block_size_x*block_size_y).astype('float64')

        new_landsat = new_ref.reshape(1,block_size_x*block_size_y).astype('float64')

        func_ref(
            drv.In(land_ref),drv.In(class_image),drv.In(class_mean_difference),drv.In(mul_st),numpy.int32(self.window_size),drv.InOut(new_landsat),
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        new_ref = new_landsat.reshape(block_size_x,block_size_y)
        #print(new_landsat[0,9000:9050])
        return new_ref
        

        
    #像元反射率计算r(c,tj) = mean_r(c,tj) - mean_r(c,ti) +r(c,ti)
    def cal_pixel_reflectance(self,landsat_reflect_ti,class_image,class_mean_difference,mul_st):
##        print(landsat_reflect_ti,type(landsat_reflect_ti[0,0]))
        #窗口大小
        [m,n] = landsat_reflect_ti.shape  #24*24
        #总像元数
        total_pixel = m*n
        sum_variance = 0.0
        
        sum_pixel = sum(sum(landsat_reflect_ti))
        #窗口内平均像元值
        mean_pixel = sum_pixel/total_pixel
        
        #中心像元
        center = int(m/2)
        center_pixel = landsat_reflect_ti[center,center]
        
        for i in range(m):
            for j in range(n):
                sum_variance = sum_variance +(landsat_reflect_ti[i,j]-mean_pixel)**2

        #类别数
        temp = []
        for i in range(m):
            for j in range(n):
                if(class_image[i,j] not in temp):
                    temp.append(class_image[i,j])
        length = len(temp)
##        print(class_mean_reflect_ti,length,sum_variance)
        if(length):
            theta_b = math.sqrt(1/total_pixel*sum_variance)/length
        else:
            return center_pixel


##        print(mul_st,theta_b)

        #权重
        weight = numpy.zeros((m,n))

        for i in range(m):
            for j in range(n):
                weight_st_temp = mul_st[i,j]
                if(math.fabs(landsat_reflect_ti[i,j]-center_pixel)<=theta_b):
                    weight_st = weight_st_temp
                    weight_dijk = 1/(1 + math.sqrt((center-i)**2+(center-j)**2)/750)
                    weight[i,j] = weight_st*weight_dijk

##        wijk = numpy.zeros((m,n))
        reflectance_ij = numpy.zeros((m,n))
        
        weight_sum = sum(sum(weight))
##                print(weight_sum)
        if(weight_sum == numpy.inf):
            print('11111')
            ref = []
            array = numpy.where(weight == numpy.inf)
            for i in range(len(array[0])):
                ref.append(landsat_reflect_ti[array[0][i],array[1][i]])
            reflectance = sum(ref)/len(ref)
            print(reflectance)
        else:
            for i in range(m):
                for j in range(n):
                    if(math.fabs(landsat_reflect_ti[i,j]-center_pixel)<=theta_b):
                        #测试
                        try:
##                                    print(weight[i,j])
                            wijk = float(weight[i,j])/float(weight_sum)
                        except:
                            
                            print(weight[i,j],mul_st[i,j],weight)

                        temp_ref = class_mean_difference[i,j] + landsat_reflect_ti[i,j]
    ##                    print(wijk,class_mean_reflect_tj[class_image[i,j]],landsat_reflect_ti[i,j],class_mean_reflect_ti[class_image[i,j]])
                        #去除异常点
                        if(1>temp_ref>0):
##                                    print('fanshelv',temp_ref)
                            reflectance_ij[i,j] = wijk*temp_ref
                        else:
                            reflectance_ij[i,j] = wijk*landsat_reflect_ti[i,j]
                        

            reflectance = sum(sum(reflectance_ij))
##                print(reflectance[k,h])
##        
        return reflectance

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

    #读取分类影像，裁剪成与landsat一致大小
    def read_classimage(self,classimage_path,X,Y):
        class_image = self.read_image(classimage_path)
        class_data = self.read_data(class_image,1)
        class_data = class_data[0:X,0:Y]
        class_data = class_data.astype('uint8')
        return class_data
    
    def cal_sijk(self,sijk,landsat_reflectance_t1,class_image,time1_class_mean_reflectance,total,length,nTheads,nBlocks,landsat_m,landsat_n):
        print('正在计算sijk')
        #sijk计算
        landsat_r = landsat_reflectance_t1.reshape(1,total).astype('float64')
        class_image_temp = class_image.astype('int32').reshape(1,total)
        sijk_temp = sijk.reshape(1,total).astype('float64')
        time1_class_mean_reflectance = time1_class_mean_reflectance.astype('float64')
        mod_sijk = SourceModule("""
        #include<math.h>
        __global__ void func(double *sijk_temp,double *landsat_r,int *class_image,double *time1_class_mean_reflectance,int total,int length)
        {
            const int i = blockIdx.x*blockDim.x+threadIdx.x;
            if(i>total){
                return;
            }
            if(class_image[i]<length)
            {
                float temp = abs(landsat_r[i] - time1_class_mean_reflectance[class_image[i]]);
                if(temp){sijk_temp[i] = 1/temp;}
                
            }
        }
        """
            )
        func_sijk = mod_sijk.get_function("func")
        func_sijk(
            drv.InOut(sijk_temp),drv.In(landsat_r),drv.In(class_image_temp),drv.In(time1_class_mean_reflectance),total,length,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        sijk = sijk_temp.reshape(landsat_m,landsat_n)
        return sijk
    
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
    
    def cal_tijk(self,tijk,class_mean_difference,total,length,nTheads,nBlocks,landsat_m,landsat_n):
        print('正在计算tijk')
        #weight_ijk计算
        #不能调用host函数，只能调用device
        tijk_temp = tijk.reshape(1,total)
        class_mean_difference_temp = class_mean_difference.reshape(1,total)
        mod_tijk = SourceModule("""
        #include<math.h>
        //绝对值
        __device__ double new_abs(double a);
        __device__ double new_abs(double a) {
        	if (a >= 0) {
        		return a;
        	}
        	else {
        		return -a;
        	}	
        }
        __global__ void func_tijk(double *tijk,double *class_mean_difference,int total)
        {
            const int i = blockIdx.x*blockDim.x+threadIdx.x;
            if(i>total){
                return;
            }
            if(class_mean_difference[i])
            {
                tijk[i] = 1/new_abs(class_mean_difference[i]);//节省内存，class_mean_difference代表tijk
            }
        }
        """
            )
        func_tijk = mod_tijk.get_function("func_tijk")
        func_tijk(drv.InOut(tijk_temp),drv.In(class_mean_difference_temp),total,
                  block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
        tijk = tijk_temp.reshape(landsat_m,landsat_n)
        return tijk
    
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
    def fusion(self,landsat_t1_path,classimage_path,MRT_save_path,modis_path_list,save_path,java_path,MRT_path,zhouqi = 16,landsat_resolusion = 30,modis_resolusion = 240,landsat_ID=5,block_size = 512,batch = False):
        if(batch):
            #设置根目录
            os.chdir(self.root)

        else:
            windows_size = self.window_size
            #读取t1时刻landsat影像，单波段执行
            landsat_image_t1,metadata_t1_filename = self.read_landsat(landsat_t1_path,self.band_number)
            
            #landsat期初影像裁剪为8倍数,返回该影像的DN值与左下角与右上角坐标[x_min,y_min,x_max,y_max]
            [location,landsat_DN_t1] = self.image_cut(landsat_image_t1,8)
            
            #计算影像的反射率
            start_time = time.time()
            landsat_reflectance_t1 = self.cal_reflectance(metadata_t1_filename,landsat_DN_t1,self.band_number,landsat_ID)
            #print('1111',landsat_reflectance_t1[0,0])
            #            if(not landsat_reflectance_t1.any()):
            #                print(111)
            #                return 0  #影像为空，重新选择影像计算。
            end_time = time.time()
            
            #print('抽查两个点的反射率',landsat_reflectance_t1[3000,3000],landsat_reflectance_t1[2000,3000])
            
            print('landsat影像反射率计算完成，所用时间:%0.2fs'%(end_time-start_time))
            
            m,n = landsat_reflectance_t1.shape
            
            #读取分类影像
            class_image = self.read_classimage(classimage_path,m,n)
            
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
            
            #开始计算融合反射率，参数初始化
            #中心像元
            center_pos = int(windows_size/2)
            
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
            
                  
                    #光谱差异性
                    landsat_m = m
                    landsat_n = n
                    sijk = numpy.zeros((landsat_m,landsat_n))
                    #时间差异性
                    tijk = numpy.zeros((landsat_m,landsat_n))
                    #类别平均反射率之差
                    class_mean_difference = numpy.zeros((landsat_m,landsat_n))
                    mean_difference = times_class_mean_reflectance  - time1_class_mean_reflectance
            ##                print(mean_difference)
            
                    print('正在计算权重...')
                    #GPU计算
                    weight_time = time.time()                
                    total_pixel = landsat_m*landsat_n
                    #landsat_r = landsat_reflectance_t1.reshape(1,total_pixel)
                   # class_image_temp = class_image.astype(numpy.int32).reshape(1,total_pixel)
                    mean_difference_temp = numpy.array(mean_difference)
                    #sijk_temp = sijk.reshape(1,total_pixel)
                    #class_mean_difference_temp = class_mean_difference.reshape(1,total_pixel)
                    #tijk_temp = tijk.reshape(1,total_pixel)
                    total = numpy.int32(total_pixel)
                    length = numpy.int32(len(mean_difference))
                   # print(total,length)
            
                    nTheads = 256
                    nBlocks = int((total_pixel+nTheads-1)/nTheads)
                    sijk_time = time.time()
                    sijk = self.cal_sijk(sijk,landsat_reflectance_t1,class_image,time1_class_mean_reflectance,total,length,nTheads,nBlocks,landsat_m,landsat_n)
                    print('计算sijk所用时间：%0.2f'%(time.time()-sijk_time))
                    tijk_time = time.time()
                    class_mean_difference = self.cal_mean(class_mean_difference,class_image,mean_difference_temp,total,length,nTheads,nBlocks,landsat_m,landsat_n)
                    tijk = self.cal_tijk(tijk,class_mean_difference,total,length,nTheads,nBlocks,landsat_m,landsat_n)
                    print('计算tijk所用时间：%0.2f'%(time.time()-tijk_time))
                    print(sijk,tijk)   
                    #sijk = self.cal_sijk(sijk_temp,landsat_r,class_image_temp,time1_class_mean_reflectance,total,length,nTheads,nBlocks,landsat_m,landsat_n)
            
                    #class_mean_difference = self.cal_mean(class_mean_difference_temp,class_image_temp,mean_difference_temp,total,length,nTheads,nBlocks,landsat_m,landsat_n)
                    #tijk = self.cal_tijk(tijk_temp,class_mean_difference_temp,total,length,nTheads,nBlocks,landsat_m,landsat_n)
                    
                    print('计算权重所用时间：%0.2f'%(time.time()-weight_time))
                    
                    weight_st = sijk*tijk
                    
                    #分块处理,重叠三个像素（窗口大小减1）
                    #b_m = int(m/(block_size-int(windows_size/2)))+1
                    #b_n = int(n/(block_size-int(windows_size/2)))+1

                    over_size = windows_size-1 #重叠区域
                    half_size = int(over_size/2)  #去除黑边区域
                    b_m = int((m-half_size)/(block_size+half_size))+1
                    b_n = int((n-half_size)/(block_size+half_size))+1
                    block_landsat = [[0 for i in range(b_n)] for i in range(b_m)] #储存分块后影像
                    position_landsat = [[0 for i in range(b_n)] for i in range(b_m)]
                    drive = landsat_image_t1.GetDriver()
                    projection = landsat_image_t1.GetProjection()
                    fusion_time = time.time()
                    for bi in range(b_m):
                        for bj in range(b_n):
                            if(bi < b_m-1 and bj < b_n-1):
                                time1 = time.time()
                                #block_landsat[bi][bj] = landsat_reflectance_t1[bi*block_size:(bi+1)*block_size+3,bj*block_size:(bj+1)*block_size+3]
                                block_landsat[bi][bj] = self.GPUcal_pixel_ref(landsat_reflectance_t1[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:(bj+1)*block_size+over_size],
                                                                class_image[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:(bj+1)*block_size+over_size],
                                                                class_mean_difference[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:(bj+1)*block_size+over_size],
                                                                weight_st[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:(bj+1)*block_size+over_size],landsat_resolusion)
                                size_x,size_y = block_landsat[bi][bj].shape  
                                block_landsat[bi][bj] = block_landsat[bi][bj][half_size:size_x-half_size,half_size:size_y-half_size]
                                position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)
                                #self.save_image(block_landsat[bi][bj],drive,position_landsat[bi][bj],projection,'%d%d.tif'%(bi,bj),save_path)
                                #print(position_landsat[bi][bj])
                                print(time.time()-time1)
                                #print(1,bi,bj,bi*block_size,(bi+1)*block_size+over_size,bj*block_size,(bj+1)*block_size+over_size,block_landsat[bi][bj].shape)
                            if(bi == b_m-1 and bj < b_n-1):
                                block_landsat[bi][bj] = self.GPUcal_pixel_ref(landsat_reflectance_t1[bi*block_size:,bj*block_size:(bj+1)*block_size+over_size],
                                                                class_image[bi*block_size:,bj*block_size:(bj+1)*block_size+over_size],
                                                                class_mean_difference[bi*block_size:,bj*block_size:(bj+1)*block_size+over_size],
                                                                weight_st[bi*block_size:,bj*block_size:(bj+1)*block_size+over_size],landsat_resolusion)
                                size_x,size_y = block_landsat[bi][bj].shape  
                                block_landsat[bi][bj] = block_landsat[bi][bj][half_size:size_x-half_size,half_size:size_y-half_size]
                                position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)
                                os.chdir(save_path)
                                #self.save_image(block_landsat[bi][bj],drive,position_landsat[bi][bj],projection,'%d%d.tif'%(bi,bj),save_path)
                                #print(position_landsat[bi][bj])
                                #print(2,bi,bj,bi*block_size,landsat_reflectance_t1.shape[0],bj*block_size,(bj+1)*block_size+over_size,block_landsat[bi][bj].shape)
                            if(bi < b_m-1 and bj == b_n-1):
                                block_landsat[bi][bj] = self.GPUcal_pixel_ref(landsat_reflectance_t1[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:],
                                                                class_image[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:],
                                                                class_mean_difference[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:],
                                                                weight_st[bi*block_size:(bi+1)*block_size+over_size,bj*block_size:],landsat_resolusion)
                                size_x,size_y = block_landsat[bi][bj].shape  
                                block_landsat[bi][bj] = block_landsat[bi][bj][half_size:size_x-half_size,half_size:size_y-half_size]
                                position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)
                                os.chdir(save_path)
                                #self.save_image(block_landsat[bi][bj],drive,position_landsat[bi][bj],projection,'%d%d.tif'%(bi,bj),save_path)
                                #print(position_landsat[bi][bj])
                                #print(3,bi,bj,bi*block_size,(bi+1)*block_size+over_size,bj*block_size,landsat_reflectance_t1.shape[1],block_landsat[bi][bj].shape)
                            if(bi == b_m-1 and bj == b_n-1):
                                block_landsat[bi][bj] = self.GPUcal_pixel_ref(landsat_reflectance_t1[bi*block_size:,bj*block_size:],
                                                                class_image[bi*block_size:,bj*block_size:],
                                                                class_mean_difference[bi*block_size:,bj*block_size:],
                                                                weight_st[bi*block_size:,bj*block_size:],landsat_resolusion)
                                size_x,size_y = block_landsat[bi][bj].shape  
                                block_landsat[bi][bj] = block_landsat[bi][bj][half_size:size_x-half_size,half_size:size_y-half_size]
                                position_landsat[bi][bj] = (float(location[0]+(bj*block_size+half_size)*landsat_resolusion), landsat_resolusion, 0.0,float(location[3]-(bi*block_size+half_size)*landsat_resolusion),0.0 , -landsat_resolusion)
                                os.chdir(save_path)
                                #self.save_image(block_landsat[bi][bj],drive,position_landsat[bi][bj],projection,'%d%d.tif'%(bi,bj),save_path)
                                #print(position_landsat[bi][bj])
                                #print(4,bi,bj,bi*block_size,landsat_reflectance_t1.shape[0],bj*block_size,landsat_reflectance_t1.shape[1],block_landsat[bi][bj].shape)        
                     
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
                    self.save_image(block_landsat_new[0],drive,position_landsat[0][0],projection,'%s_b%d.tif'%(modis_time2,modis_band_number),save_path)
                #      block_size = 512
              #      new_land = self.GPUcal_pixel_ref(landsat_reflectance_t1[0:block_size,0:block_size],class_image[0:block_size,0:block_size],class_mean_difference[0:block_size,0:block_size],weight_st[0:block_size,0:block_size])    


                    end = time.time()
                    print('计算总时长：',end-start)
                 
class Kmeansaddadd2d_sanjiao:
    #初始化聚类数，最大迭代数，二维距离
    def __init__(self,class_count,max_iteration,data1,data2):
        self.class_count = class_count #初始化聚类数
        self.max_iteration = max_iteration #最大迭代数
        self.data1 = data1  #数据
        self.data2 = data2  #数据
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
    def cal_center(self,data1,data2,class_count):
        center = []
        
        #height,width = self.data[0].shape
#        data1 = numpy.random.randint(1,255,(7500,7000)).astype('int32')
#        data2 = numpy.random.randint(1,255,(7500,7000)).astype('int32')
        data1 = data1.astype('int32')
        data2 = data2.astype('int32')
        #class_count = 10
        data = [data1,data2]
        height,width = data1.shape
        #distance = numpy.array([0,0,0,0,0,0,0,0,0,0]).astype('float64')
        #data = [data1 for _ in range(3)]
        #初始化第一个种子点
        while True:
            m = random.randint(0,height)
            n = random.randint(0,width)
            state = True
#            for i in range(self.dim):
#                if(not self.data[i][m,n]):
            for i in range(2):
                if(not data[i][m,n]):
                    state = False
                    break
            if(state):
                break
        
        mod_cal_dis = SourceModule(
                '''
                __global__ void cal_distance(int *data1,int *data2,float *dis,double *center1,double *center2,int total,int point){
                    const int count = blockIdx.x*blockDim.x+threadIdx.x;
                    if(count>total){return;}
                    if(data1[count]<0.0001 && data1[count]<0.0001){
                        dis[count] = 0;
                        return;
                    }
                    for(int i = 0;i<point;i++){
                        double temp = (data1[count]-center1[i])*(data1[count]-center1[i])+(data2[count]-center2[i])*(data2[count]-center2[i]);
                        if(temp<dis[count]){dis[count] = temp;}
                    }
                    
                    return;
                    
                }                   
                                   
                ''')
        cal_dis = mod_cal_dis.get_function("cal_distance")
        distance = numpy.ones((height*width)).astype('float64')*sys.maxsize #初始化距离
        total = numpy.int32(height*width)
        center1 = numpy.array([data1[m,n]])
        center2 = numpy.array([data2[m,n]])
        data1 = data1.reshape(1,total)
        data2 = data2.reshape(1,total)
        nTheads = 256
        nBlocks = int((total+nTheads-1)/nTheads)
#        data1 = self.data1.reshape(1,total)
#        data2 = self.data2.reshape(1,total)
        #for i in range(self.class_count):
        for i in range(class_count-1):

            print('正在计算第%d个种子点'%(i+2))
            point = numpy.int32(i+1)
            cal_dis(drv.In(data1),drv.In(data2),drv.InOut(distance),drv.In(center1),drv.In(center2),total,point,
                    block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
            sum_dis = int(sum(distance))

            sum_dis_rand = random.randint(0,sum_dis)
            for j in range(total):
                sum_dis_rand -=distance[j]
                if(sum_dis_rand<0):
                    #j即为新的种子点
                    if(data1[0,j] and data2[0,j]):
                        center1 = center1.tolist()                    
                        center1.append(data1[0,j])
                        center1 = numpy.array(center1)
                        center2 = center2.tolist()
                        center2.append(data2[0,j])
                        center2 = numpy.array(center2)
                   # print(2,center1,center2)
                        break
        print(center1)
        print(center2)
        print('种子点计算完成')
       
        return[center1,center2]
     #计算类间最小距离
    def cal_dis_center(self,center1,center2):
        print('计算类间最小距离')   
        class_count = self.class_count
        #class_count = center1.shape[0]
        distance_center = []
        min_dis = []
        for i in range(class_count):
            min_distance = sys.maxsize
            for j in range(class_count):
                if(i!=j):
                    distance_class = self.cal_distance([center1[i],center2[i]],[center1[j],center2[j]])
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
        data2 = self.data2
        #print(data1.shape)
        class_count = self.class_count
        max_iter = self.max_iteration
        center1,center2 = self.cal_center(data1,data2,class_count)
        min_dis,distance_center = self.cal_dis_center(center1,center2)
        
        center1 = center1.astype('float64')
        center2 = center2.astype('float64')
        
        #min_dis,distance_center = self.cal_dis_center(center1,center2)
        #计算
        mod_cal_dis2 = SourceModule(
            '''
            #define CENTER_POINT %d
            #define MINDIS %d
            __global__ void cal_distance2(int *data1,int *data2,int *class_image,int *new_data1,int *new_data2,double *center1,double *center2,double *min_dis,int *center_dis,int total){
                const int count = blockIdx.x*blockDim.x+threadIdx.x;
                
                if(count>total){
                    return;
                }
                
                if(data1[count]<0.0001 || data2[count]<0.0001){
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
                    double temp = (data1[count]-center1[pre_class])*(data1[count]-center1[pre_class])+(data2[count]-center2[pre_class])*(data2[count]-center2[pre_class]);
                    if(4*temp>min_dis[pre_class]){ 
                        dis1 = MINDIS;
                        for(int i = 0;i<CENTER_POINT;i++){
                            double temp_dis = (data1[count]-center1[i])*(data1[count]-center1[i])+(data2[count]-center2[i])*(data2[count]-center2[i]);
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
                        double temp = (data1[count]-center1[i])*(data1[count]-center1[i])+(data2[count]-center2[i])*(data2[count]-center2[i]);
                        if(temp<dis1){
                            class_image[count] = i;
                            
                            dis1 = temp;
                        }
    
                    }
                }  
                    
                int class_temp = int(class_image[count]);
                new_data1[class_temp*total+count] = data1[count];
                new_data2[class_temp*total+count] = data2[count]; 
                return;
                
            }                   
                               
            '''%(class_count,sys.maxsize))
        #距离计算
        height,width = data1.shape
        total = height*width
        data1 = data1.reshape(1,total)
        data2 = data2.reshape(1,total)
        total = numpy.int32(height*width)
        class_image = (numpy.ones(total)*class_count).astype('int32')
        center1_temp = numpy.zeros(class_count)
        center1_temp = numpy.zeros(class_count)
        block_size = 256 #计算多少行
        b_m = height//block_size
        cal_dis = mod_cal_dis2.get_function('cal_distance2')
        new_data1 = numpy.zeros(total*class_count).astype('int32')
        new_data2 = numpy.zeros(total*class_count).astype('int32')
        nTheads = 256
        nBlocks = int((total+nTheads-1)/nTheads)

        for i in range(max_iter):
            juleitime = time.time()
            print('第%d次聚类'%(i+1))
            print(center1.astype('int32'))
            print(center2.astype('int32'))
            #print(class_image[:10])
            class_image_new = numpy.zeros(0)
            #cal_dis = mod_cal_dis2.get_function('cal_distance')
            #分块处理
            total = block_size*width
            center1_temp = numpy.zeros(class_count)
            center2_temp = numpy.zeros(class_count)
            for m in range(b_m):
                if(m<b_m-1):                    
                    new_data1 = numpy.zeros(total*class_count).astype('int32')
                    new_data2 = numpy.zeros(total*class_count).astype('int32')
                    data1_temp = data1[0][m*total:(m+1)*total].astype('int32')
                    data2_temp = data2[0][m*total:(m+1)*total].astype('int32')
                    class_image_temp = class_image[m*total:(m+1)*total].astype('int32')
                    nTheads = 256
                    nBlocks = int((total+nTheads-1)/nTheads)
                    cal_dis(drv.In(data1_temp),drv.In(data2_temp ),drv.InOut(class_image_temp),
                        drv.InOut(new_data1),drv.InOut(new_data2),drv.In(center1.astype('float64')),drv.In(center2.astype('float64')),
                        drv.In(min_dis),drv.In(distance_center),numpy.int32(total), 
                        block = (nTheads,1,1),grid = (nBlocks,1))
             #调整聚类中心
                    for j in range(class_count):
                        temp_data1 = new_data1[j*total:(j+1)*total].astype('float64')
                        temp_data2 = new_data2[j*total:(j+1)*total].astype('float64')
                        #print(sum(new_data1),sum(new_data2),sum(data1_temp),sum(data2_temp),center1_temp[j],sum(temp_data1),numpy.sum(temp_data1>0),sum(temp_data1)/numpy.sum(temp_data1>0))  
                        if(center1_temp[j] and center2_temp[j]):
                            if(numpy.sum(temp_data1>0) and numpy.sum(temp_data2>0)):
                                center1_temp[j] = (center1_temp[j]+sum(temp_data1)/numpy.sum(temp_data1>0))/2
                                center2_temp[j] = (center2_temp[j]+sum(temp_data2)/numpy.sum(temp_data2>0))/2
                                #print(1,center1_temp[j],center2_temp[j])
                        else:
                            if(numpy.sum(temp_data1>0) and numpy.sum(temp_data2>0)):
                                center1_temp[j] = sum(temp_data1)/numpy.sum(temp_data1>0)
                                center2_temp[j] = sum(temp_data2)/numpy.sum(temp_data2>0)
                                #print(2,center1_temp[j],center2_temp[j])
                            else:
                                center1_temp[j] = 0
                                center2_temp[j] = 0
                                #print(3,center1_temp[j],center2_temp[j])
                        #print(center1_temp[j],center2_temp[j])
                    #print(class_image_temp[:10])
                    class_image_new = numpy.hstack((class_image_new,class_image_temp)).astype('int32')
                else:
                    
                    data1_temp = data1[0][m*total:].astype('int32')
                    data2_temp = data2[0][m*total:].astype('int32')
                    class_image_temp = class_image[m*total:].astype('int32')
                    total = data1_temp.shape[0]
                    new_data1 = numpy.zeros(total*class_count).astype('int32')
                    new_data2 = numpy.zeros(total*class_count).astype('int32')
                    nTheads = 256
                    nBlocks = int((total+nTheads-1)/nTheads)
                    cal_dis(drv.In(data1_temp),drv.In(data2_temp ),drv.InOut(class_image_temp),
                        drv.InOut(new_data1),drv.InOut(new_data2),drv.In(center1.astype('float64')),drv.In(center2.astype('float64')),
                        drv.In(min_dis),drv.In(distance_center),numpy.int32(total), 
                        block = (nTheads,1,1),grid = (nBlocks,1))
                    for j in range(class_count):
                        temp_data1 = new_data1[j*total:(j+1)*total].astype('float64')
                        temp_data2 = new_data2[j*total:(j+1)*total].astype('float64')
                        #print(center1_temp[j],sum(temp_data1),numpy.sum(temp_data1>0),sum(temp_data1)/numpy.sum(temp_data1>0)) 
                        if(numpy.sum(temp_data1>0) and numpy.sum(temp_data2>0)):
                            center1_temp[j] = (center1_temp[j]+sum(temp_data1)/numpy.sum(temp_data1>0))/2
                            center2_temp[j] = (center2_temp[j]+sum(temp_data2)/numpy.sum(temp_data2>0))/2
                    class_image_new = numpy.hstack((class_image_new,class_image_temp)).astype('int32')
            #print(center1_temp)
            #print(center2_temp)
            class_image = copy.deepcopy(class_image_new).astype('int32')
            if(numpy.sum(abs(center1-center1_temp)<1)>=class_count and numpy.sum(abs(center2-center2_temp)<1)>=class_count):
                break
            center1 = copy.deepcopy(center1_temp)
            center2 = copy.deepcopy(center2_temp)
            #重新计算类间距离
            min_dis,distance_center = self.cal_dis_center(center1,center2)
            print('第%d次聚类'%(i+1),'所用时间:%0.2f'%(time.time()-juleitime))
        print('聚类完成！')   
        return class_image.reshape((height,width))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root',type=str,help = '根路径，必选')
parser.add_argument('--landsat_t1_path', type=str,help = 'landsat初期影像解压缩后文件夹路径，必选')
parser.add_argument('--modis_path', type=str,help = 'MODIS影像文件夹路径,必选')
parser.add_argument('--MRT_save_path', type=str,help = 'MODIS转投影、裁剪与重采样后保存路径，必选')
parser.add_argument('--save_path', type=str,help = '合成影像保存路径，必选')
parser.add_argument('--java_path', type=str,help = 'Java安装路径，例如C:\Program Files\Java\jre1.8.0_144，必选')
parser.add_argument('--MRT_path', type=str,help = 'MRT安装路径，例如D:\Program Files\MRT，必选')
parser.add_argument('--classimage_path', type=str,help = '分类影像保存路径，例如D:\class_ST.tif，必选')
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
fusion = Fusion(args.root,class_count = args.class_count,band_number = args.band_number,window_size = args.windows_size)
#聚类 
time1 = time.time()
if(args.landsat_ID == 8):
    data_r,temp = fusion.read_landsat(args.landsat_t1_path,4)
    data_n,temp = fusion.read_landsat(args.landsat_t1_path,5)
else:
    data_r,temp = fusion.read_landsat(args.landsat_t1_path,3)
    data_n,temp = fusion.read_landsat(args.landsat_t1_path,4)

a = Kmeansaddadd2d_sanjiao(args.class_count,args.max_iter,data_r,data_n)
print('聚类所用时间：%0.2f'%(time.time()-time1))
time2 = time.time()
batch = False
fusion.fusion(args.landsat_t1_path,args.classimage_path,args.MRT_save_path,modis_path_list,args.save_path,args.java_path,args.MRT_path,args.zhouqi,args.landsat_resolusion,args.modis_resolusion,args.landsat_ID,args.block_size,batch)
print('融合所用时间：%0.2f'%(time.time()-time2))









    
