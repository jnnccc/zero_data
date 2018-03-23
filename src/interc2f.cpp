#define _FILE_OFFSET_BITS 64
#include<iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "CRSRData.h"
using namespace std;
extern "C"
       { //2018-01-23T22:22:22.000
char 	module10_mp_utcsta_[23],module10_mp_utcsto_[23],module10_mp_utc1_[23],module10_mp_utc2_[23];
int		module10_mp_sampling_;
//说明：
//时间格式：2018-01-23T22:22:22.000
//module10_mp_utcsta_[23]：数据文件起始时刻
//module10_mp_utcsta_[23]：数据文件结束时刻
//module10_mp_utc1_[23]：读取数据开始时刻
//module10_mp_utc2_[23]：读取数据结束时刻


void  	rsrinfo_(char *datafile);//获取数据文件基本信息 
void 	getrsrdata_(int datai[],int dataq[],char *datafile,int *chanel); //读取数据
}

void rsrinfo_(char *datafile)
{	
//输入：datafile
//输出：采样率，量化率，通道本振设置，数据文件开始时刻和结束时刻
	CRSRDATA RsrData(datafile);
	module10_mp_sampling_=RsrData.Data_info.Cdata_Rate;
    cout<<"Chanel 1 NCO setting:"<<RsrData.Data_info.Cdata_NCO[0]<<endl;
    cout<<"Chanel 2 NCO setting:"<<RsrData.Data_info.Cdata_NCO[1]<<endl;
    cout<<"Chanel 3 NCO setting:"<<RsrData.Data_info.Cdata_NCO[2]<<endl;
    cout<<"Chanel 4 NCO setting:"<<RsrData.Data_info.Cdata_NCO[3]<<endl;
    cout<<"Bit number:"<<RsrData.Data_info.Cdata_Bit<<endl;
    cout<<"Sampling frequency:"<<RsrData.Data_info.Cdata_Rate<<endl;	
    cout<<"Data start time:"<<RsrData.Data_info.StartUTC<<endl;	
    cout<<"Data  stop time:"<<RsrData.Data_info.StopUTC<<endl;	
strncpy(module10_mp_utcsta_,RsrData.Data_info.StartUTC,23);
strncpy(module10_mp_utcsto_,RsrData.Data_info.StopUTC,23);

}

//读取数据，返回从module10_mp_utc1_到module10_mp_utc2_的数据，数据存储在datai,dataq,中
void getrsrdata_(int datai[],int dataq[],char *datafile,int *chanel)
{
//输入：module10_mp_utc1_，module10_mp_utc2_，*chanel	
//输出：datai,dataq
CRSRDATA RsrData(datafile);	
    RsrData.Cdata_read(module10_mp_utc1_,module10_mp_utc2_,*chanel,datai,dataq);
}
