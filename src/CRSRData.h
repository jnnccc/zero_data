/***********************************************************************************************
数据处理过程中需要使用原始数据，原始数据打包封装成帧后存储到文件中。数据帧中即有采样数据也有采样数据相关的辅助信息。
此代码的主要功能是实现不同方式的数据文件读取，从中获得采样数据及辅助信息。
1)打开数据文件，返回采样率，字长，数字本振，开始时间，截止时间。还同时得到一些类中的私有变量：Cdata_fp，Data_start，PPS_time
  实际上，在新建一个类对象时，会调用此函数，得到所有必要信息存放在结构体Data_info中。
2）关闭数据文件
3)三种数据读取方式：全部返回两路数据（I路和Q路），正常返回实际读出的数据点数，异常时返回－1及相应的错误码。
		(1）指定起始、截止时间和通道号，返回该通道在指定时间内的两路数据
		(2）指定起始时间和通道号，返回该通道由起始时间开始指定数据点数的两路数据
		(3）指定数据点数和通道号，返回该通道由当前位置开始指定长度的两路数据
4）获取毫秒字
5）获取采样率
6）获取字长
7）获取数字本振
8）获取起始UTC时间
9）获取截止UTC时间
10）获取当前数据帧的UTC时间

所有函数返回1代表成功执行，返回0代表操作失败。exit(1)失败exit(0)成功

会对文件指针进行移动的函数有：Cdata_open,Find_PPS,Cdata_read。

UTC时间格式为ISOC格式：  '1987-04-12T16:31:12.000'  字符串
通道号为1 2 3 4 之一
Cdata_erro含义：1－
Author：lm
Date:2012-07-27

************************************************************************************************/
#ifndef CRSRDATA_H

#define CRSRDATA_H

#define _USE_LARGEFILE 64
#define _FILE_OFFSET_BITS 64

#ifndef  WIN32
#include <unistd.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


class CRSRDATA

{

public:
	 CRSRDATA(char *Cdata_filename);
	 ~CRSRDATA();

public:

	struct DATA_INFO
	{
		int Cdata_Rate;
		int Cdata_Bit;
		int Cdata_NCO[4];
		int Interval;  //每包所占时间长度
		float blkpt;     //每秒包数

		char StartUTC[24];
		char StopUTC[24];
		char CurrentUTC[24];
	};


	int Cdata_open(char *Cdata_filename);
	int Cdata_close();

	int get_RateBit();
//	int get_Bit();
	void get_NCO(int Ch_No);

	void getPPSTime();
	void get_mstime();
	int getStartUTC();
	int getStopUTC();
	int get_UTC();

	void Cdata_perro();

	int Cdata_read(char *Start_UTC,char *Stop_UTC,int Ch_No,int *pIData,int *pQData);
	int Cdata_read(char *Start_UTC,int Length,int Ch_No,int *pIData,int *pQData);
	int Cdata_read(int Length,int Ch_No,int *pIData,int *pQData);

	DATA_INFO Data_info;



private:
	struct RESIDUE  //用于记录未读取整包数据的情况
	{
		int Last_ReadNum;  //上次最后一包数据中读出的采样点数
		int Last_Remaind;  //上次最后一包数据中剩余的采样点数
	};
	struct TIME //数据中记录的复位时间
	{
		int year;
		int month;
		int day;
		int hour;
		int minute;
		int second;
		int msecond;
	};

	int Find_PPS(FILE *f_fp);
	int Read_dat(int tm_skip,int BlkNum,int Ch_No,int *pIData,int *pQData,int Length,int RemNum);

	int Cdata_error;
	FILE *Cdata_fp;
	int Data_start;  //用于记录数据开始的位置（相对文件头部偏移的帧数）//此处应该不是偏移的数据帧而是偏移的字节数
	bool Chg_file;   //用于标识文件是否已经读完（0-未读完，1-读完）

	int Cdatafp_Position;  //上次读取完数据后文件指针的位置（相对文件头部偏移的帧数）  //为了适应大文件读取此变量应该以帧为单位

	char Head_words[16];
	unsigned int Tail_words[188];
	RESIDUE This_residue;
	TIME PPS_time;  //保存复位的时间
	TIME Start_time,Stop_time; //保存第一个整毫秒的时间和最后一包数据的时间
	int ms_time;


};

#endif
