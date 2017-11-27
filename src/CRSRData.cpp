/***********************************************************************************************
数据处理过程中需要使用原始数据，原始数据打包封装成帧后存储到文件中。数据帧中即有采样数据也有采样数据相关的辅助信息。
此代码的主要功能是实现不同方式的数据文件读取，从中获得采样数据及辅助信息。
1)打开数据文件，返回采样率，字长，数字本振，开始时间，截止时间。还同时得到一些类中的私有变量：Cdata_fp，Data_start，PPS_time
2）关闭数据文件
3)三种数据读取方式：全部返回两路数据（I路和Q路），正常返回实际读出的数据点数，异常时返回－1及相应的错误码。
		(1）指定起始、截止时间和通道号，返回该通道在指定时间内的两路数据
		(2）指定起始时间和通道号，返回该通道由起始时间开始指定长度的两路数据
		(3）指定数据长度和通道号，返回该通道由当前位置开始指定长度的两路数据
4）获取毫秒字
5）获取采样率
6）获取字长
7）获取数字本振
8）获取起始UTC时间
9）获取截止UTC时间
10）获取当前数据帧的UTC时间

所有函数返回1代表成功执行，返回0代表操作失败。exit(1)失败exit(0)成功

会对文件指针进行移动的函数有：Cdata_open,Find_PPS,Cdata_read。

UTC时间格式为ISOC格式：  '1987-04-12T16:31:12.814'  字符串
Cdata_erro含义：1－
Author：lm
Date:2012-07-27

************************************************************************************************/
//#define _LARGEFILE_SOURCE
//#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#include<iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "CRSRData.h"
using namespace std;
CRSRDATA:: CRSRDATA(char *Cdata_filename)
{
	Data_info.Cdata_Rate = 0;
	Data_info.Cdata_Bit = 0;
//	memset(Data_info.Cdata_NCO,0,sizeof((int)*4));
	Data_info.Interval = 0;
	Data_info.blkpt = 0.000;
//	memset(Data_info.StartUTC,0,sizeof((char)*24));
//	memset(Data_info.StopUTC,0,sizeof((char)*24));
//	memset(Data_info.CurrentUTC,0,sizeof((char)*24));
	Cdata_error = 0;
	Data_start = 0;
	Cdatafp_Position = 0;
	Chg_file = 0;
//	memset(Head_words,0,sizeof((char)*16));
//	memset(Tail_words,0,sizeof((unsigned long int)*24));
	This_residue.Last_ReadNum = 0;
	This_residue.Last_Remaind = 0;
	ms_time = 0;
	Cdata_open(Cdata_filename);

//	printf("Start:%d,Current:%d\n",Data_start,Cdatafp_Position);

}
CRSRDATA:: ~CRSRDATA()
{
	Cdata_close();
}
int CRSRDATA::Cdata_open(char *Cdata_filename)
{
	char f_file[100];
	int Count = 0;
//	char H_buffer[16];
//	char D_buffer[32000];
//	U32 T_buffer[188];
	FILE *f_fp;

	strcpy(f_file,Cdata_filename);
//	printf("test\n");
	if(NULL==(f_fp = fopen(f_file,"rb")))
//	if(!(f_fp =fopen(f_file,"rb")))
	{
		printf("cannot open file:%s\n",f_file);
		return(0);
		exit(1);
	}
//	printf("test\n");

	while(Count<6)
	{
		if(16==(fread(Head_words,1,16,f_fp)))
			break;
		sleep(5000);
		Count++;
		if(5==Count) printf("Empty file!\n");
	}
	fseeko(f_fp,32000,1);
	fread(Tail_words,4,188,f_fp);
	get_RateBit();
//	get_Rate();
//	get_Bit();
	get_NCO(1);
	get_NCO(2);
	get_NCO(3);
	get_NCO(4);
	getPPSTime();
	if(0==Find_PPS(f_fp))  exit(1);
	Cdata_fp = f_fp;
	if(Data_start!=ftell(Cdata_fp)) printf("Cdata_fp start position offset!\n");
	Data_start = ftell(Cdata_fp);

	fseeko(f_fp,32016,1);
	fread(Tail_words,4,188,f_fp);
	getStartUTC();
	fseeko(f_fp,-752,2);
	fread(Tail_words,4,188,f_fp);
	getStopUTC();
//	fclose(f_fp);
	return(1);
}

int CRSRDATA::Cdata_close()
{
	if(-1==fclose(Cdata_fp))
	{
		printf("cannot close file!\n");
		return(0);
		exit(1);
	}
	else 
	{
//		printf("File closed~\n");
		return(1);
	}
}


int CRSRDATA::get_RateBit()
{
	unsigned int rate_bit;
	int rate_ind,bit_ind;
//	unsigned int rate_array[10] = {160000,16000,50000,100000,200000,2000000,4000000,8000000,16000000,32000000};
	unsigned int rate_array[10] = {1000000,10000,50000,100000,200000,2000000,4000000,8000000,16000000,32000};
	int bit_array[5] = {16,8,4,2,1};
	rate_bit = Tail_words[3];

	rate_ind = rate_bit/256;
	bit_ind = rate_bit%256;
	if(-1<=rate_ind&&rate_ind<10&&-1<bit_ind&&bit_ind<5)
	{
		Data_info.Cdata_Rate = rate_array[rate_ind];
		Data_info.Cdata_Bit = bit_array[bit_ind];
		Data_info.Interval = 32e6/(Data_info.Cdata_Rate*Data_info.Cdata_Bit);
		Data_info.blkpt = 1000.000/Data_info.Interval;
		return(1);
	}
	else
	{
		printf("Illegal Rate_Bit word!\n");
		return(0);
	}
}

/*int CRSRDATA::get_Rate();
{
	int Rate_ind;
	int rate_array[10] = {16e4,16e3,50e3,1e5,2e5,2e6,4e6,8e6,16e6,32e3};
	rate_bit = Tail_words[4];

	rate_ind = floor(rate_bit/256);
	Cdata_Rate = rate_array[rate_ind];
}

int CRSRDATA:get_Bit();
{
	int rate_bit;
	int bit_array[5] = {16,8,4,2,1};
	rate_bit = Tail_words[4];

	bit_ind = rate_bit%256;
	Cdata_Bit = bit_array[bit_ind];
}*/

void CRSRDATA::get_NCO(int Ch_No)
{
	int nco_hs,nco_ls;

	nco_hs = Tail_words[Ch_No+6]*1e6;
	nco_ls = Tail_words[Ch_No+14];
	Data_info.Cdata_NCO[Ch_No-1] = nco_hs+nco_ls; //search for 1000ms
}


void CRSRDATA::getPPSTime()
{
	unsigned int date1,date2;
	date1 = Tail_words[1];
	date2 = Tail_words[2];
	PPS_time.msecond = 0;
	PPS_time.second = date1/16777216;
	PPS_time.minute = (date1%16777216)/65536;
	PPS_time.hour = (date1%65536)/256;
	PPS_time.day = date1%256;
	PPS_time.month = date2/16777216+1;
	PPS_time.year = (date2%16777216)/65536*100+(date2%65536)/256;
//     PPS_time.day = date1/16777216;
//     PPS_time.hour = (date1%16777216)/65536;
//     PPS_time.minute = (date1%65536)/256;
//     PPS_time.second = date1%256;
//     PPS_time.month = date2%256+1;
//     PPS_time.year = (date2%16777216)/65536*100+(date2%65536)/256;
//     cout<<"year"<<PPS_time.year<<endl;
//     cout<<"mon"<<PPS_time.month<<endl;
//     cout<<"day"<<PPS_time.day<<endl;
//     cout<<"hour"<<PPS_time.hour<<endl;
//     cout<<"minute"<<PPS_time.minute<<endl;
//     cout<<"second"<<PPS_time.second<<endl;

}


int CRSRDATA::getStartUTC()
{
	get_mstime();
	unsigned int tot_s;
	int tot_s_cnt,sec_cnt,min_cnt,hr_cnt,day_cnt,mon_cnt,year_cnt;
	int mon_all[] = {31,28,31,30,31,30,31,31,30,31,30,31};

	ms_time -= Data_info.Interval;	//因为每包数据的毫秒字记录的是包结束的时刻，所以这里有个减包时长的动作
	tot_s_cnt = ms_time/1000;
	tot_s = tot_s_cnt+PPS_time.second+PPS_time.minute*60+PPS_time.hour*3600;
	sec_cnt = tot_s%60;
	min_cnt = (tot_s%3600)/60;
	hr_cnt = (tot_s%86400)/3600;
	day_cnt = tot_s/(86400);
	day_cnt = day_cnt+PPS_time.day;
	mon_cnt = PPS_time.month;
	year_cnt = PPS_time.year;
	tot_s_cnt = ms_time%1000;

	while (day_cnt >= mon_all[mon_cnt-1]+1)
	{
    		if (0==(mon_cnt == 2 && (year_cnt%4) == 0 && (year_cnt%100) !=0))
		{
			day_cnt = day_cnt-mon_all[mon_cnt];
			mon_cnt = mon_cnt+1;
		}
		else
		{ 
			if(day_cnt >= 30)
			{
				day_cnt = day_cnt-29;
				mon_cnt = mon_cnt+1;
			}
			else
				break;
		}
		if(mon_cnt == 13)
		{
			year_cnt = year_cnt+1;
			mon_cnt = 1;
		}
	}
	Start_time.year = year_cnt;
	Start_time.month = mon_cnt;
	Start_time.day = day_cnt;
	Start_time.hour = hr_cnt;
	Start_time.minute = min_cnt;
	Start_time.second = sec_cnt;
	Start_time.msecond = tot_s_cnt;
//	printf("Data start time:(UTC)%04d-%02d-%02dT%02d:%02d:%02d.%03d\n",year_cnt,mon_cnt,day_cnt,hr_cnt,min_cnt,sec_cnt,tot_s_cnt);
	sprintf(Data_info.StartUTC,"%04d-%02d-%02dT%02d:%02d:%02d.%03d",year_cnt,mon_cnt,day_cnt,hr_cnt,min_cnt,sec_cnt,tot_s_cnt);
	Data_info.StartUTC[23] = '\0';
	return(1);
}

int CRSRDATA::getStopUTC()
{
	get_mstime();
	unsigned int tot_s;
	int tot_s_cnt,sec_cnt,min_cnt,hr_cnt,day_cnt,mon_cnt,year_cnt;
	int mon_all[] = {31,28,31,30,31,30,31,31,30,31,30,31};

//	ms_time -= Data_info.Interval;	//因为求的是数据结束的时刻，所以此处不应该再有减包时长的动作
	tot_s_cnt = ms_time/1000;
	tot_s = tot_s_cnt+PPS_time.second+PPS_time.minute*60+PPS_time.hour*3600;
	sec_cnt = tot_s%60;
	min_cnt = (tot_s%3600)/60;
	hr_cnt = (tot_s%86400)/3600;
	day_cnt = tot_s/(86400);
	day_cnt = day_cnt+PPS_time.day;
	mon_cnt = PPS_time.month;
	year_cnt = PPS_time.year;
	tot_s_cnt = ms_time%1000;

	while (day_cnt >= mon_all[mon_cnt-1]+1)
	{
    		if (0==(mon_cnt == 2 && (year_cnt%4) == 0 && (year_cnt%100) !=0))
		{
			day_cnt = day_cnt-mon_all[mon_cnt];
			mon_cnt = mon_cnt+1;
		}
		else
		{ 
			if(day_cnt >= 30)
			{
				day_cnt = day_cnt-29;
				mon_cnt = mon_cnt+1;
			}
			else
				break;
		}
		if(mon_cnt == 13)
		{
			year_cnt = year_cnt+1;
			mon_cnt = 1;
		}
	}
	Stop_time.year = year_cnt;
	Stop_time.month = mon_cnt;
	Stop_time.day = day_cnt;
	Stop_time.hour = hr_cnt;
	Stop_time.minute = min_cnt;
	Stop_time.second = sec_cnt;
	Stop_time.msecond = tot_s_cnt;
//	printf("Data stop time:(UTC)%04d-%02d-%02dT%02d:%02d:%02d.%03d\n",year_cnt,mon_cnt,day_cnt,hr_cnt,min_cnt,sec_cnt,tot_s_cnt);
	sprintf(Data_info.StopUTC,"%04d-%02d-%02dT%02d:%02d:%02d.%03d",year_cnt,mon_cnt,day_cnt,hr_cnt,min_cnt,sec_cnt,tot_s_cnt);
	Data_info.StopUTC[23] = '\0';
	return(1);
}

int CRSRDATA::get_UTC()
{
	get_mstime();
	unsigned int tot_s;
	int tot_s_cnt,sec_cnt,min_cnt,hr_cnt,day_cnt,mon_cnt,year_cnt;
	int mon_all[] = {31,28,31,30,31,30,31,31,30,31,30,31};

	//计算当前时间，毫秒为单位，如果指针刚好位于一包中间，则计算的时间截止到已读取的位置
	ms_time = ms_time+This_residue.Last_ReadNum/Data_info.Cdata_Rate;	//求当前时刻，应该指已读取数据的结束时刻或接下来数据的起始时刻，故无需减包时长动作
	tot_s_cnt = ms_time/1000;
	tot_s = tot_s_cnt+PPS_time.second+PPS_time.minute*60+PPS_time.hour*3600;
	sec_cnt = tot_s%60;
	min_cnt = (tot_s%3600)/60;
	hr_cnt = (tot_s%86400)/3600;
	day_cnt = tot_s/(86400);
	day_cnt = day_cnt+PPS_time.day;
	mon_cnt = PPS_time.month;
	year_cnt = PPS_time.year;
	tot_s_cnt = ms_time%1000;

	while (day_cnt >= mon_all[mon_cnt-1]+1)
	{
    		if (0==(mon_cnt == 2 && (year_cnt%4) == 0 && (year_cnt%100) !=0))
		{
			day_cnt = day_cnt-mon_all[mon_cnt];
			mon_cnt = mon_cnt+1;
		}
		else
		{ 
			if(day_cnt >= 30)
			{
				day_cnt = day_cnt-29;
				mon_cnt = mon_cnt+1;
			}
			else
				break;
		}
		if(mon_cnt == 13)
		{
			year_cnt = year_cnt+1;
			mon_cnt = 1;
		}
	}
//	printf("Data current time:(UTC)%04d-%02d-%02dT%02d:%02d:%02d.%03d\n",year_cnt,mon_cnt,day_cnt,hr_cnt,min_cnt,sec_cnt,tot_s_cnt);
	sprintf(Data_info.CurrentUTC,"%04d-%02d-%02dT%02d:%02d:%02d.%03d",year_cnt,mon_cnt,day_cnt,hr_cnt,min_cnt,sec_cnt,tot_s_cnt);
	Data_info.CurrentUTC[23] = '\0';
//	printf("Current position:%ld\n",Cdatafp_Position);
	return(1);
}


void CRSRDATA::get_mstime()
{
	ms_time = Tail_words[0];
}

int CRSRDATA::Find_PPS(FILE *f_fp)  //search for 1000ms
{
	int Second_flag = 1;
	int Count = 0;
	while(Second_flag)
	{
		fseeko(f_fp,32016,1);
		if(188==fread(Tail_words,4,188,f_fp))
		{
			get_mstime();
			Second_flag = (ms_time-Data_info.Interval)%1000;
			Count = 0;
		}
		else
		{
			fseeko(f_fp,-32768,1);  //此处如此处理可能会有隐患
			printf("Read is faster than write!\n");
			sleep(5);
			Count++;
			if(Count==5)
			{
				printf("Cannot find PPS position!\n");
				return(0);
			}
		}
	}
	fseeko(f_fp,-32768,1);
	Data_start = ftell(f_fp);
	return(1);
}

void CRSRDATA::Cdata_perro()
{
	
}


int CRSRDATA::Cdata_read(char *Start_UTC,char *Stop_UTC,int Ch_No,int *pIData,int *pQData)
{
	if(Ch_No<1||Ch_No>4)
	{
		printf("Illegal channel number\n");
		exit(1);
	}
	if((strcmp(Start_UTC,Data_info.StartUTC)<0)||(strcmp(Start_UTC,Data_info.StopUTC)>0))
	{
		printf("Start UTC beyond the range:%s->%s\n",Data_info.StartUTC,Start_UTC);
		exit(1);
	}
	if((strcmp(Stop_UTC,Data_info.StopUTC)>0)||(strcmp(Stop_UTC,Data_info.StartUTC)<0))
	{
		printf("Stop UTC beyond the range\n");
		exit(1);
	}

	int tm_year[2],tm_mon[2],tm_day[2],tm_hr[2],tm_min[2],tm_s[2],tm_ms[2],tm_skip,BlkNum;
	float blk_skip,RemNum;
	sscanf(Start_UTC,"%4d-%2d-%2dT%2d:%2d:%2d.%3d",&tm_year[0],&tm_mon[0],&tm_day[0],&tm_hr[0],&tm_min[0],&tm_s[0],&tm_ms[0]);
	if(tm_mon[0]-Start_time.month==1) tm_day[0] = Start_time.day+1;
	if(tm_day[0]-Start_time.day==1)  tm_hr[0] +=24; 
	tm_skip = (((tm_hr[0]-Start_time.hour)*60+(tm_min[0]-Start_time.minute))*60+(tm_s[0]-Start_time.second))*1000+(tm_ms[0]-Start_time.msecond);
	blk_skip = tm_skip*Data_info.blkpt/1000;
	BlkNum = int(blk_skip);
	RemNum = (blk_skip-BlkNum)*32000/Data_info.Cdata_Bit;

//	fseeko(Cdata_fp,(Data_start+32768*BlkNum),0);  //跳过整数包
	double st= Data_start+32768.00*BlkNum;
//	printf("sizeof(off_t):%d,st:%f\n",sizeof(off_t),st/32768.00);
	fseeko(Cdata_fp,st,0);  //跳过整数包
	Cdatafp_Position = ftello(Cdata_fp)/32768;
//	printf( "Skip %ld(%ld) packages ,start from %ld\n",Cdatafp_Position,BlkNum,Data_start/32768);
	
	sscanf(Stop_UTC,"%4d-%2d-%2dT%2d:%2d:%2d.%3d",&tm_year[1],&tm_mon[1],&tm_day[1],&tm_hr[1],&tm_min[1],&tm_s[1],&tm_ms[1]);
	if(tm_mon[1]-tm_mon[0]==1) tm_day[1] = tm_day[0]+1;
	if(tm_day[1]-tm_day[0]==1)  tm_hr[1] +=24; 
	tm_skip = (((tm_hr[1]-tm_hr[0])*60+(tm_min[1]-tm_min[0]))*60+(tm_s[1]-tm_s[0]))*1000+(tm_ms[1]-tm_ms[0]);
	blk_skip = tm_skip*Data_info.blkpt/1000+RemNum*Data_info.Cdata_Bit/32000;
	BlkNum = int(blk_skip);
	This_residue.Last_ReadNum = (blk_skip-BlkNum)*32000/Data_info.Cdata_Bit;
	

	tm_skip = RemNum; //再次利用tm_skip，以跳过RemNum
	int Length = BlkNum*32000/Data_info.Cdata_Bit-RemNum+This_residue.Last_ReadNum;
	int *LpIData = pIData;
	int *LpQData = pQData;
	return(Read_dat(tm_skip,BlkNum,Ch_No,pIData,pQData,Length,RemNum));
}


int CRSRDATA::Cdata_read(char *Start_UTC,int Length,int Ch_No,int *pIData,int *pQData)
{
	if(Ch_No<1||Ch_No>4)
	{
		printf("Illegal channel number\n");
		exit(1);
	}
	if((strcmp(Start_UTC,Data_info.StartUTC)<0)||(strcmp(Start_UTC,Data_info.StopUTC)>0))
	{
		printf("Start UTC beyond the range\n");
		exit(1);
	}


	int tm_year,tm_mon,tm_day,tm_hr,tm_min,tm_s,tm_ms,tm_skip,BlkNum;
	float blk_skip,RemNum;
	sscanf(Start_UTC,"%4d-%2d-%2dT%2d:%2d:%2d.%3d",&tm_year,&tm_mon,&tm_day,&tm_hr,&tm_min,&tm_s,&tm_ms);
	if(tm_mon-Start_time.month==1) tm_day = Start_time.day+1;
	if(tm_day-Start_time.day==1)  tm_hr +=24; 
	tm_skip = (((tm_hr-Start_time.hour)*60+(tm_min-Start_time.minute))*60+(tm_s-Start_time.second))*1000+(tm_ms-Start_time.msecond);
	blk_skip = tm_skip*Data_info.blkpt/1000;
	BlkNum = int(blk_skip);
	RemNum = (blk_skip-BlkNum)*32000/Data_info.Cdata_Bit;

//	fseeko(Cdata_fp,(Data_start+32768L*(long)BlkNum),0);  //跳过整数包
	double st=Data_start+32768.00*BlkNum;
	fseeko(Cdata_fp,st,0);  //跳过整数包
	Cdatafp_Position = ftello(Cdata_fp)/32768;
	
	BlkNum = (Length+RemNum)/(32000/Data_info.Cdata_Bit);
	This_residue.Last_ReadNum = (Length+RemNum)-BlkNum*32000/Data_info.Cdata_Bit;

	tm_skip = RemNum; //再次利用tm_skip，以跳过RemNum

	int *LpIData = pIData;
	int *LpQData = pQData;
	return(Read_dat(tm_skip,BlkNum,Ch_No,pIData,pQData,Length,RemNum));
}


int CRSRDATA::Cdata_read(int Length,int Ch_No,int *pIData,int *pQData)
{
	if(Chg_file)
	{
		printf("Here is the end of file\n");
		exit(1);
	}
	if(Ch_No<1||Ch_No>4)
	{
		printf("Illegal channel number\n");
		exit(1);
	}
	
	int tm_skip,BlkNum;
	Length -= This_residue.Last_Remaind;
	BlkNum = Length/(32000/Data_info.Cdata_Bit);
	This_residue.Last_ReadNum = Length-BlkNum*32000/Data_info.Cdata_Bit;

	int RemNum = -1*This_residue.Last_Remaind;

	fseeko(Cdata_fp,Cdatafp_Position*32768.00,0);  //将指针移回到上次读取结束的位置
	if(This_residue.Last_Remaind !=0&&!Chg_file)
	{
		int i=0;
		int F_end = 0;
		int read_error = 0;
		int chnum = Ch_No-1;
		tm_skip = This_residue.Last_Remaind; 
		switch(Data_info.Cdata_Bit) //4 2 1 三种比特率处理，将8通道合并读出.比特率8 16，分开读出
		{
				case 1:  
				{
					char dat_tmp[32000];
					memset(dat_tmp,0,32000*sizeof(char));
					read_error = fread(dat_tmp,1,tm_skip,Cdata_fp);
					while(read_error!=tm_skip)
					{
						printf("Read is faster than write,pause(18),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,1,tm_skip,Cdata_fp);
					}
					chnum *= 2;
					for(i=0;i<tm_skip;i++)
					{
						dat_tmp[i] >>= chnum;
						*(pIData+i) = dat_tmp[i]&1;
						*(pQData+i) = (dat_tmp[i]&2)/2;
					}
					break;
				}
				case 2:
				{
					short int dat_tmp[16000];
					memset(dat_tmp,0,16000*sizeof(short int));
					read_error = fread(dat_tmp,2,tm_skip,Cdata_fp);
					while(read_error!=tm_skip)
					{
						printf("Read is faster than write,pause(2),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,2,tm_skip,Cdata_fp);
					}
					chnum *= 4;
					for(i=0;i<tm_skip;i++)
					{	
						dat_tmp[i] >>= chnum;
						*(pIData+i) = dat_tmp[i]&3;
						*(pQData+i) = (dat_tmp[i]&12)/4;
					}
					break;
				}

				case 4:
				{
					int dat_tmp[8000];
					memset(dat_tmp,0,8000*sizeof(int));
					read_error = fread(dat_tmp,4,tm_skip,Cdata_fp);
					while(read_error!=tm_skip)
					{
						printf("Read is faster than write,pause(3),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,4,tm_skip,Cdata_fp);
					}
					chnum *= 8;
					for(i=0;i<tm_skip;i++)
					{
						dat_tmp[i] >>= chnum;
						*(pIData+i) = dat_tmp[i]&15;
						*(pQData+i) = (dat_tmp[i]&240)/16;
					}
					break;
				}
				case 8:
				{
					char dat_tmp[32000];
					memset(dat_tmp,0,32000*sizeof(char));
					//tm_skip *= 8;
					read_error = fread(dat_tmp,1,tm_skip*8,Cdata_fp);
					while(read_error!=tm_skip*8)
					{
						printf("Read is faster than write,pause(4),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,1,tm_skip*8,Cdata_fp);
					}
					chnum *= 2;
					for(i=0;i<tm_skip;i++)
					{
						*(pIData+i) = dat_tmp[i*8+chnum];
						*(pQData+i) = dat_tmp[i*8+chnum+1];
					}
					break;
				}
			case 16:
				{
					short int dat_tmp[16000];
					memset(dat_tmp,0,16000*sizeof(short int));
					//tm_skip *= 8;
					read_error = fread(dat_tmp,2,tm_skip*8,Cdata_fp);
					while(read_error!=tm_skip*8)
					{
						printf("Read is faster than write,pause(5),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,2,tm_skip*8,Cdata_fp);
					}
					chnum *= 2;
					for(i=0;i<tm_skip;i++)
					{
						*(pIData+i) = dat_tmp[i*8+chnum];
						*(pQData+i) = dat_tmp[i*8+chnum+1];
					}
					break;
				}
			default :
				printf("Illegal data format: Bit Error\n");
				return(0);
		}
		pIData += tm_skip;
		pQData += tm_skip;
		fread(Tail_words,4,188,Cdata_fp);
		Cdatafp_Position = ftello(Cdata_fp)/32768;
		while(feof(Cdata_fp)!=0&&F_end<10)
		{
			sleep(5000);
			if(1==fread(&i,1,1,Cdata_fp))
			{
				fseeko(Cdata_fp,-1,1);
				F_end = 0;
				break;
			}
			F_end ++;
		}
		Length = -1*RemNum;  //此时既没有读取整数包也没有读入小数包部分
		if(F_end==9) 
		{
			printf("Here is the end of file!\n");
			Chg_file = 1;
			return(Length);
		}
	}

	tm_skip = 0;
	int *LpIData = pIData;
	int *LpQData = pQData;
	return(Read_dat(tm_skip,BlkNum,Ch_No,pIData,pQData,Length,RemNum));
}


int CRSRDATA::Read_dat(int tm_skip,int BlkNum,int Ch_No,int *pIData,int *pQData,int Length,int RemNum)
{
	int blkcount = 0;
	while(blkcount<BlkNum)
	{
		int F_end = 0;
		int read_error = 0;
		int chnum = Ch_No-1;
		int i=0;
		read_error = fread(Head_words,1,16,Cdata_fp);
		while(read_error!=16)
		{
			printf("Read is faster than write,pause(6),%d,%d\n",read_error,tm_skip);
			fseeko(Cdata_fp,-1*read_error,1);
			sleep(1000);
			read_error = fread(Head_words,1,16,Cdata_fp);
		}
		switch(Data_info.Cdata_Bit) //4 2 1 三种比特率处理，将8通道合并读出.比特率8 16，分开读出
		{
			case 1: 
				{
					char dat_tmp[32000];
					memset(dat_tmp,0,32000*sizeof(char));
					read_error = fread(dat_tmp,1,32000,Cdata_fp);
					while(read_error!=32000)
					{
						printf("Read is faster than write,pause(7),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,1,32000,Cdata_fp);
					}
					chnum *= 2;
					for(i=tm_skip;i<32000;i++)
					{
						dat_tmp[i] >>= chnum;
						*(pIData+i-tm_skip) = dat_tmp[i]&1;
						*(pQData+i-tm_skip) = (dat_tmp[i]&2)/2;
					}
					pIData += 32000-tm_skip;
					pQData += 32000-tm_skip;
					tm_skip = 0;
					break;
				}
			case 2:
				{
					short int dat_tmp[16000];
					memset(dat_tmp,0,16000*sizeof(short int));
					read_error = fread(dat_tmp,2,16000,Cdata_fp);
					while(read_error!=16000)
					{
						printf("Read is faster than write,pause(8),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-2*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,2,16000,Cdata_fp);
					}
					chnum *= 4;
					for(i=tm_skip;i<16000;i++)
					{
						dat_tmp[i] >>= chnum;
						*(pIData+i-tm_skip) = dat_tmp[i]&3;
						*(pQData+i-tm_skip) = (dat_tmp[i]&12)/4;
					}
					pIData += 16000-tm_skip;
					pQData += 16000-tm_skip;
					tm_skip = 0;
					break;
				}
			case 4:
				{
					int dat_tmp[8000];
					memset(dat_tmp,0,8000*sizeof(int));
					read_error = fread(dat_tmp,4,8000,Cdata_fp);
					while(read_error!=8000)
					{
						printf("Read is faster than write,pause(9),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-4*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,4,8000,Cdata_fp);
					}
					chnum *= 8;
					for(i=tm_skip;i<8000;i++)
					{
						dat_tmp[i] >>= chnum;
						*(pIData+i-tm_skip) = dat_tmp[i]&15;
						*(pQData+i-tm_skip) = (dat_tmp[i]&240)/16;
					}
					pIData += 8000-tm_skip;
					pQData += 8000-tm_skip;
					tm_skip = 0;
					break;
				}
			case 8:
				{
					char dat_tmp[32000];
					memset(dat_tmp,0,32000*sizeof(char));
					read_error = fread(dat_tmp,1,32000,Cdata_fp);
					while(read_error!=32000)
					{
						printf("Read is faster than write,pause(10),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,1,32000,Cdata_fp);
					}
					chnum *= 2;
					for(i=tm_skip;i<4000;i++)
					{
						*(pIData+i-tm_skip) = dat_tmp[i*8+chnum];
						*(pQData+i-tm_skip) = dat_tmp[i*8+chnum+1];
					}
					pIData += 4000-tm_skip;
					pQData += 4000-tm_skip;
					tm_skip = 0;
					break;
				}
			case 16:
				{
					short int dat_tmp[16000];
					memset(dat_tmp,0,16000*sizeof(short int));
					read_error = fread(dat_tmp,2,16000,Cdata_fp);
					while(read_error!=16000)
					{
						printf("Read is faster than write,pause(11),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-2*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,2,16000,Cdata_fp);
					}
					chnum *= 2;
					for(i=tm_skip;i<2000;i++)
					{
						*(pIData+i-tm_skip) = dat_tmp[i*8+chnum];
						*(pQData+i-tm_skip) = dat_tmp[i*8+chnum+1];
					}
					pIData += 2000-tm_skip;
					pQData += 2000-tm_skip;
					tm_skip = 0;
					break;
				}
			default :
				printf("Illegal data format: Bit Error\n");
				return(-1);
		}
		fread(Tail_words,4,188,Cdata_fp);
		Cdatafp_Position = ftello(Cdata_fp)/32768;
		do
		{
			if(1!=fread(&i,1,1,Cdata_fp))
			{
				sleep(2000);
				F_end ++;
			}
			else
			{
				fseeko(Cdata_fp,-1,1);
				F_end = 0;
				break;
			}
		}while(feof(Cdata_fp)!=0&&F_end<10);
		blkcount ++;
		Length = blkcount*32000/Data_info.Cdata_Bit-RemNum;  //此时还没有读入小数包部分顾暂时不加This_residue.Last_ReadNum
		if(F_end==10) 
		{
			printf("Here is the end of file!\n");
			Chg_file = 1;
			return(Length);
		}
	}



	if(blkcount==BlkNum&&This_residue.Last_ReadNum!=0&&!Chg_file)
	{
		int F_end = 0;
		int read_error = 0;
		int chnum = Ch_No-1;
		int i=0;
		tm_skip = This_residue.Last_ReadNum; 
		read_error = fread(Head_words,1,16,Cdata_fp);
		while(read_error!=16)
		{
			printf("Read is faster than write,pause(12),%d,%d\n",read_error,tm_skip);
			fseeko(Cdata_fp,-1*read_error,1);
			sleep(1000);
			read_error = fread(Head_words,1,16,Cdata_fp);
		}
		switch(Data_info.Cdata_Bit) //4 2 1 三种比特率处理，将8通道合并读出.比特率8 16，分开读出
		{
			case 1:  
				{
					char dat_tmp[32000];
					memset(dat_tmp,0,32000*sizeof(char));
					read_error = fread(dat_tmp,1,tm_skip,Cdata_fp);
					while(read_error!=tm_skip)
					{
						printf("Read is faster than write,pause(13),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,1,tm_skip,Cdata_fp);
					}
					chnum *= 2;
					for(i=0;i<tm_skip;i++)
					{
						dat_tmp[i] >>= chnum;
						*(pIData+i) = dat_tmp[i]&1;
						*(pQData+i) = (dat_tmp[i]&2)/2;
					}
					break;
				}
			case 2:
				{
					short int dat_tmp[16000];
					memset(dat_tmp,0,16000*sizeof(short int));
					read_error = fread(dat_tmp,2,tm_skip,Cdata_fp);
					while(read_error!=tm_skip)
					{
						printf("Read is faster than write,pause(14),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-2*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,2,tm_skip,Cdata_fp);
					}
					chnum *= 4;
					for(i=0;i<tm_skip;i++)
					{	
						dat_tmp[i] >>= chnum;
						*(pIData+i) = dat_tmp[i]&3;
						*(pQData+i) = (dat_tmp[i]&12)/4;
					}
					break;
				}
			case 4:
				{
					int dat_tmp[8000];
					memset(dat_tmp,0,8000*sizeof(int));
					read_error = fread(dat_tmp,4,tm_skip,Cdata_fp);
					while(read_error!=tm_skip)
					{
						printf("Read is faster than write,pause(15),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-4*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,4,tm_skip,Cdata_fp);
					}
					chnum *= 8;
					for(i=0;i<tm_skip;i++)
					{
						dat_tmp[i] >>= chnum;
						*(pIData+i) = dat_tmp[i]&15;
						*(pQData+i) = (dat_tmp[i]&240)/16;
					}
					tm_skip = 0;
					break;
				}
			case 8:
				{
					char dat_tmp[32000];
					memset(dat_tmp,0,32000*sizeof(char));
					read_error = fread(dat_tmp,1,tm_skip*8,Cdata_fp);
					while(read_error!=tm_skip*8)
					{
						printf("Read is faster than write,pause(16),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-1*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,1,tm_skip*8,Cdata_fp);
					}
					chnum *= 2;
					for(i=0;i<tm_skip;i++)
					{
						*(pIData+i) = dat_tmp[i*8+chnum];
						*(pQData+i) = dat_tmp[i*8+chnum+1];
					}
					tm_skip = 0;
					break;
				}
			case 16:
				{
					short int dat_tmp[16000];
					memset(dat_tmp,0,16000*sizeof(short int));
					read_error = fread(dat_tmp,2,tm_skip*8,Cdata_fp);
					while(read_error!=tm_skip*8)
					{
						printf("Read is faster than write,pause(17),%d,%d\n",read_error,tm_skip);
						fseeko(Cdata_fp,-2*read_error,1);
						sleep(1000);
						read_error = fread(dat_tmp,2,tm_skip*8,Cdata_fp);
					}
					chnum *= 2;
					for(i=0;i<tm_skip;i++)
					{
						*(pIData+i) = dat_tmp[i*8+chnum];
						*(pQData+i) = dat_tmp[i*8+chnum+1];
					}
					tm_skip = 0;
					break;
				}
			default :
				printf("Illegal data format: Bit Error\n");
				return(-1);
		}
		This_residue.Last_Remaind = 32000/Data_info.Cdata_Bit-This_residue.Last_ReadNum;
		Cdatafp_Position = ftello(Cdata_fp)/32768;
		Length = blkcount*32000/Data_info.Cdata_Bit-RemNum+This_residue.Last_ReadNum;//此时已读入小数部分
	}

	Cdatafp_Position = ftello(Cdata_fp)/32768;
//	printf( "Read %d packages ,start from %ld\n",Cdatafp_Position,Data_start/32768);
	get_UTC();

	return(Length);
}
