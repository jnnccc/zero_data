#define _FILE_OFFSET_BITS 64
#include<iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "CRSRData.h"
using namespace std;
extern "C"
       {
char 	module10_mp_utcsta_[23],module10_mp_utcsto_[23],module10_mp_utc1_[23],module10_mp_utc2_[23];
int		module10_mp_sampling_;
/*struct {
		char datafile_[80];
		int mode_,span2_,chanel_;
		float amp_win_,span1_;
 		} param_mp_rsr_data_par_	;	*/
void  	rsrinfo_(char *datafile);
void 	getrsrdata_(int datai[],int dataq[],char *datafile,int *chanel);
}

void rsrinfo_(char *datafile)
{	
//	CRSRDATA RsrData(param_mp_rsr_data_par_.datafile_);
	CRSRDATA RsrData(datafile);
	module10_mp_sampling_=RsrData.Data_info.Cdata_Rate;
/*	cout<<"Array number:"<<RsrData.Data_info.Cdata_Rate<<endl;
	cout<<"Array number:"<<RsrData.Data_info.Cdata_Bit<<endl;
	cout<<"Array number:"<<RsrData.Data_info.Cdata_NCO[0]<<endl;
	cout<<"Array number:"<<RsrData.Data_info.Cdata_NCO[1]<<endl;
	cout<<"Array number:"<<RsrData.Data_info.Cdata_NCO[2]<<endl;
	cout<<"Array number:"<<RsrData.Data_info.Cdata_NCO[3]<<endl;
	cout<<"Array number:"<<RsrData.Data_info.Interval<<endl;
	cout<<"Array number:"<<RsrData.Data_info.blkpt<<endl;
	cout<<"Array number:"<<RsrData.Data_info.StartUTC<<endl;
	cout<<"Array number:"<<RsrData.Data_info.StopUTC<<endl;*/
//	cout<<"data file:"<<datafile<<endl;
//	data_process_mp_utcsta_=RsrData.Data_info.StartUTC;
//	data_process_mp_utcsto_=RsrData.Data_info.StopUTC;
//	utc2=RsrData.Data_info.StopUTC;
//	cout<<utc2<<endl;
//	cout<<utc1<<endl;
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

//para_mp_doppler0_par_.sampling_=RsrData.Data_info.Cdata_Rate;
//	cout<<para_mp_rsr_data_par_.read_span<<endl;
}


void getrsrdata_(int datai[],int dataq[],char *datafile,int *chanel)
{
CRSRDATA RsrData(datafile);	
//	CRSRDATA RsrData(param_mp_rsr_data_par_.datafile_);
//	char *Start_UTC = RsrData.Data_info.StartUTC;
//    char *Stop_UTC = RsrData.Data_info.StopUTC;
//    Start_UTC = "2011-12-09T09:31:27.000";
//     Stop_UTC = "2011-12-09T09:31:28.000";
//	utc1=RsrData.Data_info.StartUTC;
//	utc2=RsrData.Data_info.StopUTC ;
//int *sample, pIData[*sample], pQData[*sample];
//    int *pIData = (int *) malloc (sizeof(int)**sample);
//    int *pQData = (int *) malloc (sizeof(int)**sample);
//    memset(pIData,0,*sample*sizeof(int));
//    memset(pQData,0,*sample*sizeof(int));
//	RsrData.Cdata_read(utc1,utc2,2,pIData,pQData);
    RsrData.Cdata_read(module10_mp_utc1_,module10_mp_utc2_,*chanel,datai,dataq);
//    for(int i = 0; i <= 9; i++)
//       printf("%d\n",datai[i]);
//	cout<<data_process_mp_utc1_<<endl;
//	cout<<data_process_mp_utc2_<<endl;
//	ctof_mp_dataI_=*pIData;
//	ctof_mp_dataQ_=*pQData;
}
