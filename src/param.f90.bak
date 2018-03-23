module param
IMPLICIT NONE

!public read_driv,eph_gen,occultation_gen,GLOBAL_PAR!,tracking_gen,doppler_gen,occul_gen
!private EPH_PAR, TRACKING_PAR,OCCULTATION_PAR,DOPPLER_PAR,driv_id,driv_length,itemp,dtemp,dtemp1,cte
!private FRAME, ABCORR,KER,out_file, ntarget, obsrvr, utc0,utc1,spk_id,et0,&
!&step
integer,parameter :: sp = selected_real_kind(p=6,r=37)
integer,parameter :: dp = selected_real_kind(p=15,r=307)
integer,parameter :: qp = selected_real_kind(p=33,r=4931)
real(dp),parameter :: au= 149597870.693d0

integer :: runmode
integer :: itemp11,itemp12,itemp13,itemp14,itemp15,itemp16,itemp17,itemp18,itemp19
integer*8::itemp,itemp1,itemp2,itemp3,itemp4,itemp5,itemp6,itemp7,itemp8,itemp9,N_data,m_data
integer*4,parameter::IB=4,rp=8,idebug=119,driv_id=20,driv_id0=21,iout=21,iout1=22,driv_length=500
real*8::dtemp,dtemp0,dtemp1,dtemp2,dtemp3,dtemp4,dtemp5,dtemp6,dtemp7,dtemp8,dtemp9,s6(6),p3(3),mtemp1(3,3),mtemp2(3,3),delta_t=0.d0,time_tag
real(dp),parameter::xnorm(3)=(/1.d0,0.d0,0.d0/),ynorm(3)=(/0.d0,1.d0,0.d0/),znorm(3)=(/0.d0,0.d0,1.d0/) 
!DOUBLE PRECISION,ALLOCATABLE::t(:),s0(:),s1(:),jt(:),js0(:),js1(:)
real*8,allocatable::data1(:),data2(:),data3(:),data4(:),data5(:),data6(:)
real*16::dtempk
CHARACTER*80::ctemp,ctemp0,ctemp1,ctemp2,ctemp3
!DOUBLE PRECISION:: ET0,step
!积分涉及到的天体信息：nb(1)-天体数目，nb(2,:)-天体编号

type integ_par
	integer,allocatable::nb(:)
	integer*4 ::mode,srp,relat
	integer::order1
	integer::order2
	real*8 cd
	real*8,allocatable::gm(:)
	character*80::center
end type integ_par

type ksg_par
	integer*4 :: n,m,iflag,nloop
	logical ::zzdt
	real*8::h,alim(3),work(1000000)
end type ksg_par


TYPE GLOBAL1_PAR
character*2::RUNMODE
character*8::taskid
CHARACTER*50::KER,target,driv_name
END TYPE GLOBAL1_PAR

TYPE STA1_PAR
DOUBLE PRECISION::PRES,RHUM,CELS,LAT,sta_hei,freq
END TYPE STA1_PAR

TYPE AUXCOM1_PAR
integer::mode
CHARACTER*30::utc0,utc1
character*20:: targ1,targ2,targ3,targ4,targ5,targ6
DOUBLE PRECISION::step
integer*4::nb
INTEGER,allocatable::BID(:)
END TYPE AUXCOM1_PAR


TYPE EST1_PAR
integer :: mode,NP,itermax,dim_xc
DOUBLE PRECISION,ALLOCATABLE:: xcmin(:),xcmax(:)
character*4::sampling
CHARACTER*80::infile,UPLINK_STATION,DOWNLINK_STATION,span
real(kind=16):: TRANS_RATIO
END TYPE EST1_PAR


TYPE EPH1_PAR
integer::type
CHARACTER*50::FRAME,CENTER,utc0,utc1,TARGET
DOUBLE PRECISION::step,tz
END TYPE EPH1_PAR

TYPE TRACKING1_PAR
CHARACTER*50::utc0,utc1,TRACKING_STA,dir
DOUBLE PRECISION::altlim,dt
integer::defor
END TYPE TRACKING1_PAR

TYPE DOPPLER1_PAR
integer::TRACKING_MODE
CHARACTER*50::utc0,utc1,UPLINK_STATION,DOWNLINK_STATION
DOUBLE PRECISION::step_pre,step_integ,TRANS_FREQ
real(kind=16):: TRANS_RATIO
CHARACTER*1::TRANS_band
END TYPE DOPPLER1_PAR

TYPE MKSPK1_PAR
INTEGER(KIND=4)::FILE_TYPE
CHARACTER*50::SETUP_FILE,INPUT_FILE,OUTPUT_FILE
END TYPE MKSPK1_PAR

TYPE OCCULTATION1_PAR
CHARACTER*50::utc0,utc1,front_target,front_shape,front_frame,back_target,back_shape,back_frame,TRACKING_STA
DOUBLE PRECISION::step
END TYPE OCCULTATION1_PAR

TYPE OCC_LEVEL021_PAR
CHARACTER*50:: obsfreq,FILENAME
END TYPE OCC_LEVEL021_PAR

TYPE RSR_data1_PAR
CHARACTER*80:: DATAFILE
character(len=24)::arb0time,arb1time
INTEGER::mode,chanel
real::amp_win,span1,span2
integer*4::datatype,itermax
real*8::xcmin(6),xcmax(6) 
END TYPE RSR_data1_PAR

TYPE PRO141_PAR
CHARACTER*50::utc0,utc1,UPLINK_STATION,DOWNLINK_STATION
INTEGER(kind=2)::mode,runmode
INTEGER(kind=8)::sample,itermax,np
real(kind=8)::freq,span,XCmin(5),XCmax(5)
END TYPE PRO141_PAR


TYPE PRO151_PAR
CHARACTER*50::utc0,utc1,UPLINK_STATION,DOWNLINK_STATION
INTEGER(kind=2)::mode,NB
INTEGER,allocatable::BID(:)
real(kind=8)::span,dt
END TYPE PRO151_PAR

TYPE PRO171_PAR
CHARACTER*50::utc0,utc1,UPLINK_STATION,DOWNLINK_STATION
INTEGER(kind=4)::mode,NB
INTEGER,allocatable::BID(:)
real(kind=8)::span,dt,coord(6)
END TYPE PRO171_PAR


TYPE PRO181_PAR
CHARACTER*50::utc0,utc1
INTEGER(kind=4)::NB,mode
INTEGER,allocatable::BID(:)
real(kind=8)::step,coord(6)
END TYPE PRO181_PAR

TYPE(GLOBAL1_PAR)::GLOBAL_PAR
TYPE(EPH1_PAR)::EPH_PAR
TYPE(EST1_PAR)::EST_PAR
TYPE(TRACKING1_PAR)::TRACKING_PAR
TYPE(DOPPLER1_PAR)::DOPPLER_PAR
TYPE(OCCULTATION1_PAR)::OCCULTATION_PAR
TYPE(OCC_LEVEL021_PAR)::OCC_LEVEL02_PAR
TYPE(STA1_PAR)::STA_PAR
TYPE(MKSPK1_PAR)::MKSPK_PAR
TYPE(rsr_data1_PAR)::rsr_data_PAR
TYPE(AUXCOM1_PAR)::AUXCOM_PAR
TYPE(PRO141_PAR)::PRO14_PAR
TYPE(PRO151_PAR)::PRO15_PAR
TYPE(PRO171_PAR)::PRO17_PAR
TYPE(PRO181_PAR)::PRO18_PAR
type(ksg_par)::ksg
type(integ_par)::integ
endmodule param 

