module utilmod
use param
!use fund_const
!USE orbit_elements

private tictime, toctime
double precision::tictime, toctime

contains

subroutine tic
  integer(8) C,R,M
    CALL SYSTEM_CLOCK (COUNT=C, COUNT_RATE=R, COUNT_MAX=M)
    tictime = dble(C)/dble(R)

endsubroutine

subroutine toc
  integer(8) C,R,M
    CALL SYSTEM_CLOCK (COUNT=C, COUNT_RATE=R, COUNT_MAX=M)
    toctime = dble(C)/dble(R)
  write(*,*) toctime-tictime
endsubroutine

subroutine toc1
  integer(8) C,R,M
    CALL SYSTEM_CLOCK (COUNT=C, COUNT_RATE=R, COUNT_MAX=M)
    toctime = dble(C)/dble(R)
  write(idebug,*) "one block use : ",toctime-tictime," seconds"
endsubroutine


subroutine combtime(mjd,sec,mjdd)
integer,intent(in):: mjd
DOUBLE PRECISION,intent(in)::sec
DOUBLE PRECISION,intent(out):: mjdd
mjdd=dble(mjd)+sec/86400.0d0
endsubroutine

subroutine splittime(jd,mjd,sec)
DOUBLE PRECISION,intent(in)::jd
integer,intent(out)::mjd
DOUBLE PRECISION,intent(out)::sec
DOUBLE PRECISION temp
if(jd.gt.2400000.5d0)then
     temp=jd-2400000.5d0
else
     temp=jd
endif
mjd=int(temp)
sec=(temp-dble(mjd))*86400.0d0
endsubroutine



subroutine markhere()
write(*,*)'                                                '
write(*,*)'                                                '
write(*,*)'                                                '
write(*,*)'------------12345678901234567890-------------|--'

endsubroutine markhere



subroutine scanfile(fname,N)
implicit none
CHARACTER(len=*)::fname
integer*8 N
open(33,file=fname)
N=0
do while(0<1)
	READ(33,*,END=124)cTEMP
     N=N+1
enddo
124 close(33) 

endsubroutine scanfile

subroutine deg2dms(deg,ideg,imin,isec)
     integer ideg,imin,isec
     real*8 deg
        ideg=floor(deg)
        imin=floor((deg-ideg)*60d0)
        isec=floor(((deg-ideg)*60d0-imin)*60d0)

endsubroutine deg2dms

!     Last change:  OO    7 Feb 1999    1:56 pm                         
!     Subroutine for computing time of week (in seconds) and GPS-week   
!     from Gregorian date, hour, min and sec.                           
!     (NOTE: Leap seconds are disregarded here, GPS = UTC + leap secs)  
!     GPS-week=0 , time of week=0  at Sunday 06.01.1980 00:00 o'clock   
!                                                                       
!     Author  : Ola Ovstedal                                            
!     Created : 06.08.1991                Last modified: 07.02.1999     
!     ------------------------------------------------------------------
!     Changes:                                                          
!     07.02.1999: Changed INTEGER*2 --> INTEGER, AND INT2 --> INT       
!                 Compiled With Lahey F90 4.5                           
!                 Use 4 digits for year                                 
!     ------------------------------------------------------------------
      SUBROUTINE GPSTIME (year, month, day, hour, min, sec, week, tow) 
!     INTEGER*2 year,month,day,hour,min,mday(12),i,nday  ! 4/2.99       
      INTEGER year, month, day, hour, min, mday (0:12), i, nday 
      REAL(8) sec, tow, doweek, week 
      DATA (mday (i), i = 0, 12) / 0, 31, 59, 90, 120, 151, 181, 212,   &
      243, 273, 304, 334, 365 /                                         
!     ***Tot. nr of days since 05.01.1980*******************************
                                                                ! 7/2.99
      nday = (year - 1980) * 365 + mday (month - 1) + day - 5 
!     **Correction for leap-years***************************************
                                                                ! 7/2.99
      nday = nday + (year - 1980) / 4 
                                                                ! 7/2.99
      IF (MOD ( (year - 1980), 4) .EQ.0.AND.month.LT.3) nday = nday - 1 
!     **GPS-week, day of week and tow (sec)*****************************
!     week=INT2(nday/7)             ! 4/2.96                            
      week = INT (nday / 7) 
      doweek = ( (nday / 7.0D0) - week) * 7.0D0 + 1.0D0 
      tow = (doweek - 1.0D0) * 86400.0D0 + hour * 3600.0D0 + min *      &
      60.0D0 + sec                                                      
      RETURN 
      END SUBROUTINE GPSTIME              

subroutine et2mjd(et,mjd)
implicit none
real*8::et,mjd
CALL ET2UTC ( ET, 'J', 7, ctemp )
READ(ctemp,'(3x,f22.7)',err=112)mjd
mjd=mjd-2400000.5d0
goto 113
112 write(*,*)'time error!'
113 continue
endsubroutine et2mjd

subroutine mjd2et(mjd,et)
implicit none
real*8::et,mjd
mjd=mjd+2400000.5d0
write(ctemp,*)mjd
ctemp='jd '//ctemp
CALL STR2ET ( ctemp, ET )
write(*,*)et
endsubroutine mjd2et



endmodule utilmod          
