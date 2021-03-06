module module10
use ftn
use plot_module ,only: plot1
use lib_array
use param
use utilmod
use math_lib
use iso_c_binding, only: C_CHAR, c_null_char
implicit none
!private
!public program10
!public utcsta,utcsto,utc1,utc2,datai,dataq
character(kind=c_char,len=23), bind(C, name="module10_mp_utcsta_") :: utcsta
character(kind=c_char,len=23), bind(C, name="module10_mp_utcsto_") :: utcsto
character(kind=c_char,len=23), bind(C, name="module10_mp_utc1_") :: utc1
character(kind=c_char,len=23), bind(C, name="module10_mp_utc2_") :: utc2
integer, bind(C, name="module10_mp_sampling_") :: sampling
integer(kind=4)::ihealth,inorm
integer,allocatable::datai(:),dataq(:),s0(:)
real*8,allocatable::tt(:)
real*8::phtemp,amptemp
real*8,parameter:: perr=1d-5
private
public program10, utcsta,utcsto,utc1,utc2,datai,dataq!,program10
contains

subroutine program10
implicit none
RSR_data_PAR%DATAFILE=trim(RSR_data_PAR%DATAFILE)//C_NULL_CHAR
!open (idebug,file='log')
CALL FURNSH ( 'input/driv.ker')
select CASE (rsr_data_PAR%mode)
case (0)
	call rsrinfo(RSR_data_PAR%DATAFILE)
case (1)
	call rsrinfo(RSR_data_PAR%DATAFILE)
	N_data=sampling
	call ftn_set_length(N_data)
!	write(*,*)sampling
	allocate(datai(N_data),dataq(N_data))
	write(*,*)utcsta,utcsto
	call data_process_1
case (2,4)
	call rsrinfo(RSR_data_PAR%DATAFILE)
	N_data=sampling*RSR_data_PAR%span2
	call ftn_set_length(N_data)
	allocate(tt(N_data),datai(N_data),dataq(N_data),s0(N_data))
	call data_process_2
case (3)
	call rsrinfo(RSR_data_PAR%DATAFILE)
	N_data=sampling*RSR_data_PAR%span2+1
	allocate(datai(N_data),dataq(N_data))
     call data_process_3
case default
    write(*,*) 'wrong data process mode'
end select
CALL UNLOAD ( 'input/driv.ker')

close(idebug)
end subroutine program10


subroutine data_process_1
implicit none
double precision ET ,ET0,ET1
real*4 f0,f1,df
real*4,ALLOCATABLE::f(:),m(:),t(:)
real dt,sp
integer*8 I,N,ind(1)
integer pgopen
character*200 disp1,space1,disp2,space2
character*8 str
sp=rsr_data_PAR%span1
df=1./sp
f0=0.
f1=sampling/2.
itemp=floor((f1-f0)/df)
allocate(f(itemp),m(itemp))
itemp=sampling*sp
allocate(t(itemp))
if (rsr_data_PAR%arb0time(1:1)=='1') utcsta=rsr_data_PAR%arb0time(2:24)
if (rsr_data_PAR%arb1time(1:1)=='1') utcsto=rsr_data_PAR%arb1time(2:24)	
!ctemp=rsr_data_PAR%arbtime
!write(*,*)ctemp(1:1)
!if (ctemp(1).eq.'1') utcsta=ctemp(2:24)
CALL STR2ET ( utcsta, ET0 )
CALL STR2ET ( utcsto, ET1 )
write(*,*) 'data time from '//utcsta//' to'//utcsto
!write(*,*) utcsto
!write(*,*) ET0,ET1
ET0=dble(floor(ET0)+1)+0.183929676d0
ET1=dble(floor(ET1)-1)+0.183929676d0
dtemp=et0
N=floor(ET1-ET0)
call linspace(f0,f1,f)
call linspace(0.0,sp,t(1:itemp))
!call ET2UTC ( 0.d0,'ISOC', 3, utcsta )
IF (PGOPEN('/Xwin') .LE. 0) STOP
call PGSCH(1.2) 	! 字体大小
call PGSCF (2)		! 字体种类
!space='"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'
!space1='"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'
!space1='"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'

do i=1,N
	et0=dtemp+dble(i-1)
	call ET2UTC ( et0,'ISOC', 3, utc1 )
	call ET2UTC ( et0+1.d0,'ISOC', 3, utc2 )
     utc1=utc1(1:20)//'000'!'2011-12-09T08:37:14.000'
     utc2=utc2(1:20)//'000'!'2011-12-09T08:37:16.000'
	
	call getrsrdata(datai,dataq,RSR_data_PAR%DATAFILE,RSR_data_PAR%chanel)
!write(*,*)dataq(1:10)
!stop

!write(*,*)utc1
!write(*,*)'----------------------------'
!write(*,*)datai(1:10)
select case(RSR_data_PAR%datatype)
case(0)
	call fftw(t(1:itemp),datai(1:itemp),f,m)
case(1)
	call fftw(t(1:itemp),dataq(1:itemp),f,m)
case(2)
	call fftw(t(1:itemp),datai(1:itemp)+dataq(1:itemp),f,m)
case default
     write(*,*) 'wrong data type'
end select
   	write(ctemp,'(f5.1)') i*100.d0/N 
	ind=maxloc(m)
	write( str,'(f8.0)') f(ind(1))
!
	call plot1(f,m,RSR_data_PAR%amp_win,RSR_data_PAR%chanel,utc1(1:19),str)
!   	write(ctemp,'(f5.1)') i*100.d0/N 
!	ind=maxloc(m)
!	write( str,'(f8.0)') f(ind(1))
!	write(*,*)trim(ctemp1)	
!    	disp='echo '//trim(space)//'progress: '//trim(adjustl(ctemp))//'%      data time:'//utc1(1:19)//'\c"' !b的数目和字宽有关，这里是4
!!!	write(*,*)ctemp0
!	disp='echo '//trim(space)//'progress: '//trim(adjustl(ctemp))//'%      data time:'//utc1(1:19)//'   frequecy: '//trim(ctemp1)//'\c"'

! disp1='echo '//trim(space1)//'progress:'//trim(adjustl(ctemp))//'% data time: '//utc1(1:19)//' frequnecy:'//str//'\c"'
!call system(disp1)

enddo

end subroutine data_process_1

subroutine data_process_2
implicit none
integer(kind=4) :: NP,dim_xc,strategy,refresh,nfeval,ii
integer (kind=8)::i,N,itermax
integer(kind=4), dimension(3), parameter :: method=(/0, 0, 0/)
DOUBLE PRECISION:: bestmem_XC(7),vtr,cr_xc,f_xc,f_cr,bestval
double precision ::ET ,ET0,ET1,etemp
CHARACTER *80:: basename
CALL GETARG(1,basename)

      np=300
  dim_xc=7
strategy=3  !2,3,4,5 皆可选 3的收敛速度快点,2,4,5的速度较慢
! refresh=50
     vtr=0.d0
   cr_xc=0.85d0
    f_xc=0.5d0
    f_cr=1.d0

!边界初始化
RSR_data_PAR%XCmin(1)=0.d0
RSR_data_PAR%XCmax(1)=2.d0*pi

write(ctemp,'(i1)')rsr_data_PAR%chanel
write(ctemp0,'(i1)')RSR_data_PAR%datatype
write(ctemp1,"(f8.1)") rsr_data_PAR%SPAN2
if (rsr_data_PAR%mode .eq. 2) then
open(iout,file='output/'//trim(basename)//'_'//'ch'//ctemp(1:1)//'_'//ctemp0(1:1)//'_'//trim(ADJUSTL(adjustr(ctemp1))))
open (idebug,file='log/'//trim(basename)//'_'//'ch'//ctemp(1:1)//'_'//ctemp0(1:1)//'_'//trim(ADJUSTL(adjustr(ctemp1)))//'.log')
else
open(iout,file='output/debug.out')
open(idebug,file='log/debug.log')
endif
!utcsta='2011-12-09T08:39:19.000'
if (rsr_data_PAR%arb0time(1:1)=='1') utcsta=rsr_data_PAR%arb0time(2:24)
if (rsr_data_PAR%arb1time(1:1)=='1') utcsto=rsr_data_PAR%arb1time(2:24)
CALL STR2ET ( utcsta, ET0 )
CALL STR2ET ( utcsto, ET1 )
!call markhere
write(*,*) 'data time from '//utcsta//' to'//utcsto
ET0=dble(floor(ET0)+1)+dble(ET0-floor(ET0))
ET1=dble(floor(ET1)-1)+dble(ET1-floor(ET1))
etemp=et0
N=floor((ET1-ET0)/RSR_data_PAR%span2)
call linspace(-dble(RSR_data_PAR%span2)/2.d0,dble(RSR_data_PAR%span2)/2.d0-1.d0/dble(sampling),tt)
call ftn_set_tt(tt(1))
ihealth=0
inorm=0
bestmem_XC=0.d0
CALL random_seed()
do i=1,N
call tic
    write(*,*)RSR_data_PAR%XCmin
    write(*,*)RSR_data_PAR%XCmax
    et0=etemp+RSR_data_PAR%span2*dble(i-1)
    et1=etemp+RSR_data_PAR%span2*dble(i)
    call ET2UTC ( et0,'ISOC', 3, utc1 )
    call ET2UTC ( et1,'ISOC', 3, utc2 )
!     call ET2UTC ( et0,'ISOC', 3, '2011-12-09T08:37:14.000' )
!     call ET2UTC ( et1,'ISOC', 3, '2011-12-09T08:37:16.000' )
!   utc1=utc1(1:20)//'000'!'2011-12-09T08:37:14.000'
!   utc2=utc2(1:20)//'000'!'2011-12-09T08:37:16.000'
    write(*,*)utc1,utc2
    call getrsrdata(datai,dataq,RSR_data_PAR%DATAFILE,RSR_data_PAR%chanel)
select case(RSR_data_PAR%datatype)
case(0)
    s0=datai
case(1)
    s0=dataq
case(2)
    s0=datai+dataq
case default
    write(*,*) 'wrong data type'
end select
call ftn_set_s0(s0(1))
if (ihealth==0) then
call check_data  !判断数据质量，并给出部分边界参数
endif

if (ihealth==1) then
write(idebug,*)utc1
write(idebug,*)RSR_data_PAR%XCmin
write(idebug,*)RSR_data_PAR%XCmax
write(idebug,*)'--------------------'
    if (inorm==0) then
        itermax=500
    else
        itermax=RSR_data_PAR%itermax
    endif

call DE_Fortran90(ftn_eval, Dim_XC, RSR_data_PAR%XCmin, RSR_data_PAR%XCmax, VTR, NP, itermax, F_XC,&
                CR_XC, strategy,RSR_data_PAR%refresh, bestmem_XC, bestval, nfeval, F_CR, method)
    if(   dabs(bestmem_XC(2)-RSR_data_PAR%XCmin(2)).le.perr.or.dabs(bestmem_XC(2)-RSR_data_PAR%XCmax(2)).le.perr.or. &
        & dabs(bestmem_XC(3)-RSR_data_PAR%XCmin(3)).le.perr.or.dabs(bestmem_XC(3)-RSR_data_PAR%XCmax(3)).le.perr.or. &
        & dabs(bestmem_XC(4)-RSR_data_PAR%XCmin(4)).le.perr.or.dabs(bestmem_XC(4)-RSR_data_PAR%XCmax(4)).le.perr.or. &
        & dabs(bestmem_XC(5)-RSR_data_PAR%XCmin(5)).le.perr.or.dabs(bestmem_XC(5)-RSR_data_PAR%XCmax(5)).le.perr.or. &
        & dabs(bestmem_XC(7)-RSR_data_PAR%XCmin(7)).le.perr.or.dabs(bestmem_XC(7)-RSR_data_PAR%XCmax(7)).le.perr.or. &
        & dabs(bestmem_XC(6)-RSR_data_PAR%XCmin(6)).le.perr.or.dabs(bestmem_XC(6)-RSR_data_PAR%XCmax(6)).le.perr           )then
!       &  abs(bestmem_XC(5)-amptemp)/amptemp .gt. 2d-1 ) then

        ihealth=0
        inorm=0
    else
     dtemp =bestmem_XC(2)+2.d0*bestmem_XC(3)*(-RSR_data_PAR%span2/2.d0)+3.d0*bestmem_XC(4)*(-RSR_data_PAR%span2/2.d0)**2
     dtemp1=bestmem_XC(2)+2.d0*bestmem_XC(3)*( RSR_data_PAR%span2/2.d0)+3.d0*bestmem_XC(4)*( RSR_data_PAR%span2/2.d0)**2

        RSR_data_PAR%XCmin(2)=bestmem_XC(2)+(dtemp1-dtemp)-20.d0
        RSR_data_PAR%XCmax(2)=bestmem_XC(2)+(dtemp1-dtemp)+20.d0
        RSR_data_PAR%XCmin(3)=bestmem_XC(3)-5.d0
        RSR_data_PAR%XCmax(3)=bestmem_XC(3)+5.d0
        if (inorm==0) then
        RSR_data_PAR%XCmin(5)=bestmem_XC(5)-500.d0
        RSR_data_PAR%XCmax(5)=bestmem_XC(5)+500.d0
        if (RSR_data_PAR%XCmin(5)<0.d0) RSR_data_PAR%XCmin(5)=0.d0
        else
          RSR_data_PAR%XCmin(5)=bestmem_XC(5)-200.d0
          RSR_data_PAR%XCmax(5)=bestmem_XC(5)+200.d0
        if (RSR_data_PAR%XCmin(5)<0.d0) RSR_data_PAR%XCmin(5)=0.d0
        endif

        inorm=1
    endif
call reout(iout,bestmem_XC,bestval,ihealth)
!bestmem_XC_temp=bestmem_XC
!amptemp=bestmem_XC(5)
else
!continue
call reout(iout,bestmem_XC,bestval,ihealth)
endif
if (inorm .eq. 1) then
    call toc1
endif


enddo

close(30)

end subroutine data_process_2


subroutine data_process_3
implicit none
integer*8::i
if (rsr_data_PAR%arb0time(1:1)=='1') utc1=rsr_data_PAR%arb0time(2:24)
if (rsr_data_PAR%arb1time(1:1)=='1') utc2=rsr_data_PAR%arb1time(2:24)
open(iout,file='output/signal4')
write(*,*)utc1,utc2
call getrsrdata(datai,dataq,RSR_data_PAR%DATAFILE,RSR_data_PAR%chanel)

do i=1,N_data
write(iout,*) datai(i),dataq(i)
enddo 
close(iout)
end subroutine data_process_3

subroutine check_data
real*8 f0,f1,df,sp,fm
real*8,ALLOCATABLE::f(:),m(:),frq(:),amp(:),res(:)
integer*8 I,j,k,ind(1),n_block
sp=rsr_data_PAR%span1
!df=2.d0/sp
!f0=0.d0
!f1=sampling/2.d0
!itemp=floor((f1-f0)/2.d0)
n_block=int(rsr_data_PAR%span2/rsr_data_PAR%span1)
if (n_block .gt. 50) n_block=50
write(*,*)n_block
allocate(frq(n_block),amp(n_block),res(n_block))
itemp1=sp*sampling
write(*,*)'checking data...'
loop1: do j=1,n_block
!allocate(f(itemp),m(itemp))
df=1.d0/(5d-1*sp)
f0=0.d0
f1=sampling/2.d0
itemp=floor((f1-f0)/df)
allocate(f(itemp),m(itemp))
call linspace(f0,f1,f)
loop2: do i=1,10
	call fftw1(tt(1:itemp1),s0((j-1)*itemp1+1:j*itemp1),f,m)
	dtemp=maxval(m)
	ind=maxloc(m)
	fm=f(ind(1))
	f0=fm-df
	f1=fm+df
	df=df/10.d0
	itemp2=floor((f1-f0)/df)
	deallocate(f,m)
	if (df<0.01) exit loop2
	allocate(f(itemp2),m(itemp2))
	call linspace(f0,f1,f)
enddo loop2
	frq(j)=fm
	amp(j)=dtemp
	write(*,*) j,fm,dtemp
enddo loop1
!call fitpol(2,frq,res)
i=0
do j=1,n_block-1
if (abs(frq(j+1)-frq(j))>10.0) then
	i=i+1
else
	k=j
endif

enddo
!如果不连续数据量大于10，则数据不可用
if (i.eq.0) then !频率无不连续点
	ihealth=1
	inorm=0
     RSR_data_PAR%xcmin(2)=(frq(k)-200.d0)*2*pi
     RSR_data_PAR%xcmax(2)=(frq(k)+200.d0)*2*pi
     RSR_data_PAR%xcmin(5)=amp(k)-500.d0
     RSR_data_PAR%xcmax(5)=amp(k)+500.d0
    
    if (RSR_data_PAR%xcmin(5)<0)    RSR_data_PAR%xcmin(5)=0.d0
else

	if(dble(i)/dble(n_block)>5d-1)  then !不连续点数目大于10%
		ihealth=0
		inorm=0
		write(*,*)'sum of not continue ponits is: ' ,i
		write(*,*)'this data block is not healthy! '
		return
	else					!不连续点数目小于10%
		ihealth=1
		inorm=0

     	RSR_data_PAR%xcmin(2)=(frq(k)-200.d0)*2*pi
     	RSR_data_PAR%xcmax(2)=(frq(k)+200.d0)*2*pi
     	RSR_data_PAR%xcmin(5)=amp(k)-500.d0
     	RSR_data_PAR%xcmax(5)=amp(k)+500.d0

		if (RSR_data_PAR%xcmin(5)<0)	RSR_data_PAR%xcmin(5)=0.d0
	endif
endif
amptemp=sum(amp)/n_block
!stop
end subroutine check_Data



subroutine reout(iwrite,bestmem_XC,bestval,ihealth)
implicit none
integer(kind=4):: iwrite,ihealth
real*8:: delta_phi,phi_l,phi_r,bestmem_XC(7),bestval,fre_l,fre_r,et
delta_phi=bestmem_XC(2)*RSR_data_PAR%span2+(bestmem_XC(4)*RSR_data_PAR%span2**3)/4.d0
phi_l=modulo(bestmem_XC(1)+bestmem_XC(2)*(-RSR_data_PAR%span2/2.d0)+bestmem_XC(3)*(-RSR_data_PAR%span2/2.d0)**2+bestmem_XC(4)*(-RSR_data_PAR%span2/2.d0)**3,2.d0*pi)
phi_r=modulo(bestmem_XC(1)+bestmem_XC(2)*( RSR_data_PAR%span2/2.d0)+bestmem_XC(3)*( RSR_data_PAR%span2/2.d0)**2+bestmem_XC(4)*( RSR_data_PAR%span2/2.d0)**3,2.d0*pi)
fre_l=bestmem_XC(2)+2.d0*bestmem_XC(3)*(-RSR_data_PAR%span2/2.d0)+3.d0*bestmem_XC(4)*(-RSR_data_PAR%span2/2.d0)**2
fre_r=bestmem_XC(2)+2.d0*bestmem_XC(3)*( RSR_data_PAR%span2/2.d0)+3.d0*bestmem_XC(4)*( RSR_data_PAR%span2/2.d0)**2
CALL STR2ET ( utc1, ET )
call ET2UTC ( et+RSR_data_PAR%span2/2.d0,'ISOC', 3, ctemp(1:23))
write(iwrite,118)utc1(1:23),phi_l,fre_l
write(iwrite,119)ctemp(1:23),bestmem_XC,delta_phi,bestval,ihealth
write(iwrite,118)utc2(1:23),phi_r,fre_r
118 FORMAT(A23,f8.3, f13.3)
119 FORMAT(A23,f8.3, f13.3,f14.6,E16.6,f10.1,f7.1,E16.6,f20.6,f15.3,i5)
!120 FORMAT(A21,f8.3, f13.3)
end subroutine reout


subroutine fftw1(t,s,f,m)
implicit none
integer*8 i,j,N,K
integer s(:)
real*8 realtemp,imgtemp,t(:),f(:),m(:)!,start,finish
K=size(f)
N=size(t)
!$OMP PARALLEL DO default(shared) PRIVATE(I,j),SCHEDULE(dynamic,10),REDUCTION(+:realtemp,imgtemp)
do i=1,K
    realtemp=0.d0
    imgtemp =0.d0
    do j=1,N
        realtemp=realtemp+s(j)*dcos(2.d0*pi*f(i)*t(j))
        imgtemp =imgtemp +s(j)*dsin(2.d0*pi*f(i)*t(j))
    enddo
    m(i)=2.d0*dsqrt(realtemp**2+imgtemp**2)/N
enddo
!$OMP END PARALLEL DO

endsubroutine fftw1


subroutine fftw(t,s,f,m)
implicit none
integer*8 i,j,N,K
integer s(:)
real*4 realtemp,imgtemp,t(:),f(:),m(:)!,start,finish
K=size(f)
N=size(t)
!$OMP PARALLEL DO default(shared) PRIVATE(I,j),SCHEDULE(dynamic,10),REDUCTION(+:realtemp,imgtemp)
do i=1,K
    realtemp=0.0
    imgtemp =0.0
    do j=1,N
        realtemp=realtemp+s(j)*cos(2.*pi*f(i)*t(j))
        imgtemp =imgtemp +s(j)*sin(2.*pi*f(i)*t(j))
    enddo
    m(i)=2.*sqrt(realtemp**2+imgtemp**2)/N
enddo
!$OMP END PARALLEL DO

endsubroutine fftw


endmodule module10
