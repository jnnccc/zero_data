module plot_module 
implicit none
private 
public plot0,plot1

contains

subroutine plot0(x,y,st)
real x(:),y(:)
integer st,PGOPEN
IF (PGOPEN('/Xwin') .LE. 0) STOP
!Call PGASK (.FALSE.)
CALL PGSCI(14)
CALL PGENV(x(1),x(size(x)),minval(y),maxval(y),0,1)
CALL PGSCI(2)
select case(st)
case(0)
	call  PGPT (size(x),x,y,'.')
case(1) 
	CALL PGLINE(size(x),x,y)
case default
	CALL PGLINE(size(x),x,y)
end select
end subroutine plot0


subroutine plot1(f,m,mmax,chanel,ctime,str)
integer n_point,PGOPEN,chanel
real f(:),m(:),fmax,mmax
character(len=80) ch
character(len=8)str
character(len=19) ctime
!CALL PGEX0
!IF (PGOPEN('/XWIN') .LE. 0) STOP
n_point=size(f)
fmax=f(n_point)
write(ch,'(i1)')chanel
Call PGASK (.FALSE.)
CALL PGSCI(14)
CALL PGENV(f(1),fmax,0.,mmax,0,1)
!CALL PGSCI(14)
CALL PGBOX('G',fmax/10.,0,'G',mmax/5.,0)

!CALL PGENV(wd(1),wd(2),wd(3),wd(4),0,1)
CALL PGSCI(3)
ch='signal spectrum of chanel '//trim(ch)//' at utc time: '//ctime
CALL PGLAB('Peak frequency ='//trim(str)//'(Hz)', 'amplitude',trim(ch))
!call PGTEXT(0., mmax+20., 'utc time: '//ctime//', freq='//trim(str)//'Hz')
CALL PGSCI(2)
CALL PGLINE(n_point,f,m)
!CALL PGUNSA
end subroutine plot1
endmodule plot_module
