subroutine read_driv
use param
!use utilmod
!use module11 
!read the driv file
! VARIABLE  I/O  DESCRIPTION
! --------  ---  --------------------------------------------------
! N         O    NUMBER OF STATE OF SPACECRAFT
    
IMPLICIT NONE
integer i
!CHARACTER*100 FRAME, ABCORR,KER,out_file, ntarget, obsrvr, utc0,utc1
!character*50 temp
OPEN(driv_id,FILE='../input/driv',STATUS='unknown')
    do i=1,driv_length
        READ(driv_id,*)ctemp
        if (ctemp(1:11).eq.'$DOPPLERL0') goto 991
    enddo
991 	READ(driv_id,*)rsr_data_PAR%mode
	READ(driv_id,*)rsr_data_PAR%DATAFILE
     READ(driv_id,*)rsr_data_PAR%chanel
     READ(driv_id,*)rsr_data_PAR%datatype
     READ(driv_id,*)rsr_data_PAR%arb0time
     READ(driv_id,*)rsr_data_PAR%arb1time
     READ(driv_id,*)ctemp(1:2)
    	READ(driv_id,*)rsr_data_PAR%SPAN1 
	READ(driv_id,*)rsr_data_PAR%amp_win
	READ(driv_id,*)ctemp(1:2)
     READ(driv_id,*)rsr_data_PAR%SPAN2
	READ(driv_id,*)rsr_data_PAR%itermax
	READ(driv_id,*)rsr_data_PAR%xcmin
	READ(driv_id,*)rsr_data_PAR%xcmax
    	rewind(driv_id)
    	write(*,*)rsr_data_PAR
	
!CALL FURNSH ( global_par%ker )
!write(*,*)global_par%ker
!CALL STR2ET  ( utc0, ET0 )
!CALL STR2ET  ( utc1, ET1 )
!N=floor((et1-et0)/step)
!close(driv_id0)
close(driv_id)
!CALL UNLOAD ( KER )
end subroutine read_driv

