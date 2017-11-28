module ftn

interface

subroutine ftn_set_length(length) bind(C, name = "ftn_set_length")
use iso_c_binding
integer(kind=c_long_long), intent(in) :: length
end subroutine ftn_set_length

subroutine ftn_set_tt(tt) bind(C, name = "ftn_set_tt")
use iso_c_binding
real(kind=c_double) :: tt
end subroutine ftn_set_tt

subroutine ftn_set_s0(s0) bind(C, name = "ftn_set_s0")
use iso_c_binding
integer(kind=c_int) :: s0
end subroutine ftn_set_s0

subroutine ftn_eval(x, result) bind(C, name = "ftn_eval")
use iso_c_binding
real(kind=c_double), intent(in) :: x
real(kind=c_double), intent(out) :: result
end subroutine ftn_eval

end interface

end module ftn

