module math_lib
use lib_array
DOUBLE PRECISION,parameter::pi=dacos(-1.d0)
real*8::cdiff15(15)
data cdiff15/-0.4162538721097508E-04 ,  0.6798801907577090E-03, &
-0.5303056113282120E-02,   0.2651523747302387E-01 , -0.9722241356664973E-01, &
0.2916669378946157E+00 , -0.8750002078492521E+00  , 0.7953637748414621E-12, &
0.8750002078478781E+00 , -0.2916669378937363E+00  , 0.9722241356623978E-01, &
-0.2651523747288770E-01,   0.5303056113250826E-02 , -0.6798801907531016E-03, &
0.4162538721075304E-04/
private
public pi,t_polynomial_coefficients
contains

subroutine t_polynomial_coefficients ( n, c )

!*****************************************************************************80
!
!! T_POLYNOMIAL_COEFFICIENTS: coefficients of the Chebyshev polynomial T(n,x).
!
!  First terms:
!
!    N/K     0     1      2      3       4     5      6    7      8    9   10
!
!     0      1
!     1      0     1
!     2     -1     0      2
!     3      0    -3      0      4
!     4      1     0     -8      0       8
!     5      0     5      0    -20       0    16
!     6     -1     0     18      0     -48     0     32
!     7      0    -7      0     56       0  -112      0    64
!
!  Recursion:
!
!    T(0,X) = 1,
!    T(1,X) = X,
!    T(N,X) = 2 * X * T(N-1,X) - T(N-2,X)
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    11 May 2003
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the highest order polynomial to compute.
!    Note that polynomials 0 through N will be computed.
!
!    Output, real ( kind = 8 ) C(0:N,0:N), the coefficients of the Chebyshev T
!    polynomials.
!
  implicit none

  integer ( kind = 4 ) n

  real ( kind = 8 ) c(0:n,0:n)
  integer ( kind = 4 ) i

  if ( n < 0 ) then
    return
  end if

  c(0:n,0:n) = 0.0D+00
  c(0,0) = 1.0D+00

  if ( n == 0 ) then
    return
  end if

  c(1,1) = 1.0D+00

  do i = 2, n
    c(i,0)     =                        - c(i-2,0)
    c(i,1:i-2) = 2.0D+00 * c(i-1,0:i-3) - c(i-2,1:i-2)
    c(i,  i-1) = 2.0D+00 * c(i-1,  i-2)
    c(i,  i  ) = 2.0D+00 * c(i-1,  i-1)
  end do

  return
end subroutine t_polynomial_coefficients

end module math_lib
