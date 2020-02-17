import numpy as np
import scipy.integrate as integrate
import scipy.constants as cnst
from scipy.interpolate import UnivariateSpline

# a general function for approximating $\int_0^b f(x) dx$ simultaneously for
# multiple values of right end points $b$.
def integrate_spline_approx(f,right_endpts,npts=100):
    x_min = min(0,*right_endpts)
    x_max = max(right_endpts)
    pad = (x_max-x_min)/10
    x_min-=pad
    x_max+=pad
    
    x_space = np.linspace(x_min,x_max,npts)

    # Get a spline interpolation of f over x_space
    # with no smoothing (s=0). This is probably the
    # speed bottleneck of this function.
    spl = UnivariateSpline(x_space,f(x_space),s=0)

    # the spline can be integrated analytically (fast)
    return np.array([spl.integral(0,right_endpt) for right_endpt in right_endpts])
    

# the integrand in the calculation of mu from z,cosmology
def integrand(zba, omegam, omegade, w):
    return 1.0/np.sqrt(
        omegam*(1+zba)**3 + omegade*(1+zba)**(3.+3.*w) + (1.-omegam-omegade)*(1.+zba)**2
    )

# integration of the integrand given above
def hubble(z,omegam, omegade,w):
    # if an iterable of z's was given then use the spline
    # method for calculating the integral
    if hasattr(z,'__len__'):
        I = integrate_spline_approx(lambda zba: integrand(zba,omegam,omegade,w), z)

    # otherwise, integrate numerically in the usual way
    else:
        I = integrate.quad(integrand,0., z,args = (omegam,omegade,w))[0]
    return I

# Dlz
# N.B. z is z_CMB
def Dlz(omegam, omegade, h, z, omega, z_helio):
    hubbleint = hubble(z,omegam,omegade,omega)
    omegakmag =  np.sqrt(np.abs(1-omegam-omegade))
    if (omegam+omegade)>1:
        distance = (cnst.c*1e-5 *(1+z_helio)/(h*omegakmag)) *np.sin(hubbleint*omegakmag)
    elif (omegam+omegade)==1:
        distance = cnst.c*1e-5 *(1+z_helio)* hubbleint/h
    else:
        distance = cnst.c*1e-5 *(1+z_helio)/(h*omegakmag) *np.sinh(hubbleint*omegakmag)
    return distance

# muz: distance modulus as function of params, redshift
def muz(cosmo_param, z, z_helio):
    omegam = cosmo_param[0]
    omegade = cosmo_param[1]
    w = -1.0
    #w = cosmo_param[1]
    #h = cosmo_param[2]
    h = 0.72
    return (5.0 * np.log10(Dlz(omegam, omegade, h, z, w, z_helio))+25)
