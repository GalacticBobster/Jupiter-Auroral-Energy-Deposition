# This is a Python code to read in the JEDI precipitating energetic proton and electron fluxes and calculate the
# pressure level of energy depositionfrom which one can derive ionization, heating production, emissions, etc.
# The code requires a JEDI input file and uses the H2 loss function of PAdovani et al. A&A 501, 619–631 (2009)
# DOI: 10.1051/0004-6361/200911794.

#Import libraries
import math
import pandas as pd
import numpy as np
import scipy
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import glob
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from pylab import *
from numba import jit, prange

#change to working directory
#os.chdir('/Users/hwaite/Desktop/JUNO 2016 2023/JEDIjupiteraurora')
'''
#NOTE that throughout the code an important relation to keep in mind isthe
# interrelationship of dele and column depth – see Padovani et al. equation 20
# where we see that LF(E) = - dE/dN(H2), so that dE the differential of the energy
# dele is directly related to the differential of the column depth, N. So, if you choose
# dele you are in principle selecting an increment of the column depth.
'''

#Height grid (KINETICS, Moses and Poppe)
'''
# This is the first input file where we need an accurate spline fit function to be able to
# interpolate this height grid at high resolution when we are forced to decrease the
# altitude/pressure step size to select a smaller dele so that the dele used in the energy
# loss calculation does not exceed 1% of the primary electron beam energy. This
decrease
# in dele is required to maintain the full detail of the energy loss cross section as shown
in
# the Padovani et al. paper.
'''

Hgrid = [-5.920E+01, -5.454E+01, -4.991E+01, -4.528E+01, -4.062E+01, -3.691E+01, -3.413E+01, -3.135E+01,
-2.856E+01, -2.576E+01, -2.299E+01, -2.020E+01, -1.741E+01, -1.464E+01, -1.278E+01, -1.091E+01,
-9.065E+00, -7.205E+00, -5.361E+00, -3.500E+00, -1.643E+00, 2.190E-01, 2.076E+00, 3.926E+00,
 5.784E+00, 7.645E+00, 9.498E+00, 1.136E+01, 1.322E+01, 1.507E+01, 1.694E+01, 1.879E+01,
 2.065E+01, 2.252E+01, 2.437E+01, 2.622E+01, 2.809E+01, 2.995E+01, 3.180E+01, 3.367E+01,
 3.552E+01, 3.737E+01, 3.924E+01, 4.110E+01, 4.480E+01, 4.945E+01, 5.502E+01, 6.059E+01,
 6.616E+01, 7.173E+01, 7.824E+01, 8.475E+01, 9.125E+01, 9.775E+01, 1.043E+02, 1.108E+02,
 1.182E+02, 1.257E+02, 1.331E+02, 1.406E+02, 1.481E+02, 1.555E+02, 1.630E+02, 1.705E+02,
 1.780E+02, 1.855E+02, 1.931E+02, 2.007E+02, 2.083E+02, 2.160E+02, 2.236E+02, 2.313E+02,
 2.391E+02, 2.469E+02, 2.547E+02, 2.626E+02, 2.706E+02, 2.786E+02, 2.867E+02, 2.948E+02,
 3.040E+02, 3.132E+02, 3.225E+02, 3.328E+02, 3.442E+02, 3.566E+02, 3.709E+02, 3.883E+02,
 4.086E+02, 4.317E+02, 4.577E+02, 4.875E+02, 5.202E+02, 5.556E+02, 5.930E+02, 6.331E+02,
 6.752E+02, 7.192E+02, 7.650E+02, 8.119E+02, 8.596E+02, 9.084E+02, 9.580E+02, 1.008E+03,
 1.058E+03, 1.109E+03, 1.161E+03, 1.212E+03, 1.264E+03, 1.316E+03, 1.369E+03]

#Pressure grid (KINETICS, Moses and Poppe)
'''
# and once again we need a high-resolution relationship between the height gris and
the
# pressure grid to properly interpolate the column depth associated with a decrease in
the
# column depth of H2 needed to bring the dele into the constrained range 1% of primary
# beam energy. Keep in mind that when we transform a pressure differential between
steps
# into a column density differential we must take into account the temperature changes
# and the composition changes that occur over the range of the pressure differential.
'''

Pgrid = [6.708E+03, 5.976E+03, 5.305E+03, 4.689E+03, 4.122E+03, 3.706E+03, 3.414E+03, 3.137E+03,
 2.876E+03, 2.630E+03, 2.402E+03, 2.185E+03, 1.983E+03, 1.795E+03, 1.675E+03, 1.562E+03,
 1.455E+03, 1.352E+03, 1.255E+03, 1.162E+03, 1.074E+03, 9.901E+02, 9.113E+02, 8.370E+02,
 7.668E+02, 7.004E+02, 6.384E+02, 5.805E+02, 5.270E+02, 4.782E+02, 4.323E+02, 3.906E+02,
 3.515E+02, 3.160E+02, 2.831E+02, 2.531E+02, 2.251E+02, 1.999E+02, 1.770E+02, 1.563E+02,
 1.380E+02, 1.220E+02, 1.079E+02, 9.567E+01, 7.594E+01, 5.772E+01, 4.223E+01, 3.130E+01,
 2.340E+01, 1.763E+01, 1.275E+01, 9.287E+00, 6.818E+00, 5.039E+00, 3.751E+00, 2.803E+00,
 2.018E+00, 1.456E+00, 1.053E+00, 7.609E-01, 5.507E-01, 3.984E-01, 2.888E-01, 2.091E-01,
 1.517E-01, 1.100E-01, 7.977E-02, 5.787E-02, 4.206E-02, 3.056E-02, 2.222E-02, 1.619E-02,
 1.179E-02, 8.605E-03, 6.291E-03, 4.609E-03, 3.385E-03, 2.497E-03, 1.849E-03, 1.378E-03,
 9.982E-04, 7.319E-04, 5.446E-04, 4.006E-04, 2.945E-04, 2.200E-04, 1.653E-04, 1.228E-04,
 9.084E-05, 6.700E-05, 4.939E-05, 3.601E-05, 2.623E-05, 1.907E-05, 1.390E-05, 1.009E-05,
 7.311E-06, 5.285E-06, 3.806E-06, 2.743E-06, 1.978E-06, 1.425E-06, 1.025E-06, 7.390E-07,
 5.313E-07, 3.832E-07, 2.753E-07, 1.981E-07, 1.428E-07, 1.025E-07, 7.373E-08]
 
'''
# All of these atmospheric inputs must be interpolated in accurate high resolution to
make
# this process work properly
'''

H = [4.029E+02, 1.670E+03, 5.329E+03, 1.264E+04, 2.331E+04, 3.589E+04, 4.409E+04, 6.956E+04,
 1.052E+05, 1.216E+05, 1.504E+05, 1.745E+05, 2.078E+05, 2.423E+05, 2.714E+05, 3.019E+05,
 3.356E+05, 3.738E+05, 4.163E+05, 4.633E+05, 5.140E+05, 5.642E+05, 6.155E+05, 6.639E+05,
 7.117E+05, 7.607E+05, 8.324E+05, 9.253E+05, 1.035E+06, 1.164E+06, 1.316E+06, 1.486E+06,
 1.679E+06, 1.913E+06, 2.278E+06, 4.513E+06, 6.421E+06, 8.171E+06, 9.821E+06, 1.137E+07,
 1.274E+07, 1.382E+07, 1.456E+07, 1.484E+07, 1.393E+07, 1.164E+07, 9.512E+06, 7.972E+06,
 6.839E+06, 6.008E+06, 5.279E+06, 4.703E+06, 4.245E+06, 3.884E+06, 3.600E+06, 3.368E+06,
 3.148E+06, 2.973E+06, 2.846E+06, 2.771E+06, 2.749E+06, 2.785E+06, 2.895E+06, 3.111E+06,
 3.471E+06, 4.030E+06, 4.867E+06, 6.070E+06, 7.743E+06, 1.042E+07, 1.693E+07, 4.468E+07,
 1.563E+08, 4.481E+08, 9.344E+08, 1.504E+09, 2.007E+09, 2.335E+09, 2.462E+09, 2.401E+09,
 2.186E+09, 1.894E+09, 1.593E+09, 1.285E+09, 1.004E+09, 7.553E+08, 5.794E+08, 4.532E+08,
 3.593E+08, 2.882E+08, 2.340E+08, 1.904E+08, 1.562E+08, 1.290E+08, 1.073E+08, 8.940E+07,
 7.481E+07, 6.272E+07, 5.262E+07, 4.427E+07, 3.731E+07, 3.147E+07, 2.654E+07, 2.243E+07,
 1.893E+07, 1.602E+07, 1.352E+07, 1.143E+07, 9.665E+06, 8.158E+06, 6.894E+06]

He = [2.203E+19, 2.032E+19, 1.870E+19, 1.716E+19, 1.568E+19, 1.457E+19, 1.376E+19, 1.298E+19,
 1.223E+19, 1.149E+19, 1.078E+19, 1.011E+19, 9.464E+18, 8.832E+18, 8.422E+18, 8.027E+18,
 7.653E+18, 7.281E+18, 6.914E+18, 6.567E+18, 6.219E+18, 5.898E+18, 5.566E+18, 5.256E+18,
 4.957E+18, 4.676E+18, 4.359E+18, 4.036E+18, 3.736E+18, 3.449E+18, 3.178E+18, 2.931E+18,
 2.699E+18, 2.489E+18, 2.291E+18, 2.108E+18, 1.928E+18, 1.756E+18, 1.586E+18, 1.415E+18,
 1.249E+18, 1.091E+18, 9.485E+17, 8.234E+17, 6.218E+17, 4.469E+17, 3.122E+17, 2.232E+17,
 1.621E+17, 1.192E+17, 8.413E+16, 5.985E+16, 4.284E+16, 3.085E+16, 2.245E+16, 1.648E+16,
 1.169E+16, 8.338E+15, 5.969E+15, 4.270E+15, 3.058E+15, 2.188E+15, 1.567E+15, 1.120E+15,
 8.005E+14, 5.717E+14, 4.079E+14, 2.907E+14, 2.070E+14, 1.472E+14, 1.045E+14, 7.415E+13,
 5.249E+13, 3.711E+13, 2.621E+13, 1.848E+13, 1.301E+13, 9.154E+12, 6.426E+12, 4.504E+12,
 3.017E+12, 2.018E+12, 1.348E+12, 8.593E+11, 5.236E+11, 3.019E+11, 1.723E+11, 9.614E+10,
 5.272E+10, 2.855E+10, 1.539E+10, 8.098E+09, 4.259E+09, 2.238E+09, 1.187E+09, 6.252E+08,
 3.300E+08, 1.739E+08, 9.118E+07, 4.802E+07, 2.535E+07, 1.338E+07, 7.041E+06, 3.728E+06,
 1.962E+06, 1.039E+06, 5.460E+05, 2.877E+05, 1.518E+05, 7.950E+04, 4.176E+04]

H2 = [1.397E+20, 1.288E+20, 1.186E+20, 1.088E+20, 9.941E+19, 9.234E+19, 8.725E+19, 8.227E+19,
 7.751E+19, 7.286E+19, 6.837E+19, 6.412E+19, 6.000E+19, 5.599E+19, 5.340E+19, 5.089E+19,
 4.852E+19, 4.616E+19, 4.383E+19, 4.164E+19, 3.943E+19, 3.739E+19, 3.529E+19, 3.332E+19,
 3.143E+19, 2.964E+19, 2.763E+19, 2.559E+19, 2.368E+19, 2.187E+19, 2.015E+19, 1.858E+19,
 1.711E+19, 1.578E+19, 1.453E+19, 1.336E+19, 1.223E+19, 1.114E+19, 1.006E+19, 8.977E+18,
 7.929E+18, 6.927E+18, 6.023E+18, 5.230E+18, 3.951E+18, 2.842E+18, 1.988E+18, 1.423E+18,
 1.035E+18, 7.624E+17, 5.396E+17, 3.850E+17, 2.765E+17, 2.000E+17, 1.462E+17, 1.079E+17,
 7.712E+16, 5.547E+16, 4.010E+16, 2.901E+16, 2.103E+16, 1.524E+16, 1.107E+16, 8.030E+15,
 5.835E+15, 4.240E+15, 3.083E+15, 2.242E+15, 1.631E+15, 1.188E+15, 8.647E+14, 6.301E+14,
 4.592E+14, 3.349E+14, 2.445E+14, 1.785E+14, 1.304E+14, 9.551E+13, 6.994E+13, 5.127E+13,
 3.623E+13, 2.567E+13, 1.824E+13, 1.252E+13, 8.329E+12, 5.332E+12, 3.450E+12, 2.243E+12,
 1.468E+12, 9.690E+11, 6.476E+11, 4.320E+11, 2.916E+11, 1.987E+11, 1.373E+11, 9.522E+10,
 6.651E+10, 4.664E+10, 3.277E+10, 2.316E+10, 1.644E+10, 1.170E+10, 8.324E+09, 5.955E+09,
 4.253E+09, 3.051E+09, 2.182E+09, 1.564E+09, 1.123E+09, 8.037E+08, 5.764E+08]


CH4 = [2.932E+17, 2.704E+17, 2.489E+17, 2.284E+17, 2.087E+17, 1.938E+17, 1.832E+17, 1.727E+17,
 1.627E+17, 1.530E+17, 1.435E+17, 1.346E+17, 1.260E+17, 1.175E+17, 1.121E+17, 1.068E+17,
 1.018E+17, 9.690E+16, 9.201E+16, 8.740E+16, 8.276E+16, 7.849E+16, 7.408E+16, 6.995E+16,
 6.597E+16, 6.222E+16, 5.800E+16, 5.371E+16, 4.971E+16, 4.589E+16, 4.229E+16, 3.900E+16,
 3.592E+16, 3.311E+16, 3.049E+16, 2.803E+16, 2.563E+16, 2.334E+16, 2.107E+16, 1.878E+16,
 1.658E+16, 1.448E+16, 1.258E+16, 1.091E+16, 8.228E+15, 5.904E+15, 4.113E+15, 2.931E+15,
 2.120E+15, 1.552E+15, 1.089E+15, 7.695E+14, 5.461E+14, 3.894E+14, 2.800E+14, 2.028E+14,
 1.413E+14, 9.861E+13, 6.884E+13, 4.787E+13, 3.324E+13, 2.299E+13, 1.588E+13, 1.092E+13,
 7.481E+12, 5.103E+12, 3.464E+12, 2.339E+12, 1.571E+12, 1.049E+12, 6.950E+11, 4.573E+11,
 2.982E+11, 1.928E+11, 1.235E+11, 7.832E+10, 4.907E+10, 3.041E+10, 1.856E+10, 1.120E+10,
 6.270E+09, 3.477E+09, 1.910E+09, 9.719E+08, 4.557E+08, 1.937E+08, 7.547E+07, 2.545E+07,
 7.394E+06, 1.849E+06, 4.064E+05, 7.526E+04, 1.249E+04, 1.882E+03, 2.679E+02, 3.486E+01,
 4.283E+00, 4.960E-01, 5.393E-02, 5.724E-03, 5.928E-04, 5.964E-05, 5.786E-06, 5.626E-07,
 5.259E-08, 4.942E-09, 4.450E-10, 3.996E-11, 3.571E-12, 3.063E-13, 2.635E-14]

Tgrid = [3.000E+02, 2.898E+02, 2.795E+02, 2.692E+02, 2.590E+02, 2.507E+02, 2.444E+02, 2.382E+02,
 2.318E+02, 2.255E+02, 2.194E+02, 2.129E+02, 2.064E+02, 2.002E+02, 1.960E+02, 1.917E+02,
 1.873E+02, 1.829E+02, 1.788E+02, 1.743E+02, 1.701E+02, 1.654E+02, 1.613E+02, 1.569E+02,
 1.524E+02, 1.476E+02, 1.443E+02, 1.417E+02, 1.390E+02, 1.366E+02, 1.340E+02, 1.313E+02,
 1.283E+02, 1.251E+02, 1.217E+02, 1.183E+02, 1.150E+02, 1.121E+02, 1.099E+02, 1.088E+02,
 1.087E+02, 1.100E+02, 1.119E+02, 1.143E+02, 1.201E+02, 1.269E+02, 1.328E+02, 1.375E+02,
 1.414E+02, 1.446E+02, 1.478E+02, 1.510E+02, 1.544E+02, 1.579E+02, 1.609E+02, 1.630E+02,
 1.644E+02, 1.650E+02, 1.653E+02, 1.654E+02, 1.654E+02, 1.654E+02, 1.654E+02, 1.654E+02,
 1.654E+02, 1.654E+02, 1.654E+02, 1.654E+02, 1.656E+02, 1.657E+02, 1.660E+02, 1.664E+02,
 1.668E+02, 1.675E+02, 1.683E+02, 1.694E+02, 1.709E+02, 1.728E+02, 1.754E+02, 1.789E+02,
 1.842E+02, 1.915E+02, 2.014E+02, 2.169E+02, 2.410E+02, 2.828E+02, 3.305E+02, 3.801E+02,
 4.327E+02, 4.865E+02, 5.395E+02, 5.925E+02, 6.419E+02, 6.870E+02, 7.267E+02, 7.618E+02,
 7.916E+02, 8.168E+02, 8.378E+02, 8.548E+02, 8.685E+02, 8.795E+02, 8.882E+02, 8.951E+02,
 9.006E+02, 9.048E+02, 9.082E+02, 9.107E+02, 9.128E+02, 9.144E+02, 9.157E+02]

# Read in atmsopheric data file from Randy Gladstone
asatm_header = ["ALT","TEMP","PRESS","[H]","[H2]","[He]","[CH4]","[C2H2]","[C2H4]","[C2H6]","[CH3C2H]","[C3H8]","[C4H2]","[C4H10]"]
dfatm = pd.read_csv('jupiterAurAtm_cleaned.txt',names=asatm_header,skiprows=10, sep=' ')
dfatm.info()
#print('ATM', dfatm)
#Store data for altitude from body center, H2, He, H, CH4, temperature, pressure (barye) in arrays for later use
jeqrad = 66854.
conkmcm = 1e5
#altgrd = np.array(dfatm['ALT']*conkmcm) #Altitude grid (km)
#temperature = np.array(dfatm['TEMP']) #Temperature (K)
#pressure = np.array(dfatm['PRESS']) #Pressure (barye)
data = genfromtxt('../data/Jupiter_UVS.txt')
H0 = (data[:, 0]) #Km
T0 = (data[:, 1]) #K
P0 = (data[:, 2]) #barye
#H2 = (data[:, 4]) #cc

'''
# I do not understand how this relates to the above atmospheric inputs?
'''

#new height grid
#Hgrid = [-5.920E+01, -5.454E+01, -4.991E+01, -4.528E+01, -4.062E+01, -3.691E+01, -3.413E+01, -3.135E+01,
#-2.856E+01, -2.576E+01, -2.299E+01, -2.020E+01, -1.741E+01, -1.464E+01, -1.278E+01, -1.091E+01,
#-9.065E+00, -7.205E+00, -5.361E+00, -3.500E+00, -1.643E+00, 2.190E-01, 2.076E+00, 3.926E+00,
# 5.784E+00, 7.645E+00, 9.498E+00, 1.136E+01, 1.322E+01, 1.507E+01, 1.694E+01, 1.879E+01,
# 2.065E+01, 2.252E+01, 2.437E+01, 2.622E+01, 2.809E+01, 2.995E+01, 3.180E+01, 3.367E+01,
# 3.552E+01, 3.737E+01, 3.924E+01, 4.110E+01, 4.480E+01, 4.945E+01, 5.502E+01, 6.059E+01,
# 6.616E+01, 7.173E+01, 7.824E+01, 8.475E+01, 9.125E+01, 9.775E+01, 1.043E+02, 1.108E+02,
# 1.182E+02, 1.2E+02, 1.257E+02,1.28E+02, 1.30E+02, 1.331E+02, 1.406E+02, 1.41E+02, 1.42E+02, 1.45E+02, 1.481E+02, 1.555E+02, 1.630E+02, 1.705E+02, 1.75E+02, 1.780E+02, 1.80E+02, 1.855E+02, 1.900E+02, 1.931E+02, 2.007E+02, 
# 2.083E+02, 2.160E+02, 2.236E+02, 2.313E+02, 2.391E+02, 2.469E+02, 2.547E+02, 2.626E+02, 
# 2.706E+02, 2.786E+02, 2.867E+02, 2.948E+02,3.040E+02, 3.132E+02, 3.225E+02, 3.328E+02, 3.442E+02, 3.566E+02, 3.709E+02, 3.883E+02,
# 4.086E+02, 4.317E+02, 4.577E+02, 4.875E+02, 5.202E+02, 5.556E+02, 5.930E+02, 6.331E+02,
# 6.752E+02, 7.192E+02, 7.650E+02, 8.119E+02, 8.596E+02, 9.084E+02, 9.580E+02, 1.008E+03,
# 1.058E+03, 1.109E+03, 1.161E+03, 1.212E+03, 1.264E+03, 1.316E+03, 1.369E+03]

'''
# This is already an interpolation that may be a useful tool for the other interpolation
# that is noted above? Is one grid about the atmosphere and the other about the energy loss?
'''

m4 = genfromtxt('../data/PLANETOCOSMICS.txt')
H4 = m4[:, 0]
Tfunc = CubicSpline(Hgrid, Tgrid, bc_type='not-a-knot')
Pfunc = CubicSpline(Hgrid, Pgrid, bc_type='not-a-knot')
Tgrid = Tfunc(H4)
Pgrid = Pfunc(H4)
Hgrid = H4


altgrd = array(Hgrid)*conkmcm
temperature = array(Tgrid)
pressure = array(Pgrid)*1e3 #mbar to barye
print(len(altgrd), len(pressure))

Total = array(H) + array(H2) + array(CH4) + array(He)
xH2 = array(H2)/Total
xHe = array(He)/Total
#H2func = CubicSpline(pressure, xH2, bc_type='not-a-knot')
#Hefunc = CubicSpline(pressure, xHe, bc_type='not-a-knot')


Hden = np.array(dfatm['[H]'])
H2den = np.array(dfatm['[H2]'])
Heden = np.array(dfatm['[He]'])
CH4den = np.array(dfatm['[CH4]'])
invpres = np.flip(pressure)
invaltgrd = np.flip(altgrd)
#Subroutines for converting altitude to pressure and pressure to altitude, pressure in barye, altiude in cm
def altprs(altin,altgrd,pressure):
    csaltprs = CubicSpline(altgrd, pressure, bc_type='not-a-knot')
    presout = csaltprs(altin)
    return presout
def prsalt(presin,invpres,invaltgrd):
    csprsalt = CubicSpline(invpres, invaltgrd, bc_type='not-a-knot')
    altout = csprsalt(presin)
    return altout
presstrte = altprs(5e7,altgrd,pressure)
presstrtp = presstrte
#check
presin = 6.362e5
altout = prsalt(presin,invpres,invaltgrd)
altin = 910000.
presout = altprs(altin,altgrd,pressure)
print('ALTOUT = ',altout, 'PRESOUT = ',presout)

# Portion of code to read in the JEDI data
#
# Function to determine electron flux from JEDI on PJ7 spectra 7 of Mauk et al. 10.1002/2017GL076901
# begin at 3e4 eV and end at 1e8 eV

def elecpj7spec7(elecener):
    e1 = math.log10(3.0e4)
    e2 = math.log10(3.5e5)
    e3 = math.log10(7.0e5)
    e4 = math.log10(1.0e8)
    slp1 = 0.0
    slp3 = ((math.log10(5.3e7)-math.log10(6e5))/(math.log10(1e4)-math.log10(1e6)))
    slp2 = ((math.log10(1.05e8)-math.log10(2.4e5))/(math.log10(1e5)-math.log10(1e6)))
    val1 = 4.7e6
    logval1 = math.log10(val1)
    val2 = 8.0e5
    logval2 = math.log10(val2)
    logelecen = math.log10(elecener)
    eflx = 0.0
    if (logelecen > e1 and logelecen <= e4):
        if (logelecen < e2):
            eflx = val1
        elif (logelecen >= e2 and logelecen < e3):
            eflx = math.pow(10.,(logval1+((logelecen-e2)*slp2)))
        elif (logelecen >= e3 and logelecen <= e4):
            eflx = math.pow(10.,(logval2+((logelecen-e3)*slp3)))
    return eflx


JEDIelecench = []
JEDIprotench = []
JEDIelecintenpj7s2 = []
JEDIprotintenpj7s2 = []

#############################################
#Reading the JEDI electron spectra
with open('../JEDI_UVSpj7_Bhattacharya.txt') as fJEDIe:
      for jix in range(1):
        next(fJEDIe)
      for line in fJEDIe:
        dataHe = line.split()
        JEDIelecench.append(float(dataHe[0])) #keV
        JEDIelecintenpj7s2.append(float(dataHe[1])) #electrons/(cm^2 s ster keV)
      fJEDIe.close()
#############################################
#Reading the JEDI proton spectra
with open('../JEDI_Hpj7_Bhattacharya.txt') as fJEDIp:
      for jix in range(1):
        next(fJEDIp)
      for line in fJEDIp:
        dataHe = line.split()
        JEDIprotench.append(float(dataHe[0])) #MeV -> eV
        JEDIprotintenpj7s2.append(float(dataHe[1])) #electrons/(cm^2 s ster keV)
      fJEDIp.close()

#Function and supporting materials to determine electron flux from JEDI on PJ7 spectra 7 of Mauk et al. 10.1002/2017GL076901
#First input data from the spreadsheet that Barry Mauk sent me: energy of the JEDI electron channel,
#energy width of the energy channel, and intensity in electrons/(cm^2 s ster keV).
#Linear interpolation of JEDI channels to 10 MeV based on Mauk et al., 2018 (GRL)
#JEDIelecench =[32.07,37.67,44.82,53.9,64.905,78.18,94.045,113.52,136.81,164.77,198.5,238.255,285.27,340.895,407.13,487.005,584.575,705.93,1000.]
JEDIelecintenpj7s1 = [2.28E+06,1.97E+06,1.56E+06,1.54E+06,1.57E+06,1.80E+06,1.73E+06,1.83E+06,1.90E+06,1.93E+06,1.90E+06,1.80E+06,1.66E+06,1.46E+06,1.24E+06,9.97E+05,7.59E+05,5.39E+05,2.43E+05]
#JEDIelecintenpj7s2 = [3.65E+06,3.63E+06,3.46E+06,2.73E+06,3.00E+06,3.22E+06,3.35E+06,4.05E+06,4.52E+06,4.70E+06,4.75E+06,4.82E+06,4.59E+06,3.92E+06,2.96E+06,2.33E+06,1.73E+06,1.11E+06,7.34E+05]
print(len(JEDIelecench), len(JEDIelecintenpj7s1))
func_intpj7s1 = interp1d(log10(JEDIelecench), log10(JEDIelecintenpj7s1), kind='linear',fill_value='extrapolate')
func_intpj7s2 = interp1d(log10(JEDIelecench), log10(JEDIelecintenpj7s2), kind='linear',fill_value='extrapolate')


#UVS interpolation from guess
uvs_tail = [1000, 10000]
uvs_guess1 = [JEDIelecintenpj7s1[-1], 1e3]
uvs_guess2 = [JEDIelecintenpj7s2[-1], 0.31]
uvs_guess2LB = 0.245
uvs_guess2UB = 0.415

#0.31 is right guess for uvs value equal to 50e3
#0.245 is right guess for uvs lower bound
#0.415 is the right guess for uvs upper bound
uvs_guessfunc1 = interp1d(log10(uvs_tail), log10(uvs_guess1))
uvs_guessfunc_lower = interp1d(log10(uvs_tail), log10(uvs_guess1))
uvs_guessfunc_upper = interp1d(log10(uvs_tail), log10(uvs_guess1))
uvs_guessfunc2 = interp1d(log10(uvs_tail), log10(uvs_guess2))

#Linear extrapolation from JEDI
new_channels = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] #Energies (keV)
newint_pj7s1 = 10**uvs_guessfunc1(log10(new_channels))
newint_pj7s2 = 10**uvs_guessfunc2(log10(new_channels))


uvs_integral = 0.5*pi*(newint_pj7s2[-1] + newint_pj7s2[-5])*(new_channels[-1] - new_channels[-5])
uvs_pj7 = 50e3
corr_factor = 0.239 #Zhu et al., 2021
uvs_pj7upper = 50e3/((0.239 - 0.048)*4) #UVS measurements are distributed over 4 pi, however only pi radians are needed
uvs_pj7lower = 50e3/((0.239 + 0.048)*4) #UVS measurements are distributed over 4 pi, however only pi radians are needed
uvs_pj7check = 50e3/((0.239)*4) #UVS measurements are distributed over 4 pi, however only pi radians are needed
print("Guess = ", uvs_integral)
print("Correct = ", uvs_pj7check)
print("Percent difference = ", 100*(uvs_integral - uvs_pj7check)/uvs_pj7check)


plot(new_channels, newint_pj7s2, 'k--')
plot(JEDIelecench, JEDIelecintenpj7s2, 'k-')

for inx in range(len(new_channels)):
  JEDIelecench.append(new_channels[inx])
  JEDIelecintenpj7s1.append(newint_pj7s1[inx])
  JEDIelecintenpj7s2.append(newint_pj7s2[inx])

print("Electron channel (keV)")
print(JEDIelecench)
print("Electron flux (electrons/(cm^2 s ster keV))")
print(JEDIelecintenpj7s2)
'''
#Plot the spectrum with expected values
scatter(JEDIelecench, 0.1*JEDIelecintenpj7s2, c ='r', marker = 'x')
scatter(JEDIelecench, 0.1*JEDIelecintenpj7s2, c ='r', marker = 'x')
plot(JEDIelecench, 0.1*array(JEDIelecintenpj7s2), 'C1-')
plot(JEDIelecench, 0.01*array(JEDIelecintenpj7s2), 'C1--')

vlines(10e3,uvs_guess2LB,uvs_guess2UB, colors='red')
hlines(uvs_guess2LB, 10e3 - 1e3, 10e3 + 1e3, colors = 'red')
hlines(uvs_guess2UB, 10e3 - 1e3, 10e3 + 1e3, colors = 'red')
legend(['JEDI PJ7 (Mauk et al., 2018)', 'UVS PJ7 (Interpolated)', '0.1 x PJ7 Intensity', '0.01 x PJ7 Intensity'], loc = 'best')
xscale('log')
yscale('log')
xlabel('Energy (keV)')
ylabel('Intensity $(cm^{2}.s.sr.keV)^{-1}$')
xlim([32, 12000])
show()
savefig('../figs/JEDI_UVSPJ7.png')
'''

lenJEDI = len(JEDIelecench)
lenJEDIp = len(JEDIprotench)
JEDIelecenchwd = np.zeros(lenJEDI,dtype=float)
JEDIelecenchwd[0] = JEDIelecench[1]-JEDIelecench[0]
JEDIelecenchwd[lenJEDI-1] = JEDIelecench[lenJEDI-1]-JEDIelecench[lenJEDI-2]
for i in range(1,lenJEDI-1):
    JEDIelecenchwd[i] = (JEDIelecench[i+1]-JEDIelecench[i-1])/2.
print(JEDIelecenchwd)

JEDIprotenchwd = np.zeros(lenJEDIp,dtype=float)
JEDIprotenchwd[0] = JEDIprotench[1]-JEDIprotench[0]
JEDIprotenchwd[lenJEDIp-1] = JEDIprotench[lenJEDIp-1]-JEDIprotench[lenJEDIp-2]
for i in range(1,lenJEDIp-1):
    JEDIprotenchwd[i] = (JEDIprotench[i+1]-JEDIprotench[i-1])/2.
print(JEDIprotenchwd)




#Putting JEDI data in flux units per cm^2 per s and energies of channels in eV
JEDIelecflxpj7s2 = np.zeros(lenJEDI,dtype=float)
JEDIenerflxpj7s2 = np.zeros(lenJEDI,dtype=float)
JEDIelecencheV = np.zeros(lenJEDI,dtype=float)
JEDIelecenerflxpj7s2 = np.zeros(lenJEDI,dtype=float)
JEDIenerflxtotpj7s2 = 0.

JEDIprotflxpj7s2 = np.zeros(lenJEDIp,dtype=float)
JEDIenerpflxpj7s2 = np.zeros(lenJEDIp,dtype=float)
JEDIprotencheV = np.zeros(lenJEDIp,dtype=float)
JEDIprotenerflxpj7s2 = np.zeros(lenJEDIp,dtype=float)
JEDIenerpflxtotpj7s2 = 0.

pi = 3.14159

#For electrons
for i in range(lenJEDI):
    JEDIelecflxpj7s2[i] = JEDIelecintenpj7s2[i]*pi*JEDIelecenchwd[i] #conversion to el/cm^2 s
    JEDIelecencheV[i] = 1e3 * JEDIelecench[i]
    JEDIelecenerflxpj7s2[i] = JEDIelecencheV[i] * JEDIelecflxpj7s2[i] #conversion to eV/cm^2 s
    JEDIenerflxtotpj7s2 = JEDIenerflxtotpj7s2 + JEDIelecenerflxpj7s2[i]
JEDIenerflxtotpj7s2 = JEDIenerflxtotpj7s2 / 6.242e11
print('Energy flux for electrons in mW m^-2 for spectra 2 on PJ7 = ',JEDIenerflxtotpj7s2)
print(JEDIelecencheV)

#For protons
for i in range(lenJEDIp):
    JEDIprotflxpj7s2[i] = JEDIprotintenpj7s2[i]*pi*JEDIprotenchwd[i]
    JEDIprotencheV[i] = 1e3 * JEDIprotench[i]
    JEDIprotenerflxpj7s2[i] = JEDIprotencheV[i] * JEDIprotflxpj7s2[i]
    JEDIenerpflxtotpj7s2 = JEDIenerpflxtotpj7s2 + JEDIprotenerflxpj7s2[i]
JEDIenerpflxtotpj7s2 = JEDIenerpflxtotpj7s2 / 6.242e11
print('Energy flux for protons in mW m^-2 for spectra 2 on PJ7 = ',JEDIenerpflxtotpj7s2)
print(JEDIprotencheV)



#Generate energy deposition altitude grid from 0 to 5e7 cm with 1 cm spacing
edaltbot = 0.
edalttop = 10000.*conkmcm
edaltgrd = np.linspace(edaltbot,edalttop,num=2)
#print('edaltgrd = ',edaltgrd)
lenalt = len(edaltgrd)
edepe = np.zeros(lenalt)
edepearr = np.zeros((lenJEDI,lenalt),dtype=float)

# Inititialize output data frame
EDpoutputdf = pd.DataFrame()
EDeoutputdf = pd.DataFrame()

# Portion of code where the hand translated values of the loss functions are entered and stored, then fit with a
# cubic spline for interpolation purposes.

#Truncated cross section value set
data_H2e = genfromtxt('spNIST/H2eSP_Padovani.txt')
LEenrgygrd = data_H2e[:, 0] #eV
LEe = data_H2e[:, 1]*1E-16 #eV/cm^2
print(LEe)
data_H2p = genfromtxt('spNIST/H2eSP_Padovani.txt')
LEp = data_H2p[:, 1]

#Reading the stopping power from NIST database
#Kinetic   Collision Radiative Total     CSDA      Radiation 
KE_He = []
LEe_He = []

KE_CH4 = []
LEe_CH4 = []

KEp_He = []
LEp_He = []

KEp_CH4 = []
LEp_CH4 = []


#NIST He Stopping power
with open('spNIST/NISTe_HeSP.txt') as fHe:
      for jix in range(8):
        next(fHe)
      for line in fHe:
        dataHe = line.split()
        KE_He.append(float(dataHe[0])*1E6) #MeV -> eV
        LEe_He.append(float(dataHe[3])*1E6*4/6.022E23) #MeV cm^2/g -> eV cm^2
        
with open('spNIST/NISTp_HeSP.txt') as fHep:
      for jix in range(8):
        next(fHep)
      for line in fHep:
        dataHe = line.split()
        KEp_He.append(float(dataHe[0])*1E6) #MeV -> eV
        LEp_He.append(float(dataHe[3])*1E6*4/6.022E23) #MeV cm^2/g -> eV cm^2
      

#NIST CH4 Stopping power
with open('spNIST/NISTe_CH4SP.txt') as fCH4:
      for jix in range(8):
        next(fCH4)
      for line in fCH4:
        dataCH4 = line.split()
        KE_CH4.append(float(dataCH4[0])*1E6) #MeV -> eV
        LEe_CH4.append(float(dataCH4[3])*1E6*16/6.022E23) #MeV cm^2/g -> eV cm^2
        
with open('spNIST/NISTp_CH4SP.txt') as fCH4p:
      for jix in range(8):
        next(fCH4p)
      for line in fCH4p:
        dataCH4 = line.split()
        KEp_CH4.append(float(dataCH4[0])*1E6) #MeV -> eV
        LEp_CH4.append(float(dataCH4[3])*1E6*16/6.022E23) #MeV cm^2/g -> eV cm^2


# Functions for interpolation of cross sections
def intrplep(eintrp,LEenrgygrd,LEp):
    lindex = len(LEenrgygrd)
    lnLEenrgygrd = np.zeros(lindex,dtype=float)
    lnLEp = np.zeros(lindex,dtype=float)
    lnLEenrgygrd = np.log10(LEenrgygrd)
    lnLEp = np.log10(LEp)
    cslep = CubicSpline(lnLEenrgygrd, lnLEp, bc_type='natural')
    lneintrp = np.log10(eintrp)
    lnyintrp = cslep(lneintrp)
    yintrp = np.power(10.,lnyintrp)
    return yintrp

def intrplee(eintre,LEenrgygrd,LEe):
    lindex = len(LEenrgygrd)
    lnLEenrgygrd = np.zeros(lindex,dtype=float)
    lnLEe = np.zeros(lindex,dtype=float)
    lnLEenrgygrd = np.log10(LEenrgygrd)
    lnLEe = np.log10(LEe)
    cslee = CubicSpline(lnLEenrgygrd, lnLEe, bc_type='natural')
    lneintre = np.log10(eintre)
    lnyintre = cslee(lneintre)
    yintre = np.power(10.,lnyintre)
    return yintre

'''
#This is the place where you need to check the nergy loss between altitude steps
# and see if the dele exceeds the 1% of primary beam energy and if it does you
# must start subdividing an interpolating until the constraint is met. Then when you
# worl your way in smaller steps to the next predetermined altitude point, you can
# simply sum the energy loss points that have been subdivided and interpolated
'''


#Function to proportion energy depostion over an altitude grid
def edfillgrd(dimensione,presgrdefix,edgrde,JEDIelecenerflx,JEDIestrtenergy,edaltgrd):
    edepesum = np.zeros(lenalt,dtype=float)
    esprdfix = []
    edepesuminv = np.zeros(lenalt)
    prestop0 = presgrdefix[0]#Pressure at top
    #print('prestop0 = ', prestop0)
    alttop0 = prsalt(prestop0,invpres,invaltgrd) #Pressure at top and corresponding altitude
    icntaccum = 0
    for i in range(0,dimensione-1):
        prestop = presgrdefix[i]
        if (prestop <= 0.):
            print('prestop = ', prestop)
            break
        presbot = presgrdefix[i+1]
        if (presbot <= 0.):
            print('presbot = ', presbot)
            break
        alttop = prsalt(prestop,invpres,invaltgrd)
        altbot = prsalt(presbot,invpres,invaltgrd)
        altrng = alttop-altbot #Height grid where energy is getting deposited
        esprd = (edgrde[i]*JEDIelecenerflx)/(altrng*JEDIestrtenergy) #Energy deposited at specific height
        esprdfix.append(esprd)

    print("Esprdfix")
    print(len(esprdfix))
    return esprdfix
    

#Run Energy Deposition Case
rngidx = 100
energy = np.zeros(rngidx,dtype=float)
valuep = np.zeros(rngidx,dtype=float)
valuee = np.zeros(rngidx,dtype=float)
# Define the energy grid for this check
estrt = np.log10(LEenrgygrd[0])
estop = np.log10(LEenrgygrd[9])
energy = np.logspace(estrt, estop, num=rngidx, endpoint=True, base=10.0, dtype=None, axis=0)

for k in range (rngidx):
    enp = energy[k]
    ene = energy[k]
    valp = intrplep(enp,LEenrgygrd,LEp)
    valuep[k] = valp
#    print('valp = ',valp)
    vale = intrplee(ene,LEenrgygrd,LEe)
    valuee[k] = vale
#    print('vale = ',vale)

# Jupiter values in cgs to calculate column density
g=2479.
m = 2.3151*1.6604e-24 #Mean molecular mass updated from Gupta et al., (2022)
# Conversion bar to barye (cgs)
conversion = 1.e6

# functions to convert from pressure to column density and back
def presnH(pressurebarye):
    nH = pressurebarye/(m*g)
    return nH
def nHpres(nH):
    pressurebarye = nH*m*g
    return pressurebarye

'''
# This is useful logic for checking the size of dele in the interpolation scheme
'''

#determine pressure/column depth step size to use in energy degradation scheme

pelp = 0.0001
pele = 0.0001 #0.01-0.001 (accuracy of code)

# Determine energy loss grid
def dimp(JEDIpstrtenergy,pelp):
# Initial energy increment
    el = 10##pelp*JEDIpstrtenergy
    ie = 0
    energy = JEDIpstrtenergy
    for i in range(100000):
        ie = ie + 1
        energy = energy - el
        if (energy < 0.1):
            return ie
def dime(JEDIestrtenergy,pele):
# Initial energy increment
    el = pele*JEDIestrtenergy
    ie = 0
    energy = JEDIestrtenergy
    ix = 0
    while(True):
#Set a piecewise energy loss process for the electrons
        if(logical_and(energy >= 1e5, ix == 0)):
            el = 1*el
            ix = ix + 1
        elif(logical_or(el > 1e3, energy < 1e4)):
            #print("special condition")
            el = 100
        energy = energy - el
        if (energy < 0.1):
            break
        ie = ie + 1
        #print("Electron energy | Starting energy")
        #print(energy, JEDIestrtenergy)
    print("dimensione")
    print(ie)
    return ie


# Portion of code containing the energy degradation functions
def pendegrade(JEDIpstrtenergy,dimensionp,edgrdp):
    #edgrdp = pelp*JEDIpstrtenergy
    presgrdp[0] = presstrtp
    XSC = (intrplee(JEDIpstrtenergy,LEenrgygrd,LEp)*0.89) + (intrplee(JEDIpstrtenergy,KEp_He,LEp_He)*0.11)
    nHp[0] = edgrdp/XSC
    JEDIped[0] = JEDIpstrtenergy - edgrdp
    for j in range(dimensionp-1):
        #edgrdp[j+1] = pelp*JEDIped[0]
        JEDIped[j+1] = JEDIped[j] - edgrdp
        XSC = (intrplee(JEDIpstrtenergy,LEenrgygrd,LEp)*0.89) + (intrplee(JEDIpstrtenergy,KEp_He,LEp_He)*0.11)
        nHtmp = edgrdp/XSC
        nHp[j+1] = nHp[j]+nHtmp
        presgrdp[j+1] = nHpres(nHp[j+1])
        if (JEDIped[j+1] < 0.1):
            print('I am here p')
            break
    return presgrdp, JEDIped


# Previously the electron degradation function is used to calculate the hydrogen column depth corresponding to electrons of various energies
# The code is getting updated to include the contributions from He, partitioning the contributions from H2 and He
# For now I am considering the contribution to be 89 and 11 percent for H2 and He
'''
#check this in Julie’s model atmosphere and get ti correct at every altitude/energy loss
step
'''

def eendegrade(JEDIestrtenergy,dimensione, edgrde):
    presgrde[0] = presstrte
    XSC = (intrplee(JEDIestrtenergy,LEenrgygrd,LEe))
    JEDIeed[0] = JEDIestrtenergy #- edgrde
    ix = 0
    ELoss[0] = pele*JEDIestrtenergy
    for j in range(dimensione - 1):
        ELoss[j+1] = ELoss[j]
        if(logical_and(JEDIeed[j] >= 1e5, ix == 0)):
            ELoss[j+1] = 1*ELoss[j+1]
            ix = ix + 1
        if(logical_or(ELoss[j+1] > 1e3, JEDIeed[j] < 1e4)):
            ELoss[j+1] = 100
        JEDIeed[j+1] = JEDIeed[j] - ELoss[j+1]
        #print("Electron energy | Starting energy")
        #print(JEDIeed[j+1], JEDIestrtenergy)
        #print(j+1)
        if (JEDIeed[j+1] < 0.1):
            print('I am here e')
            break
        XSC = (intrplee(JEDIestrtenergy,LEenrgygrd,LEe))
        nHtmp = edgrde/XSC
        nHe[j+1] = nHe[j]+nHtmp
        presgrde[j+1] = nHpres(nHe[j+1])
    return presgrde, JEDIeed, ELoss


# Portion of code that iterates over JEDI energy spectrum for electrons for now
Egrid = zeros(len(altgrd))
Epgrid = zeros(len(altgrd))

#For electrons
for i in range(len(JEDIelecencheV)):
    JEDIestrtenergy = JEDIelecencheV[i]
    edgrde = pele*JEDIestrtenergy
    dimensione = dime(JEDIestrtenergy,pele)
    JEDIestrtenergy = JEDIelecencheV[i]
    JEDIeed = np.zeros((dimensione),dtype=float)
    ELoss = np.zeros((dimensione),dtype=float)
    nHe = np.zeros(dimensione,dtype=float)
    presgrde = np.zeros(dimensione,dtype=float)
    presgrde, JEDIed, ELoss = eendegrade(JEDIestrtenergy,dimensione, edgrde)
    edepe = edfillgrd(dimensione,presgrde,ELoss,JEDIelecenerflxpj7s2[i],JEDIestrtenergy,edaltgrd)
    Height = prsalt(presgrde[0:dimensione-1],invpres,invaltgrd)
    print(len(edepe))
    print(len(Height))
    #total energy
    h_min = min(Height)/1e5
    h_max = max(Height)/1e5
    print('Height (km)')
    print ('JEDIestrtenergy = ',JEDIestrtenergy)
    print(dimensione)
    print(h_min, h_max)
    mask = where((Hgrid >= h_min) & (Hgrid <= h_max))
    #print(mask[0])
    for inx in range(len(mask[0])):
      masked_hgrid = Hgrid[mask[0][inx]]
      energfunc = interp1d(Height/1e5, edepe, fill_value = [0])
      Egrid[mask[0][inx]] = Egrid[mask[0][inx]] + energfunc(masked_hgrid)
'''
#For protons      
for i in range(len(JEDIprotencheV)):
    JEDIpstrtenergy = JEDIprotencheV[i]
    edgrdp = pelp*JEDIpstrtenergy
    print ('JEDIpstrtenergy = ',JEDIpstrtenergy)
    dimensionp = dime(JEDIpstrtenergy,pelp)
    JEDIpstrtenergy = JEDIprotencheV[i]
    JEDIped = np.zeros((dimensionp),dtype=float)
    nHp = np.zeros(dimensione,dtype=float)
    presgrdp = np.zeros(dimensione,dtype=float)
    presgrdp, JEDIed = pendegrade(JEDIpstrtenergy,dimensionp,edgrdp)
    edepp = edfillgrd(dimensionp,presgrdp,edgrdp,JEDIprotenerflxpj7s2[i],JEDIpstrtenergy,edaltgrd)
    Height = prsalt(presgrdp[1:dimensionp-1],invpres,invaltgrd)
    #total energy
    h_min = min(Height)/1e5
    h_max = max(Height)/1e5
    mask = where((Hgrid >= h_min) & (Hgrid <= h_max))
    print(mask[0])
    for inx in range(len(mask[0])):
      masked_hgrid = Hgrid[mask[0][inx]]
      energfunc = interp1d(Height/1e5, edepp, fill_value = [0])
      Epgrid[mask[0][inx]] = Epgrid[mask[0][inx]] + energfunc(masked_hgrid)

'''

#edgrdp = pelp*JEDIpstrtenergy

#Plot electron energy deposition
'''
Totenerg = Egrid + Epgrid

plot(Totenerg, Hgrid, 'k-')
xscale('log')
ylim([min(Hgrid), max(Hgrid)])
xlabel('Energy deposition (eV/cm$^{3}$s)')
ylabel('Altitude (km)')
savefig('Energdep_altgr.png')
show()

if(JEDIestrtenergy == 1E6):
        plot(np.cumsum(edepe), Height/1e5, 'k-')
      if(JEDIestrtenergy == 10E6):
        plot(np.cumsum(edepe), Height/1e5, 'C7-')


xscale('log')
xlabel('Energy deposition (eV/cm^3 s)')
ylabel('Altitude (km)')
legend(['1 MeV electrons','10 MeV electrons'], loc = 'best')
xlim([1e12, 1e20])

for i in range(len(JEDIelecencheV)):
    JEDIestrtenergy = JEDIelecencheV[i]
    edgrde = pele*JEDIestrtenergy
    print ('JEDIestrtenergy = ',JEDIestrtenergy)
    dimensione = dime(JEDIestrtenergy,pele)
    JEDIestrtenergy = JEDIelecencheV[i]
    JEDIeed = np.zeros((dimensione),dtype=float)
    nHe = np.zeros(dimensione,dtype=float)
    presgrde = np.zeros(dimensione,dtype=float)
    presgrde, JEDIed = eendegrade(JEDIestrtenergy,dimensione,edgrde)
    edepe = edfillgrd(dimensione,presgrde,edgrde,JEDIelecenerflxpj7s2[i],JEDIestrtenergy,edaltgrd)
    Height = prsalt(presgrde[1:dimensione-1],invpres,invaltgrd)
    if(JEDIestrtenergy == 1E6):
      ax2.plot(np.cumsum(edepe), (presgrde[1:dimensione-1])/1e6, 'w-', alpha = 0)
    if(JEDIestrtenergy == 10E6):
      ax2.plot(np.cumsum(edepe), (presgrde[1:dimensione-1])/1e6, 'w-', alpha = 0)

ax2.set_ylabel('Pressure (bar)')
ax2.set_yscale('log')
ax2.invert_yaxis()
ax2.set_xlim([1e12, 1e20])
'''
##postprocessing to compute heating rate and ionization
##The fractionation efficiencies are taken from Waite et al., (1983)
w = 39.38 #mean energy loss per ion pair (eV/ip)
n_heat = 11.06 #efficiency of neutral heating rate
e_heat = 1.65 #efficiency of electron heating rate
h2p_heat = 38.91 #H2+ production efficiency
vd_heat = 8.31 #vibration direct
vc_heat = 3.32 #vibration cascade
hp_heat = 6.09 #H+ production efficiency


#Total H2+ production rate
H2p_rate = Egrid*h2p_heat/(w*100)
#Total H+ production rate
Hp_rate = Egrid*hp_heat/(w*100)
#Total heating rate
Heat_rate = Egrid*(n_heat)/100
e_rate = Egrid*e_heat/100
#Electron impact vibrationally excited states
v1 = 11.37 - 10.0
v2 = 12.4 - 10.0
v3 = 14.12 - 10.0

Ev1 = 1*Egrid*vd_heat/(v1*100)
Ev2 = 0.1*Egrid*vd_heat/(v2*100)
Ev3 = 0.007*Egrid*vd_heat/(v3*100)


dat2 = genfromtxt('Egert_heating_rate.txt')
he2 = dat2[:,0]
alt2 = dat2[:,1]
'''
plot(Heat_rate, Hgrid, 'k-' )
plot(e_rate, Hgrid, 'k--')
plot(he2, alt2, 'C7--')
legend(['Neutral heating rate', 'Electron heating rate', 'Egert et al., (2017)'], loc = 'best')
#plot(Hp_rate,Hgrid,'C1-')
xscale('log')
ylim([max(Pgrid), min(Pgrid)])
ylabel('Altitude (km)')
xlabel('Heating rate (eV/cm^3.s)')
#xlabel('Reaction Rate ($cm^{-3}$.$s^{-1}$)')
ylim([0,500])
xlim([10,1e10])
legend(['Neutral heating rate', 'Electron heating rate'], loc = 'best')
#legend(['$H_{2}$ + $e_{p}$ -> $H_{2}^{+}$ + e + $e_{p}$', '$H_{2}$ + $e_{p}$ -> $H^{+}$ + H + e + $e_{p}$'], loc = 'best')
#savefig('Heating_rate.png')
show()
'''
for i in range(len(Hgrid)-1, -1, -1):
  print(str(Hgrid[i]) + "," + str(Heat_rate[i]) + "," + str(e_rate[i]) )
# print(str(H2p_rate[i]/array(H2)[i]) + " " + str(Hp_rate[i]/array(H2)[i]) +  " " + str(Ev1[i]/array(H2)[i]) + " " +  str(Ev2[i]/array(H2)[i]) + " " +  str(Ev3[i]/array(H2)[i]))
#for i in range(len(H0)-1, -1, -1):
#  print(str(H2p_rate[i]/array(H2)[i]) + "," + str(Hp_rate[i]/array(H2)[i]) +  "," + str(Ev1[i]/array(H2)[i]) + "," +  str(Ev2[i]/array(H2)[i]) + "," +  str(Ev3[i]/array(H2)[i]))

'''
#EOF?
'''
