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
import glob
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from pylab import *

#change to working directory
#os.chdir('/Users/hwaite/Desktop/JUNO 2016 2023/JEDIjupiteraurora')

# Read in atmsopheric data file from Randy Gladstone
asatm_header = ["ALT","TEMP","PRESS","[H]","[H2]","[He]","[CH4]","[C2H2]","[C2H4]","[C2H6]","[CH3C2H]","[C3H8]","[C4H2]","[C4H10]"]
dfatm = pd.read_csv('jupiterAurAtm_cleaned.txt',names=asatm_header,skiprows=10, sep=' ')
dfatm.info()
#print('ATM', dfatm)
#Store data for altitude from body center, H2, He, H, CH4, temperature, pressure (barye) in arrays for later use
jeqrad = 66854.
conkmcm = 1e5
altgrd = np.array(dfatm['ALT']*conkmcm) #Altitude grid (km)
temperature = np.array(dfatm['TEMP']) #Temperature (K)
pressure = np.array(dfatm['PRESS']) #Pressure (barye)
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

#Function and supporting materials to determine electron flux from JEDI on PJ7 spectra 7 of Mauk et al. 10.1002/2017GL076901
#First input data from the spreadsheet that Barry Mauk sent me: energy of the JEDI electron channel,
#energy width of the energy channel, and intensity in electrons/(cm^2 s ster keV).
JEDIelecench =[32.07,37.67,44.82,53.9,64.905,78.18,94.045,113.52,136.81,164.77,198.5,238.255,285.27,340.895,407.13,487.005,584.575,705.93,1000.]
JEDIelecintenpj7s1 = [2.28E+06,1.97E+06,1.56E+06,1.54E+06,1.57E+06,1.80E+06,1.73E+06,1.83E+06,1.90E+06,1.93E+06,1.90E+06,1.80E+06,1.66E+06,1.46E+06,1.24E+06,9.97E+05,7.59E+05,5.39E+05,2.43E+05]
JEDIelecintenpj7s2 = [3.65E+06,3.63E+06,3.46E+06,2.73E+06,3.00E+06,3.22E+06,3.35E+06,4.05E+06,4.52E+06,4.70E+06,4.75E+06,4.82E+06,4.59E+06,3.92E+06,2.96E+06,2.33E+06,1.73E+06,1.11E+06,7.34E+05]
JEDIelecenchwd = np.zeros(19,dtype=float)
JEDIelecenchwd[0] = JEDIelecench[1]-JEDIelecench[0]
JEDIelecenchwd[18] = JEDIelecench[18]-JEDIelecench[17]
for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
    JEDIelecenchwd[i] = (JEDIelecench[i+1]-JEDIelecench[i-1])/2.
print(JEDIelecenchwd)
#Putting JEDI data in flux units per cm^2 per s and energies of channels in eV
JEDIelecflxpj7s2 = np.zeros(19,dtype=float)
JEDIenerflxpj7s2 = np.zeros(19,dtype=float)
JEDIelecencheV = np.zeros(19,dtype=float)
JEDIelecenerflxpj7s2 = np.zeros(19,dtype=float)
JEDIenerflxtotpj7s2 = 0.
pi = 3.14159
lenJEDI = len(JEDIelecench)
for i in range(lenJEDI):
    JEDIelecflxpj7s2[i] = JEDIelecintenpj7s2[i]*pi*JEDIelecenchwd[i]
    JEDIelecencheV[i] = 1e3 * JEDIelecench[i]
    JEDIelecenerflxpj7s2[i] = JEDIelecencheV[i] * JEDIelecflxpj7s2[i]
    JEDIenerflxtotpj7s2 = JEDIenerflxtotpj7s2 + JEDIelecenerflxpj7s2[i]
JEDIenerflxtotpj7s2 = JEDIenerflxtotpj7s2 / 6.242e11
print('Energy flux in mW m^-2 for spectra 2 on PJ7 = ',JEDIenerflxtotpj7s2)
print(JEDIelecencheV)
#Generate energy deposition altitude grid from 0 to 5e7 cm with 1 cm spacing
edaltbot = 0.
edalttop = 500.*conkmcm
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

#Full value set
#LEenrgygrd = [1.00E-01,5.00E-01,1.00E+00,3.60E+00,9.90E+00,1.01E+01,1.00E+02,1.00E+03,1.00E+04,1.00E+05,1.00E+06,1.00E+07,1.00E+08,1.00E+09,1.00E+10,1.00E+11]
#LEp = [1.10E-15,2.80E-15,3.00E-15,5.10E-15,7.60E-15,7.90E-15,5.00E-16,2.40E-15,7.00E-15,1.00E-14,2.40E-15,3.70E-16,6.00E-17,2.00E-17,1.00E-16,1.80E-15]
#LEe = [2.80E-19,1.05E-18,9.20E-18,5.00E-17,2.00E-17,1.00E-15,4.00E-15,9.50E-16,1.20E-16,2.70E-17,1.40E-17,1.20E-17,2.20E-17,7.40E-17,6.50E-16,7.00E-15]
#Truncated cross section value set
LEenrgygrd = [1.00E+02,1.00E+03,1.00E+04,1.00E+05,1.00E+06,1.00E+07,1.00E+08,1.00E+09,1.00E+10,1.00E+11]
LEp = [5.00E-16,2.40E-15,7.00E-15,1.00E-14,2.40E-15,3.70E-16,6.00E-17,2.00E-17,1.00E-16,1.80E-15]
LEe = [4.00E-15,9.50E-16,1.20E-16,2.70E-17,1.40E-17,1.20E-17,2.20E-17,7.40E-17,6.50E-16,7.00E-15]

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

#Functions for finding value closest to set of list values
#def closest_value_lesser(input_list, input_value):
#    arr = np.asarray(input_list)
#    i = (np.abs(arr - input_value)).argmin()
#    lstindxval = (i, arr[i])
#    return lstindxval
#def closest_value_greater(input_list, input_value):
#    arr = np.flip(input_list)
#    i = (np.abs(arr - input_value)).argmin()
#    lstindxval = (i, arr[i])
#    return lstindxval


#Function to proportion energy depostion over an altitude grid
def edfillgrd(dimensione,presgrdefix,edgrde,JEDIelecenerflx,JEDIestrtenergy,edaltgrd):
    edepesum = np.zeros(lenalt,dtype=float)
    esprdfix = []
    edepesuminv = np.zeros(lenalt)
    prestop0 = presgrdefix[0]#Pressure at top
    #print('prestop0 = ', prestop0)
    alttop0 = prsalt(prestop0,invpres,invaltgrd)#Pressure at top and corresponding altitude
    icntaccum = 0
    for i in range(1,dimensione-1):
        prestop = presgrdefix[i]
        if (prestop <= 0.):
            #print('prestop = ', prestop)
            break
        presbot = presgrdefix[i+1]
        if (presbot <= 0.):
            #print('presbot = ', presbot)
            break
        alttop = prsalt(prestop,invpres,invaltgrd)
        altbot = prsalt(presbot,invpres,invaltgrd)
        altrng = alttop-altbot #Height grid where energy is getting deposited
        #print('alttop =', alttop)
        #print('altbot =', altbot)
        esprd = (edgrde*JEDIelecenerflx)/altrng #Energy deposited at specific height
        esprdfix.append(esprd)
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
#Plot values
figtst = px.scatter(x=energy, y=valuep,log_x=True,range_x=(1e2,1e11),log_y=True,range_y=(1e-17,1e-13),title='Energy Deposition',labels={"y":  "EnergyLoss CS", "x": "Energy (eV)"})
figtst.add_scatter(x=energy, y=valuep, mode='markers',name='protons')
figtst.add_scatter(x=energy, y=valuee, mode='markers',name='electrons')
figtst.update_yaxes(exponentformat="E")
figtst.show()

# Jupiter values in cgs to calculate column density
g=2479.
m = 2.*1.6604e-24
# Conversion bar to barye (cgs)
conversion = 1.e6

# functions to convert from pressure to column density and back
def presnH(pressurebarye):
    nH = pressurebarye/(m*g)
    return nH
def nHpres(nH):
    pressurebarye = nH*m*g
    return pressurebarye

#determine pressure/column depth step size to use in energy degradation scheme

pelp = 0.1
pele = 0.001 #0.01-0.001 (accuracy of code)

# Determine energy loss grid
def dimp(JEDIpstrtenergy,pelp):
# Initial energy increment
    el = pelp*JEDIpstrtenergy
    ie = 0
    energy = JEDIpstrtenergy
    for i in range(1000):
        ie = ie + 1
        energy = energy - el
        if (energy < 1e2):
            return ie
def dime(JEDIestrtenergy,pele):
# Initial energy increment
    el = pele*JEDIestrtenergy
    ie = 0
    energy = JEDIestrtenergy
    for i in range(1000):
        ie = ie + 1
        energy = energy - el
        if (energy < 1e2):
            return ie

# Portion of code containing the energy degradation functions
def pendegrade(JEDIpstrtenergy,dimensionp,pelp):
    edgrdp = pelp*JEDIpstrtenergy
    predgrdp[0] = presstrtp
    nHp[0] = edgrdp/interplep(JEDIpstrtenergy,LEpnrgygrd,LEp)
    JEDIped[0] = JEDIpstrtrenergy - edgrdp
    for j in range(dimensionp-1):
        edgrdp[j+1] = pelp*JEDIped[0]
        JEDIped[j+1] = JEDIped[j] - edgrdp[j+1]
        nHtmp = edgrdp[j]/intrplep(JEDIped[j],LEenrgygrd,LEp)
        nHp[j+1] = nHp[j]+nHtmp
        presgrdp[j+1] = nHpres(nHp[j+1])
#        print (edgrdp[j+1],nHp[j+1],presgrdp[j+1],JEDIped[j+1])
        if (JEDIped[j+1] <1e2):
            print('I am here p')
            EDpoutputdf = pd.DataFrame(data={'Pressurebar':presgrdp,'Column depth':nHp,'EnergyDeposition':edgrdp,'ProtonEnergy':JEDIped})
            break
    return EDpoutputdf
def eendegrade(JEDIestrtenergy,dimensione,edgrde):
    presgrde[0] = presstrte
    nHe[0] = edgrde/intrplee(JEDIestrtenergy,LEenrgygrd,LEe)
    JEDIeed[0] = JEDIestrtenergy - edgrde
    for j in range(dimensione):
        JEDIeed[j+1] = JEDIeed[j] - edgrde
        nHtmp = edgrde/intrplee(JEDIeed[j],LEenrgygrd,LEe)
#       print('nHtmp =',nHtmp,'JEDIeed[j+1] = ',JEDIeed[j+1])
        nHe[j+1] = nHe[j]+nHtmp
        presgrde[j+1] = nHpres(nHe[j+1])
#       print('edgrde[j+1] = ',edgrde[j+1],'nHe[j+1] = ',nHe[j+1],'presgrde[j+1] = ',presgrde[j+1])
        if (JEDIeed[j+1] <=1e2):
            print('I am here e')
            break
    print('presgrde =',presgrde,'edgrde = ',edgrde)
    return presgrde, JEDIeed





#Produce figures
#Electrons

# Portion of code that iterates over JEDI energy spectrum for electrons for now
edepeout = np.zeros_like(edaltgrd)

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
    plot(edepe, Height/1e5)
xscale('log')
xlabel('Energy deposition per height (electrons/cm^3)')
ylabel('Altitude (km)')
'''
    print('presgrde = ',presgrde)
    for k in range(dimensione):
        if (presgrde[k] <= 0.):
            break
        presgrdefix.append(presgrde[k])
    print('presgrdefix = ',presgrdefix)
    edepe = np.zeros(lenalt)
    #edepe = edfillgrd(dimensione,presgrdefix,edgrde,JEDIelecenerflxpj7s2,JEDIestrtenergy,edaltgrd)
    #print('i=',i)
    #for j in range(0,lenalt,100):
    #    print('j=',j,'edepe = ',edepe[j])
    #for j in range(lenalt):
    #    print('j=',j)
    #edepeout[j] = edepeout[j] + edepe[j]
    #print('edepeout = ', edepeout)
    figede = px.scatter(x=edepeout,y=edaltgrd,log_x=True,title='EnergyDeposition(per altitude level)',labels={"y":  "Altitude(cm)", "x": "Energy flux (ev)"})
    figede.add_scatter(x=edepeout, y=edaltgrd, mode='markers',name='electrons')
    figede.update_xaxes(exponentformat="E")
    figede.show()
  '''



