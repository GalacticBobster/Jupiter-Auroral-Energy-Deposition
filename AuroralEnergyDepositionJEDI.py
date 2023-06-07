# This is a Python code to read in the JEDI precipitating energetic proton and electron fluxes and calculate the
# pressure level of energy depositionfrom which one can derive ionization, heating production, emissions, etc.
# The code requires a JEDI input file and uses the H2 loss function of PAdovani et al. A&A 501, 619–631 (2009)
# DOI: 10.1051/0004-6361/200911794.

#Import libraries
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

#change to working directory
os.chdir('/Users/hwaite/Desktop/JUNO 2016 2023/JEDIjupiteraurora')

# Read in atmsopheric data file from Randy Gladstone
asatm_header = ["ALT","TEMP","PRESS","[H]","[H2]","[He]","[CH4]","[C2H2]","[C2H4]","[C2H6]","[CH3C2H]","[C3H8]","[C4H2]","[C4H10]"]
dfatm = pd.read_csv('jupiterAurAtm_cleaned.txt',names=asatm_header,skiprows=1, sep=' ')
dfatm = dfatm.astype('float128')
dfatm.info()
#print('ATM', dfatm)
#Store data for altitude from body center, H2, He, H, CH4, temperature, pressure (barye) in arrays for later use
altgrd = np.array(dfatm['ALT']+66854.)
temperature = np.array(dfatm['TEMP'])
pressure = np.array(dfatm['PRESS'])
Hden = np.array(dfatm['[H]'])
H2den = np.array(dfatm['[H2]'])
Heden = np.array(dfatm['[He]'])
CH4den = np.array(dfatm['[CH4]'])
invpres = np.flip(pressure)
invaltgrd = np.flip(altgrd)
#Subroutines for converting altitude to pressure and pressure to altitude, pressure in barye
def altprs(altin,altgrd,pressure):
    csaltprs = CubicSpline(altgrd, pressure, bc_type='not-a-knot')
    presout = csaltprs(altin)
    return presout
def prsalt(presin,invpres,invaltgrd):
    csprsalt = CubicSpline(invpres, invaltgrd, bc_type='not-a-knot')
    altout = csprsalt(presin)
    return altout
#check
presin = 1e6
altout = prsalt(presin,invpres,invaltgrd)
altin = 66854.
presout = altprs(altin,altgrd,pressure)
#print('ALTOUT = ',altout, 'PRESOUT = ',presout)

# Portion of code to read in the JEDI data binned in eV


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

# Functions for interpolation
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
      
#Check Interpolation
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
def presnH(pressurebar):
    nH = (pressurebar*conversion)/(m*g)
    return nH
def nHpres(nH):
    pressurebar = nH*m*g/conversion
    return pressurebar

#determine pressure/column depth step size to use in energy degradation scheme

pelp = 0.1
pele = 0.1

# Determine energy loss grid
def dimp(JEDIpstrtenergy,pelp):
# Initial energy increment
    el = pelp*JEDIpstrtenergy
    ie = 0
    energy = JEDIpstrtenergy
    for i in range(10000000):
        ie = ie + 1
        energy = energy - el
        if (energy < 1e2):
            return ie
        el = energy*pelp               
def dime(JEDIestrtenergy,pele):
# Initial energy increment
    el = pele*JEDIestrtenergy
    ie = 0
    energy = JEDIestrtenergy
    for i in range(10000000):
        ie = ie + 1
        energy = energy - el
        if (energy < 1e2):
            return ie
        el = energy*pele

# Portion of code containing the energy degradation functions
def pendegrade(JEDIpstrtenergy,dimensionp,pelp):
    JEDIped[0] = JEDIpstrtenergy
    nHp[0] = 0.0
    presgrdp[0] = 0.0
    edgrdp[0] = 0
    for j in range(dimensionp-1):
        edgrdp[j+1] = pelp*JEDIped[j]
        JEDIped[j+1] = JEDIped[j] - edgrdp[j+1]
        nHtmp = edgrdp[j]/intrplep(JEDIped[j],LEenrgygrd,LEp)
        nHp[j+1] = nHp[j]+nHtmp
        presgrdp[j+1] = nHpres(nHp[j+1])
#        print (edgrdp[j+1],nHp[j+1],presgrdp[j+1],JEDIped[j+1])
        if (JEDIped[j+1] <1e2):
            print('I am here p')
            EDpoutputdf = pd.DataFrame(data={'Pressurebar':presgrdp,'Column depth':nHp,'EnergyDeposition':edgrdp,'ProtonEnergy':JEDIped})
            return EDpoutputdf
def eendegrade(JEDIestrtenergy,dimensione,pele):
    JEDIped[0] = JEDIpstrtenergy
    nHe[0] = 0.0
    presgrde[0] = 0.0
    edgrde[0] = 0
    for j in range(dimensione-1):
        edgrde[j+1] = pele*JEDIeed[j]
        JEDIeed[j+1] = JEDIeed[j] - edgrde[j+1]
        nHtmp = edgrde[j]/intrplee(JEDIeed[j],LEenrgygrd,LEe)
        nHe[j+1] = nHe[j]+nHtmp
        presgrde[j+1] = nHpres(nHe[j+1])
#        print (edgrde[j+1],nHe[j+1],presgrde[j+1],JEDIeed[j+1])
        if (JEDIeed[j+1] < 1e2):
            print('I am here e')
            EDeoutputdf = pd.DataFrame(data={'Pressurebar':presgrde,'Column depth':nHe,'EnergyDeposition':edgrde,'ProtonEnergy':JEDIeed})
            return EDeoutputdf   
    
# Portion of code that iterates over JEDI energy spectrum     
for i in range(len(LEenrgygrd)):
    icntstrtp = 0
    icntstrte = 0
    JEDIpstrtenergy = LEenrgygrd[i]
    JEDIpindex = i
    JEDIeindex = i
    elfp = LEp[i]
    JEDIestrtenergy = LEenrgygrd[i]
    elfe = LEe[i]
#    print ('JEDIpstrtenergy = ',JEDIpstrtenergy,'JEDIestrtenergy = ',JEDIestrtenergy)
    dimensionp = dimp(JEDIpstrtenergy,pelp)
#    print('dimensionp = ', dimensionp)
    JEDIped = np.zeros(dimensionp,dtype=float)
    JEDIped[icntstrtp] = JEDIpstrtenergy
    nHp = np.zeros(dimensionp,dtype=float)
    presgrdp = np.zeros(dimensionp,dtype=float)
    edgrdp = np.zeros(dimensionp,dtype=float)
    dimensione = dime(JEDIestrtenergy,pele)
#    print('dimensione = ', dimensione)
    JEDIeed = np.zeros((dimensione),dtype=float)
    JEDIeed[icntstrte] = JEDIestrtenergy
    nHe = np.zeros(dimensione,dtype=float)
    presgrde = np.zeros(dimensione,dtype=float)
    edgrde = np.zeros(dimensione,dtype=float)
    EDpoutput_iteration = pendegrade(JEDIpstrtenergy,dimensionp,pelp)
    EDeoutput_iteration = eendegrade(JEDIestrtenergy,dimensione,pele)
#Produce figures
#Protons
    figedp = px.scatter(x=edgrdp,y=presgrdp,log_x=True,log_y=True,title='Proton Energy Deposition (per pressure level)',labels={"y":  "Pressure(bar)", "x": "Protons eV per pressure step"})
    figedp.add_scatter(x=edgrdp, y=presgrdp, mode='markers',name='protons')
    figedp.update_yaxes(exponentformat="E")
    figedp.show()
#Electrons
    figede = px.scatter(x=edgrde,y=presgrde,log_x=True,log_y=True,title='EnergyDeposition (per pressure level)',labels={"y":  "Pressure(bar)", "x": "Electrons eV per pressure step"})
    figede.add_scatter(x=edgrde, y=presgrde, mode='markers',name='electrons')
    figede.update_yaxes(exponentformat="E")
    figede.show()            
#Store output in a Panda Data Frame foprmat
    EDpoutputdf = EDpoutputdf.append(EDpoutput_iteration,ignore_index = True)
    EDeoutputdf = EDeoutputdf.append(EDeoutput_iteration,ignore_index = True)
    print(EDpoutputdf)
