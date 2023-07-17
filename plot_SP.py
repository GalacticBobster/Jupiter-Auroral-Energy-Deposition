from pylab import *

#Variables
#Kinetic   Collision Radiative Total     CSDA      Radiation 
KE_He = []
CSP_He = []
RAD_He = []
TOT_He = []

KE_CH4 = []
CSP_CH4 = []
RAD_CH4 = []
TOT_CH4 = []


#H2 Stopping power from Padovani
dataH2 = genfromtxt('spNIST/H2eSP_Padovani.txt')
KE_H2 = dataH2[:,0]*1e-3 #eV -> keV
TOT_H2 = dataH2[:,1] #eV cm2



#NIST He Stopping power
with open('spNIST/NISTe_HeSP.txt') as fHe:
      for jix in range(8):
        next(fHe)
      for line in fHe:
        dataHe = line.split()
        KE_He.append(float(dataHe[0])*1E3) #MeV -> keV
        TOT_He.append(float(dataHe[3])*1E6*4/6.022E23) #MeV cm^2/g -> eV cm^2
      

#NIST CH4 Stopping power
with open('spNIST/NISTe_CH4SP.txt') as fCH4:
      for jix in range(8):
        next(fCH4)
      for line in fCH4:
        dataCH4 = line.split()
        KE_CH4.append(float(dataCH4[0])*1E3) #MeV -> keV
        TOT_CH4.append(float(dataCH4[3])*1E6*16/6.022E23) #MeV cm^2/g -> eV cm^2
        
#plotting stopping power
fig,axs = subplots(1,1, figsize = (10,10))
axs.plot(KE_H2, TOT_H2, 'k')
axs.plot(KE_He, TOT_He, 'k--')
axs.plot(KE_CH4, TOT_CH4, 'k-.')
axs.set_xlabel('Electron Energy (keV)')
axs.set_ylabel('Loss function (eV cm$^{2}$)')
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlim([32.07, 10000])
axs.legend(['H$_{2}$','He','CH$_{4}$'], loc = 'best')
savefig('SP_all.png')