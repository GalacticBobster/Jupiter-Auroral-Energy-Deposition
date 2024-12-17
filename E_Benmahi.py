from pylab import *
from scipy.interpolate import interp1d
#The code simulates the kappa distribution for downward electron flux (Benmahi et al., 2024)


E0 = 1E3 #eV
Q0 = 10*1e-7/1.6e-19 #erg/cm^2.s -> eV/cm^2.s units
k = 2.5
E_m = 2*E0*k/(k - 2)

E = logspace(2,6,100) #Energy band
f_E = Q0*(4*k*(k-1)*E)/(pi*E_m*((k-2)**2))
f = f_E*(E_m**(k-1))/(((2*E/(k-2)) + E_m)**(k+1))


E0 = 10E3 #eV
Q0 = 10*1e-7/1.6e-19 #erg/cm^2.s -> eV/cm^2.s units
E_m = 2*E0*k/(k - 2)

f_E2 = Q0*(4*k*(k-1)*E)/(pi*E_m*((k-2)**2))
f2 = f_E*(E_m**(k-1))/(((2*E/(k-2)) + E_m)**(k+1))

E0 = 100E3 #eV
Q0 = 1*1e-7/1.6e-19 #erg/cm^2.s -> eV/cm^2.s units
E_m = 2*E0*k/(k - 2)

f_E3 = Q0*(4*k*(k-1)*E)/(pi*E_m*((k-2)**2))
f3 = f_E*(E_m**(k-1))/(((2*E/(k-2)) + E_m)**(k+1))


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

#func_intpj7s1 = interp1d(log10(JEDIelecench), log10(JEDIelecintenpj7s1), kind='linear',fill_value='extrapolate')
func_intpj7s2 = interp1d(log10(JEDIelecench), log10(JEDIelecintenpj7s2), kind='linear',fill_value='extrapolate')


#UVS interpolation from guess
uvs_tail = [1000, 10000]
uvs_guess1 = [JEDIelecintenpj7s2[-1], 1e3]
uvs_guess2 = [JEDIelecintenpj7s2[-1], 0.31]
uvs_guess2LB = 0.245
uvs_guess2UB = 0.415

#0.31 is right guess for uvs value equal to 50e3
#0.245 is right guess for uvs lower bound
#0.415 is the right guess for uvs upper bound
#uvs_guessfunc1 = interp1d(log10(uvs_tail), log10(uvs_guess1))
#uvs_guessfunc_lower = interp1d(log10(uvs_tail), log10(uvs_guess1))
#uvs_guessfunc_upper = interp1d(log10(uvs_tail), log10(uvs_guess1))
uvs_guessfunc2 = interp1d(log10(uvs_tail), log10(uvs_guess2))

#Linear extrapolation from JEDI
new_channels = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] #Energies (keV)
#newint_pj7s1 = 10**uvs_guessfunc1(log10(new_channels))
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


#plot(new_channels, newint_pj7s2, 'k--')
#plot(JEDIelecench, JEDIelecintenpj7s2, 'k-')

for inx in range(len(new_channels)):
  JEDIelecench.append(new_channels[inx])
  #JEDIelecintenpj7s1.append(newint_pj7s1[inx])
  JEDIelecintenpj7s2.append(newint_pj7s2[inx])
  

plot(E, f*1e3/pi)
plot(E, f2*1e3/pi)
plot(E, f3*1e3/pi)
plot(array(JEDIelecench)*1e3, JEDIelecintenpj7s2)
xscale('log')
yscale('log')
xlabel('Electron Energy (eV)')
ylabel('Electron flux (electrons/(cm^2 s ster keV))')
show()
