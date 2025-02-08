from pylab import *
import netCDF4 as nc

data1 = nc.Dataset('IonRatesUVS.nc', 'r', format = 'NETCDF4')
data2 = nc.Dataset('IonRatesJEDIonly.nc', 'r', format = 'NETCDF4')

z = data1['Altitude'][:]
H_e1 = data1['H'][:]
H2_e1 = data1['H2'][:]
E_e1 = data1['E'][:]
He_e1 = data1['He'][:]
Hp_e1 = data1['H+'][:]
H3p_e1 = data1['H3+'][:]

H_e2 = data2['H'][:]
H2_e2 = data2['H2'][:]
E_e2 = data2['E'][:]
He_e2 = data2['He'][:]
Hp_e2 = data2['H+'][:]
H3p_e2 = data2['H3+'][:]


#Mendillo et al., 2022
M22 = genfromtxt('data/Mendillo_Hp.txt')
alt = M22[:,0]
Hpc = M22[:,1]

M2 = genfromtxt('data/Mendillo_H3p.txt')
alt2 = M2[:,0]
H3pc = M2[:,1]

#Gerard
G = genfromtxt('data/Gerard_H3p.txt')
H3pG = G[:,0]
altG = G[:,1]

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
ax1 = axs
ax1.plot(Hp_e1, z/1000, 'b--')
ax1.plot(Hp_e2, z/1000, 'b-')
ax1.plot(H3p_e1, z/1000, 'k--')
ax1.plot(H3p_e2, z/1000, 'k-')
ax1.plot(Hpc*1e-6, alt, 'b-.')
ax1.plot(H3pc*1e-6, alt2, 'k-.')
ax1.plot(H3pG, altG, 'r-')
ax1.legend(['H$^{+}$ (JEDI-UVS)' , 'H$^{+}$ (JEDI)', 'H$_{3}^{+}$ (JEDI-UVS)', 'H$_{3}^{+}$ (JEDI)', 'H$^{+}$ (Mendillo et al., 2022)', 'H$_{3}^{+}$ (Mendillo et al., 2022)', 'H$_{3}^{+}$  (Gerard et al., 2023)' ], loc = 'best')
ylabel('Altitude (km)')
xlabel('Concentration (cm^-3)')
xscale('log')
xlim([1e2, 1e7])
ylim([-60, 1300])
#show()
savefig('figs/Jup_Hion.png')