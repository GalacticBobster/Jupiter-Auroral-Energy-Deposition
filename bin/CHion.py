from pylab import *
import netCDF4 as nc

data1 = nc.Dataset('Photochem_v3/IonRatesUVS.nc', 'r', format = 'NETCDF4')
data2 = nc.Dataset('Photochem_v3/IonRatesJEDIonly.nc' , 'r', format = 'NETCDF4')
data3 = nc.Dataset('Photochem_v3/IonRatesJEDI01.nc' , 'r', format = 'NETCDF4')
data4 = nc.Dataset('Photochem_v3/IonRatesJEDI001.nc' , 'r', format = 'NETCDF4')

z1 = data1['Altitude'][:]/1e3

Ec1 = data1['CH5+'][:]
Ec2 = data2['CH5+'][:]
Ec3 = data3['CH5+'][:]
Ec4 = data4['CH5+'][:]
E1 = data1['C2H5+'][:]
E2 = data2['C2H5+'][:]
E3 = data3['C2H5+'][:]
E4 = data4['C2H5+'][:]


fig, axs = plt.subplots(1, 1, figsize=(6, 6))
ax1 = axs
ax1.plot(Ec1, z1,'k-')
ax1.plot(Ec2, z1,'r-')
ax1.plot(Ec3, z1,'C1-')
ax1.plot(Ec4, z1,'C1--')
ax1.plot(E1, z1,'k--')
ax1.plot(E2, z1,'r--')
ax1.plot(E3, z1,'C1--')
ax1.plot(E4, z1,'C7--')
ax1.set_xlabel('Concentration (cm$^{-3}$)')
ax1.set_ylabel('Altitude (km)')
ax1.set_xscale('log')
ax1.legend(['CH$_{5}^{+}$ (JEDI-UVS)', 'CH$_{5}^{+}$ (JEDI)', 'CH$_{5}^{+}$ (0.1 x JEDI-UVS)', 'CH$_{5}^{+}$ (0.01 x JEDI-UVS)', 'C$_{2}$H$_{5}^{+}$ (JEDI-UVS)', 'C$_{2}$H$_{5}^{+}$ (JEDI)', 'C$_{2}$H$_{5}^{+}$ (0.1 x JEDI-UVS)', 'C$_{2}$H$_{5}^{+}$ (0.01 x JEDI-UVS)'], loc = 'best')
ax1.set_xlim([1e2, 1e12])
ylim([-60, 1300])
show()
savefig('figs/Jup_C2.png')
