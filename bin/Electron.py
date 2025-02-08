from pylab import *
import netCDF4 as nc

data1 = nc.Dataset('Photochem_v3/IonRatesUVS.nc', 'r', format = 'NETCDF4')
data2 = nc.Dataset('Photochem_v3/IonRatesJEDIonly.nc' , 'r', format = 'NETCDF4')
data3 = nc.Dataset('Photochem_v3/IonRatesJEDI01.nc' , 'r', format = 'NETCDF4')
data4 = nc.Dataset('Photochem_v3/IonRatesJEDI001.nc' , 'r', format = 'NETCDF4')
data5 = nc.Dataset('Photochem/IonRatesInit.nc' , 'r', format = 'NETCDF4')

z1 = data1['Altitude'][:]/1e3

Ec1 = data1['E'][:]
Ec2 = data2['E'][:]
Ec3 = data3['E'][:]
Ec4 = data4['E'][:]
Ec5 = data5['E'][:]

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
ax1 = axs
ax1.plot(Ec1, z1,'k-')
ax1.plot(Ec2, z1,'k--')
ax1.plot(Ec3, z1,'C7-')
ax1.plot(Ec4, z1,'C7--')
ax1.plot(Ec5, z1,'C1--')

ax1.set_xlabel('Electron Concentration (cm$^{-3}$)')
ax1.set_ylabel('Altitude (km)')
ax1.set_xscale('log')
ax1.legend(['JEDI-UVS', 'JEDI', '0.1 x JEDI-UVS', '0.01 x JEDI-UVS', 'Photochemical ionosphere (Waite et al., 1983)'], loc = 'best')
ax1.set_xlim([1e2, 1e12])
ylim([-60, 1300])
show()