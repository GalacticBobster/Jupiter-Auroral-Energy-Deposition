from pylab import *

#Reading Waite 83 test file
#Pressure | neutral heating  | electron heating
data1 = genfromtxt('Waite_test.txt')
P1 = data1[:,0]
N1 = data1[:,1]
E1 = data1[:,2]

#Reading JEDI UVS output
#Pressure | neutral heating | electron heating
data2 = genfromtxt('CSDA_JEDI_UVS.txt')
H2 = data2[:,0]
N2 = data2[:,1]
E2 = data2[:,2]


#Reading Waite 83 rate
#Rate | Height
data3 = genfromtxt('Waite83_rate.txt')
E3 = data3[:,0]
H3 = data3[:,1]

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



plot(N1, Hgrid, 'k-')
plot(E3, H3, 'r')
plot(E2, H2, 'b')
xscale('log')
xlabel('Heating rate (eV/cm$^{3}$.s)')
ylabel('Altitude (km)')
legend(['10 keV (CSDA)', '10 keV (Waite et al., 1983)', 'JEDI + UVS (CSDA)'], loc = 'best')
xlim([1e2, 1e8])
ylim([0,1000])
for i in range(len(Hgrid)):
    print(Hgrid[i], N2[i])
show()


