from pylab import *
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

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


#Truncated cross section value set
data_H2e = genfromtxt('spNIST/H2eSP_Padovani.txt')
LEenrgygrd = data_H2e[:, 0] #eV
LEe = data_H2e[:, 1]*1E-16 #cm^2



E = logspace(-1, 10, 100)
X = intrplee(E,LEenrgygrd,LEe)

func1 = interp1d(log10(LEenrgygrd), log10(LEe), kind='linear',fill_value='extrapolate')
X_lin = 10**func1(log10(E))

func2 = interp1d(log10(LEenrgygrd), log10(LEe), kind='cubic',fill_value='extrapolate')
X_cub = 10**func2(log10(E))



scatter(LEenrgygrd, LEe, c = 'k')
plot(E, X, 'b-')
plot(E, X_lin, 'r-')
plot(E, X_cub, 'C1')
xscale('log')
yscale('log')
legend(['Padovani e-H2 SP', 'Cubic spline', 'log-linear (interp1d)', 'log-cubic (interp1d)'], loc = 'best')
xlabel('Energy (eV)')
ylabel('SP (eV cm^2)')
savefig('figs/SP_Padovani.png')
