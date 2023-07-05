#Author: Ananyo Bhattacharya
#Affiliation: University of Michigan, NASA Jet Propulsion Laboratory
#The code calculates the electron cooling rates corresponding to hydrogen rotational and vibrational cooliing due to inelastic cooling
#Rates of cooling are taken from Waite and Cravens (1981), PSS
from pylab import *


def H2_VibWC81(Ne, NH2, Te, Tn):
    W = 0.54 #eV
    Kb = 1.38e-23 #Boltzmann Constant
#First integral
    A1 = -34.95
    B1 = -434
    C1 = -3.915E4
    D1 = 8.554E4
    I1 = exp(A1 + (B1/(Te**0.5)) + (C1/(Te**1.5)) + (D1/(Te**2)))
#Second integral
    A2 = -34.85
    B2 = -676.3
    C2 = -1.066E5
    D2 = 3.573E5
    I2 = exp(A2 + (B2/(Te**0.5)) + (C2/(Te**1.5)) + (D2/(Te**2)))

#VCRT
    VCRT = ((8.37E13*Ne*NH2*W)/(Te**1.5))*((I1*(1-exp((W/Kb)*((Tn - Te)/(Tn*Te))))) + (I2*(2 - (2*exp((2*W/Kb)*((Tn - Te)/(Tn*Te)))))))
    return VCRT

def H2_RotWC81(Ne, NH2, Te, Tn):
   W = 0.54 #eV
   Kb = 1.38e-23 #Boltzmann Constant
   h = 6.626e-34 #Planck's constant
   c = 3e8 #Speed of light
   B = 60.853 #(1/cm) Rotational constant of hydrogen molecule
   SUM_even = 0
   SUM_odd = 0
   Qodd = 0
   Qeven = 0
   for i in range(1,10):
       j = (2*i) -1
       k = (2*i) -2 
       SUM_odd = SUM_odd + ((j+1)*(j+2)*exp(-1*B*h*c*((2*j) + 3)/(Kb*Tn)))
       SUM_even = SUM_odd + ((k+1)*(k+2)*exp(-1*B*h*c*((2*j) + 3)/(Kb*Tn)))
       EJ_odd = j*(j+1)*B*h*c 
       EJ_even = k*(k+1)*B*h*c
       gJ_odd = (2*j) + 1
       gJ_even = (2*k) + 1
       Qodd = Qodd + (gJ_odd*exp(-1*EJ_odd/(Kb*Tn))) 
       Qeven = Qeven + (gJ_even*exp(-1*EJ_even/(Kb*Tn)))

#Odd i.e. J = 1
   Aodd = -36.35
   Bodd = -279.7
   Codd = 2.366E4
   Dodd = -1.791E5
   Iodd = exp(Aodd + (Bodd/(Te**0.5)) + (Codd/(Te**1.5)) + (Dodd/(Te**2)))
   RCRT_odd = ((8.37E13*Ne*NH2)/(Te**1.5))*exp((2*B*h*c)/(Kb*Tn))*exp((10*B*h*c)/(Kb*Te))*((5*B*h*c*Iodd)/Qodd)*SUM_odd 

#Even i.e. J = 0
   Aeven = -35.41
   Beven = -306.2
   Ceven = -4.141E4
   Deven = 1.265E5
   Ieven = exp(Aeven + (Beven/(Te**0.5)) + (Ceven/(Te**1.5)) + (Deven/(Te**2)))
   RCRT_even = ((8.37E13*Ne*NH2)/(Te**1.5))*exp((6*B*h*c)/(Kb*Te))*(3*B*h*c*Ieven*SUM_even)/(Qeven)
   RCRT = (0.25*RCRT_even) + (0.75*RCRT_odd)
   return RCRT



#Electron heating rate 
E_heat = 7.8e6 #eV/cc/s
#Plasma
N_H2 = 2.67e17 #1/cc
N_e1 = 1e10 #1/cc
N_e2 = 1e9 #1/cc
T_n = 153.6
T_e = linspace(250, 300, 10000)

#Cooling rates
VibCR1 = H2_VibWC81(N_e1, N_H2, T_e, T_n)
RotCR1 = H2_RotWC81(N_e1, N_H2, T_e, T_n)
CR1 = VibCR1 + RotCR1


VibCR2 = H2_VibWC81(N_e2, N_H2, T_e, T_n)
RotCR2 = H2_RotWC81(N_e2, N_H2, T_e, T_n)
CR2 = VibCR2 + RotCR2
plot(T_e, (E_heat*ones(len(T_e)) - CR1), 'k-')
plot(T_e, (E_heat*ones(len(T_e)) - CR2), 'k--')
xlabel('Electron temperature (K)')
ylabel('Heating - Cooling rate (eV/cm^3/s)')

yscale('log')
legend(['$N_{e}$ = $10^{10}$/cc', '$N_{e}$ = $10^{9}$/cc'], loc = 'best')
savefig('Jup_Te.png')


