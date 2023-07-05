# Cooling of thermal electrons comes from H2 and O2 rotational and vibrational excitation
# Use Schunk and Nagy, 'Ionosphere reference'
def eleccrt(altgrd,elecalt,Tn,Te,o2den,h2den):
    k = 8.6171e-5
    Tchk = Te-Tn
    Lo2r = np.array(6.9e-14*elecalt*o2den*(Tchk/np.sqrt(Te)))
    h=(3300.-(839.*np.sin(1.91e-4*(Te-2700.))))
    Lo2v = (-1.)*np.array(5.2e-13*elecalt*o2den*np.exp(h*(Te-700.)/(700.*Te))*(np.exp((-1.)*(2770.*((Tchk)/(Te*Tn))))-1.))
    Lh2r = np.array(2.278e-11*elecalt*h2den*(np.exp(2.093e-4*(pow(Tchk,1.078)-1.))))
    w=0.54
    p1=pow(Te,0.5)
    p2=pow(Te,1.5)
    p3=pow(Te,2.0)
    I1=np.exp((8.554e4/p3)-34.95-(434./p1)-(3.915e4/p2))
    I2=np.exp((3.573e5/p3)-34.85-(673.3/p1)-(1.066e5/p2))
#   print(I1,I2)
    Lh2v = (8.37e13*elecalt*h2den*w/p2)*(I1*(1.-np.exp((w/k)*((1./Te)-(1./Tn))))+2.*I2*(1.-np.exp((2.*w/k)*((1./Te)-(1./Tn)))))
#   print(altgrd,Lo2r,Lo2v,Lh2r,Lh2v)
    crt = np.array(Lo2r+Lh2r+Lo2v+Lh2v)
    return crt
# Calculate integrated heating and cooling rates
ihrt = integrate.simpson(elechrt(altgrd,energyplus,energyfluxplus,nealtarray,Te),altgrd,even='avg')
icrt = integrate.simpson(eleccrt(altgrd,elecalt,Tn,Te,no2den,nh2den),altgrd,even='avg')
nethcrt = ihrt-icrt
    
# plot heating and colling rates versus altitude
figenergy = px.scatter(x=elechrt(altgrd,energyplus,energyfluxplus,nealtarray,Te), y=altgrd/1.0e5, log_x=1, title='Electron Heating and Cooling Rates ',labels={"y":  "Altitude(km)", "x": "Heating Rate(eV cm^-3 s^-1)"})
figenergy.add_scatter(x=elechrt(altgrd,energyplus,energyfluxplus,nealtarray,Te), y=altgrd/1.0e5, mode='markers',name='HRT')
figenergy.add_scatter(x=eleccrt(altgrd,elecalt,Tn,Te,no2den,nh2den), y=altgrd/1.0e5, mode='markers', name='CRT')
figenergy.update_xaxes(exponentformat="E")
figenergy.show()
print('net integrated heating-cooling rates',"{:.2e}".format(nethcrt))

