import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pumpflux import pumpvalues




class ODE:    
    def __init__(self, STARTTIME=0, STOPTIME=True, DT=True, DTMAX=True, TOLERANCE=True, Pump_flux=True, B=True, Q=True, initpH=7.4, psi_in=0, initH=0.00, initpsi_total=0):
        super(ODE, self).__init__()

        # Standard Constants:
        self.mole=6.02e23   # Avogadro's Number 
        self.RTF=25.69      # RT/F [mV]
        self.F=96485        # Faraday's Constant
        self.cap_0=1e-6     # capacitance per unit area [Farad/cm^2]


        # METHOD STIFF
        self.STARTTIME=STARTTIME
        self.STOPTIME=2000
        self.DT=0.02
        self.DTMAX=100
        self.TOLERANCE=1e-6


        self.B=B
        self.Q=Q
         
        # parameter list ()
        # vesicle shape variables (assumes spherical compartment)
        self.R=0.34         # organelle radius [microns]
        # initial volume [liters]
        self.initV=(4/3*np.pi*self.R**3*1e-12)/1000. 
        # surface area [cm^2]
        self.SA=4*np.pi*self.R**2*1e-8
        # total capacitance [Farad]
        self.cap=self.cap_0*self.SA                    

        
        # surface potentials parameters in mV:        
        self.psi_out=-50    # outside leaflet [mV]
        self.psi_in=psi_in  # inside leaflet [mV]
        
        # cytosolic concentrations parameters
        self.pHbulk=7.2     # for pH [pH units]
        self.Kbulk=0.145    # for K+ [M]
        self.Clbulk=0.01    # for Cl- [M]
        self.Nabulk=0.01    # for Na+ [M]

        # luminal concentrations parameters
        self.initpH=initpH  # pH [pH units]
        self.initk= 0.005    # K+ [M]
        self.initcl=0.11    # Cl- [M]
        self.initna=0.145   # Na+ [M]
        self.initH=initH    # total H+ [M]

        # set initial total membrane potential [mV]
        self.initpsi_total=initpsi_total

        # kinetic parameters
        self.P=6e-5         # H+ permeability
        self.Pcl= 1.2e-5     # Cl+ permeability
        self.Pk= 7.1e-7      # K+ permeability
        self.Pna= 9.6e-7     # Na+ permeability
        self.Pw=0           # Water permeability
        self.N_VATP=300     # Number of V-ATPases
        self.N_CLC=5000     # Number of ClC-7 antiporters
        self.CLC_Cl=2       # ClC-7 Cl- Stoichiometry
        self.CLC_H=1        # ClC-7 H+ Stoichiometry
        self.Oc=0.291       # Cytoplasmic Osmolarity [M]
        self.oh=0.73        # H+ Osmotic coefficient
        self.ok=0.73        # K+ Osmotic coefficient
        self.ona=0.73       # Na+ Osmotic coefficient
        self.ocl=0.73       # Cl- Osmotic coefficient    

        # buffering capacity parameters in [mM/pH unit]
        self.beta=0.04

        self.Pump_flux=Pump_flux







# run ODE simulation   
if __name__ == "__main__":

    ODEobject = ODE()
    ODEobject.main(ODEobject) 
    




