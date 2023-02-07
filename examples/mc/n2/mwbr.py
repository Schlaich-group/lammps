import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import root
from scipy.optimize import newton
from scipy import constants
from scipy import integrate
from scipy.optimize.nonlin import NoConvergence

# Modified Benedict-Webb_rubin equation of state

class MBWR:
    """ Modified Benedict-Webb_rubin equation of state """
    
    def __init__(self, rcut=None, factor=1.05, sigma = 0.375 ,epsilon = 95.2, mass=28):

        """ Initialize the MBWR class with given LJ parameters
        
        Input:
            rcut (float, optional):     cutoff (sigma units)
            factor (float, optional):   factor for the initial guess of the
                                        saturation pressure
            sigma (float, optional):    LJ sigma (nm)
            epsilon (float, optional):  LJ epsilon (k_B)
            mass (float, optional):     atomic mass (for correct DeBroglie length)
        
        """

        
        if rcut != None:
            if rcut <= 0:
                raise ValueError("Negative or zero cutoff.")
        self.__rcut = rcut
        self.__gamma = 3. # The nonlinear factor in the MBWR eqs. (cf Johnson et al.)
        self.__factor = factor

        self.__sigma = sigma
        self.__epsilon = epsilon
        self.__mass = mass

        # conversion factors (LJ units)
        self.to_lj_rho = sigma**3 # -> number density nm^(-3)
        self.to_lj_T = 1/epsilon # -> epsilon in Kelvin
        self.to_lj_p = sigma**3/epsilon * constants.bar / (constants.Boltzmann / constants.nano**3)

    def deBroglie(self, T):
        mass = self.__mass * constants.atomic_mass
        db = constants.Planck / np.sqrt(2*np.pi * mass * constants.Boltzmann * T/self.to_lj_T)
        return db/(self.__sigma*constants.nano)
        
    def _x(self):    
        # fitting parameters for the full potential
        """ Tab. 10 in Johnson et.al. """
        x = {}
        x[1] = 0.8623085097507421
        x[2] = 2.976218765822098
        x[3] = -8.402230115796038
        x[4] = 0.1054136629203555
        x[5] = -0.8564583828174598
        x[6] = 1.582759470107601
        x[7] = 0.7639421948305453
        x[8] = 1.753173414312048
        x[9] = 2.798291772190376e+03
        x[10] = -4.8394220260857657e-02
        x[11] = 0.9963265197721935
        x[12] = -3.698000291272493e+01
        x[13] = 2.084012299434647e+01
        x[14] = 8.305402124717285e+01
        x[15] = -9.574799715203068e+02
        x[16] = -1.477746229234994e+02
        x[17] = 6.398607852471505e+01
        x[18] = 1.603993673294834e+01
        x[19] = 6.805916615864377e+01
        x[20] = -2.791293578795945e+03
        x[21] = -6.245128304568454
        x[22] = -8.116836104958410e+03
        x[23] = 1.488735559561229e+01
        x[24] = -1.059346754655084e+04
        x[25] = -1.131607632802822e+02
        x[26] = -8.867771540418822e+03
        x[27] = -3.986982844450543e+01
        x[28] = -4.689270299917261e+03
        x[29] = 2.593535277438717e+02
        x[30] = -2.694523589434903e+03
        x[31] = -7.218487631550215e+02
        x[32] = 1.721802063863269e+02
        return x
    
    def _F(self,rho):
        """ The nonlinear parameter """
        return np.exp(-self.__gamma*rho**2)
    
    def _temperatureCoefficients(self,T):
        
        x = self._x()
        
        """ Tab. 5 in Johnson et al. """
        a = {}
        a[1] = x[1]*T + x[2]*np.sqrt(T) + x[3] + x[4]/T + x[5]/T**2
        a[2] = x[6]*T + x[7] + x[8]/T + x[9]/T**2
        a[3] = x[10]*T + x[11] + x[12]/T
        a[4] = x[13]
        a[5] = x[14]/T + x[15]/T**2
        a[6] = x[16]/T
        a[7] = x[17]/T + x[18]/T**2
        a[8] = x[19]/T**2

        """ Tab. 6 in Johnson et al. """
        b = {}
        b[1] = x[20]/T**2 + x[21]/T**3
        b[2] = x[22]/T**2 + x[23]/T**4
        b[3] = x[24]/T**2 + x[25]/T**3
        b[4] = x[26]/T**2 + x[27]/T**4
        b[5] = x[28]/T**2 + x[29]/T**3
        b[6] = x[30]/T**2 + x[31]/T**3 + x[32]/T**4
        
        """ Tab. 8 in Johnson et al. """
        c = {}
        c[1] = x[2]*np.sqrt(T)/2 + x[3] + 2*x[4]/T + 3*x[5]/T**2
        c[2] = x[7] + 2*x[8]/T + 3*x[9]/T**2
        c[3] = x[11] + 2*x[12]/T
        c[4] = x[13]
        c[5] = 2*x[14]/T + 3*x[15]/T**2
        c[6] = 2*x[16]/T
        c[7] = 2*x[17]/T + 3*x[18]/T**2
        c[8] = 3*x[19]/T**2
        
        return a,b,c
    
    def _densityCoefficients(self,rho):
        """ Tab. 7 in Johnson et al. """
        G = {}
        F = self._F(rho); gamma = self.__gamma
        G[1] =  (1-F)                 / (2*gamma)
        G[2] = -(F*rho**2  -  2*G[1]) / (2*gamma)
        G[3] = -(F*rho**4  -  4*G[2]) / (2*gamma)
        G[4] = -(F*rho**6  -  6*G[3]) / (2*gamma)
        G[5] = -(F*rho**8  -  8*G[4]) / (2*gamma)
        G[6] = -(F*rho**10 - 10*G[5]) / (2*gamma)
        return G

    def P(self,rho,T):
        """ Pressure at given density and temperature according to the MBWR EOS 
        
        Args:
            rho (float or numpy array): Density (LJ units).
            T (float): Temperature (LJ units).

        Returns:
            float or numpy array: Pressure (LJ units)
        
        """

        a,b,c = self._temperatureCoefficients(T)
        # EOS, cf Eq.(7) in Johnson et. al.
        P = rho*T # ideal term
        for i in range(1,9):
            P += a[i]*rho**(i+1)
        for i in range(1,7):
            P += self._F(rho)*b[i]*rho**(2*i+1)
        
        if self.__rcut != None:
            P -= 32/9*np.pi*rho**2* ( (1/self.__rcut)**9
                                    - 3/2 * (1/self.__rcut)**3 )
    
        return P

    def _Arsingle(self,singleRho,T):
        """ residual Helmholtz free energy
        
        Single data point in rho for testing purpose.
        """
        
        a,b,c = self._temperatureCoefficients(T)
        G = self._densityCoefficients(singleRho)
        A = 0.
        for i in range(1,9):
            A += (a[i]*singleRho**i)/i
        for i in range(1,7):
            A += b[i]*G[i]
        
        if self.__rcut != None:
            A -= 32/9*np.pi*singleRho**2* ( (1/self.__rcut)**9
                                    - 3/2 * (1/self.__rcut)**3 )
        
        return A
    
    def Ar(self,rho,T):
        """ residual Helmholtz free energy """
        helper = np.vectorize(self._Arsingle)
        return helper(rho,T)
    
    def Gr(self,rho,P,T):
        """ residual Gibbs free energy """
        return self.Ar(rho,T) + P/rho - T
    
    def G(self,rho,P,T):
        """ Gibbs free energy """

        # Use DeBroglie length with corresponing atomic mass self.__m
        
        return self.Ar(rho,T) + P/rho + T*np.log(rho*self.deBroglie(T)**3)

    def _dmu(self,pressures,T):
        """ Returns the chemical potential difference μ_l - μ_v 
        
        Input:
            pressures:  np.array
            T:          float
        
        Returns:
            dmu:        np.array
        
        """
        vdmu = np.vectorize(self.__dmu)
        return vdmu(p,T)

    def __dmu(self,p,T):
        """ Returns the chemical potential difference μ_l - μ_v """

        rho = np.linspace(0,1.,10000)
        G = self.G(rho,p,T)
        index=argrelextrema(G,np.less)[0]
        if len(index) == 2: # only consider coexistence regime
            return G[index[1]]-G[index[0]]
        else:
            return np.inf

    def __Psat(self,T):
        """ Saturation pressure
        -> Determined by coexistence μ_l - μ_v = 0
        """
                
        def antoine(T, A = 3.31885, B = 7.31828, C = 0.039433):
            """ Initial guess for root searching """
            return np.exp(A-B/(T+C))
        
        if self.__rcut == None:
            Pguess = antoine(T) * self.__factor
        else:
            Pguess = antoine(T) * (1-32./9.*(.8)**2*np.pi*((1/self.__rcut)**9-3/2*(1/self.__rcut)**3))
#         if rcut != None:
#             Pguess += 32/9*np.pi*.33**2* ( (1/self.__rcut)**9
#                                     - 3/2 * (1/self.__rcut)**3 )
#         Ptest = np.linspace(.95*Pguess,1.08*Pguess,100000)
#         dmu = self._dmu(Ptest,T)
#         return Ptest[np.argmin(dmu)]
        
        return newton(self.__dmu,Pguess,args=(T,), maxiter=10000, tol=1e-10)
#         return bisect(self.__dmu,self.__dmu(antoine(T)*.9,T), self.__dmu(antoine(T)*1.1,T), args=(T,))
    
    def Psat(self,T):
        vPsat = np.vectorize(self.__Psat)
        return vPsat(T)
    
    def rhosat(self,T):
        rho = np.linspace(0,1.,10000)
        def helper(T):
            Psat = self.Psat(T)
            G = self.G(rho,Psat,T)
            index=argrelextrema(G,np.less)[0]
            return rho[index]
        vhelper = np.vectorize(helper)
        return vhelper(T)
    
    def critical(self,x=None):
        """ Returns the critical temperature and density according to Eqs (11) and (12) in Johnson et al. """
        def minfunction(x):
            Tc = x[0]
            rhoc = x[1]

            rho = np.linspace(0,1,1000)
            der1 = np.gradient(self.P(rho,Tc),rho)
            der2 = np.gradient(np.gradient(self.P(rho,Tc),rho),rho)

            index = np.argmin(np.abs(rho-rhoc))
            return np.array([der1[index],der2[index]])
        
        if x == None:
            if self.__rcut != None:
                x = [1.35-1.7*self.__rcut**(-2),.3]
            else:
                x = [1.3,.3]
        
        sol = root(minfunction,x, method='hybr', tol=1e-10,
                  options = { 'col_deriv': 0, 'xtol': 1e-8, 'maxfev': 0, 'band': None, 'eps': 1e-2, 'factor': 100, 'diag': None})
        return sol.x, sol.success

    def rho(self,P,T,rhoinit=0):
        """
        Invert the equation of state to obtain the density
        by solving the root P(rho)-P = 0

        rhoinit: starting value for root search
                 as rule of thumb: ~0 for gas, ~1 for liquid
        """
        def helper(rho,P0,T):
            return self.P(rho,T)-P0
        def rhoOfP(P,T,rhoinit):
            try: 
                x = newton(helper, rhoinit, args=(P,T))
            except RuntimeError as msg:
                """ Avoid no convergence error:
                    - in fugacityCoeff the (inverse!) return value is used as integrand
                    - convergence issue only for P>Psat and large T
                    - use density at which the pressure is closest (~ interpolation as
                        quad estimates the accuracy)
                """
                converged = False
                x = float(str(msg).rsplit(' ')[-1])
                #print (msg, 'using', x)
            return x
        vRhoOfP = np.vectorize(rhoOfP)
        return vRhoOfP(P,T,rhoinit)
    
    def fugacityCoeff(self,P,T,rhoinit=0):
        """ Returns the fugacity coefficient """
        # all in LJ units!
        fugaIntegrand = lambda P: 1/(self.rho(P,T,rhoinit)*T) - 1 / P
        lnPhi = lambda P: integrate.quad(fugaIntegrand,0,P)
        vlnPhi = np.vectorize(lnPhi)
        phi = np.exp(vlnPhi(P)[0])
        return phi

    def compressibility(self,P,T):
        return P / ( self.rho(P,T) * T )

    def fsat(self, T):
        """ A helper that directly returns the saturation fugacity """
        Psat = self.Psat(T)
        phisat = self.fugacityCoeff(Psat, T)
        fsat = phisat*Psat
        return fsat
