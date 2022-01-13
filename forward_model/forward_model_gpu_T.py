def gen_forward_model(f,Tmax,Ne,Te,Ti,ion_composition):
    import numpy as np
    import scipy.constants as c
    import pycuda.autoinit
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    import skcuda.fft as cufft
    from pycuda.compiler import SourceModule
    import pycuda.cumath
    
    def chirpz_gpu(f,n,dt,dw,wo):
        """transforms f(t) into F(w)
        f(t) is n-point array and output F(w) is n-points starting at wo.
        dt and dw, sampling intervals of f(t) and F(w), and wo are 
        prescribed externally in an idependent manner 
        --- see Li, Franke, Liu [1991]"""
        f_temp = np.copy(f)
        f_temp[0]=0.5*f_temp[0] # first interval is over dt/2, and hence ...

        W_gpu = pycuda.cumath.exp((-1j*dw*dt*(gpuarray.arange(n).astype(np.float64))**2/2.))
        W_cpu = W_gpu.get()
        
        S_gpu = pycuda.cumath.exp(-1j*wo*dt*(gpuarray.arange(n).astype(np.float64)))
        f_gpu = gpuarray.to_gpu(f_temp)
        
        S_gpu = f_gpu*W_gpu*S_gpu; 
        
        f_gpu.gpudata.free()
        W_gpu.gpudata.free()
        
        nFFT = 2*n
        x_gpu = gpuarray.zeros(nFFT,np.complex128)
        x_gpu[:n] = S_gpu

        S_gpu.gpudata.free()

        plan = cufft.Plan(nFFT,np.complex128,np.complex128)
        cufft.fft(x_gpu,x_gpu,plan)
        x_cpu = x_gpu.get()
        x_gpu.gpudata.free()

        Y=pycuda.cumath.exp((1j*dw*dt*(gpuarray.arange(nFFT).astype(np.float64))**2/2.))
        y = Y.get()
        Y.gpudata.free()
        
        y[n:] = y[:n][::-1]
        Y = gpuarray.to_gpu(y)
        
        cufft.fft(Y,Y,plan)
        y_cpu = Y.get()
        Y.gpudata.free()
        
        #F = gpuarray.empty(nFFT,np.complex64)
        F = gpuarray.to_gpu(x_cpu * y_cpu)
        cufft.ifft(F,F,plan,True)

        dest = F[:n].get()
        dest = dest * dt * W_cpu
        
        F.gpudata.free()
      
        return dest
    
    #Ne=152511381727.2139 #Electron density (1/m^3)
    B=15000.0e-9 #Magnetic Field (T)
    # Ion Composition
    NH,N4,NO=(1-ion_composition)/2*Ne,(1.-ion_composition)/2*Ne,ion_composition*Ne
    #Te,TH,T4,TO=1000,1000,1000,1000
    TH,T4,TO = Ti,Ti,Ti
    # Physical Paramters (MKS):
    me=c.electron_mass#9.1093826e-31 # Electron mass in kg
    mH,m4,mO=c.atomic_mass,4.*c.atomic_mass,16.*c.atomic_mass # Ion mass

    qe=c.electron_volt#1.60217653e-19 # C (Electron charge)
    K=c.k#1.3806505e-23 # Boltzmann cobstant m^2*kg/(s^2*K);
    eps0=c.epsilon_0#8.854187817e-12 # F/m (Free-space permittivity)
    c=c.c#299.792458e6 # m/s (Speed of light)
    re=2.817940325e-15 # Electron radius

    Ce,CH,C4,CO=np.sqrt(K*Te/me),np.sqrt(K*TH/mH),np.sqrt(K*T4/m4),np.sqrt(K*TO/mO) # Thermal speeds (m/s)
    Omge,OmgH,Omg4,OmgO=qe*B/me,qe*B/mH,qe*B/m4,qe*B/mO # Gyro-frequencies

    # Debye Lengths
    debe,debH,deb4,debO=np.sqrt(eps0*K*Te/(Ne*qe**2)),np.sqrt(eps0*K*TH/(NH*qe**2)),\
    np.sqrt(eps0*K*T4/(N4*qe**2)),np.sqrt(eps0*K*TO/(NO*qe**2))
    debp=1./np.sqrt(1./debe/debe+1./debH/debH+1./deb4/deb4+1./debO/debO) # Plasma Debye Length
    
    #Tmax=10*1.0e-6 #total integration time for electron Gordeyev integral 
    #Tmax = 4e-6
    N = len(f)
    dt=Tmax/N
    dw = (f[1]-f[0])*2*np.pi
    wo = f[0] * 2*np.pi
    w = 2*np.pi*f
        # the following pseudocode is for Coulomb collision of species s with species p 
    # vTsp=sqrt(2*(Cs**2+Cp**2)) #most probable interaction speed of s={e,H,O} with p={e,H,O}
    # msp=ms*mp/(ms+mp) #reduced mass for s and p
    # bm_sp=qs*qp/(4*pi*eps0)/msp/vTsp**2 #bmin for s and p
    # log_sp=log(debp/bm_sp) #coulomb logarithm for s and p
    # nusp=Np*qs**2*qp**2*log_sp/(3*pi**(3/2)*eps0**2*ms*msp*vTsp**3) # collision freq of s with p --- 2.104 Callen 2003

    # electron-electron
    vTee=np.sqrt(2.*(Ce**2+Ce**2))
    mee=me*me/(me+me)
    bm_ee=qe*qe/(4*np.pi*eps0)/mee/vTee**2
    log_ee=np.log(debp/bm_ee)
    nuee=Ne*qe**2*qe**2*log_ee/(3*np.pi**(3/2)*eps0**2*me*mee*vTee**3) 
    # electron-hydrogen
    vTeH=np.sqrt(2*(Ce**2+CH**2))
    meH=me*mH/(me+mH)
    bm_eH=qe*qe/(4*np.pi*eps0)/meH/vTeH**2
    log_eH=np.log(debp/bm_eH)
    nueH=NH*qe**2*qe**2*log_eH/(3*np.pi**(3/2)*eps0**2*me*meH*vTeH**3)
    # electron-helium
    vTe4=np.sqrt(2*(Ce**2+C4**2))
    me4=me*m4/(me+m4)
    bm_e4=qe*qe/(4*np.pi*eps0)/me4/vTe4**2
    log_e4=np.log(debp/bm_e4)
    nue4=N4*qe**2*qe**2*log_e4/(3*np.pi**(3/2)*eps0**2*me*me4*vTe4**3)
    # electron-oxygen
    vTeO=np.sqrt(2*(Ce**2+CO**2))
    meO=me*mO/(me+mO)
    bm_eO=qe*qe/(4*np.pi*eps0)/meO/vTeO**2
    log_eO=np.log(debp/bm_eO)
    nueO=NO*qe**2*qe**2*log_eO/(3*np.pi**(3/2)*eps0**2*me*meO*vTeO**3)
    # electron Coulomb collision frequency
    nue=nuee+nueH+nue4+nueO
    nuel=nueH+nue4+nueO
    nuep=nuel+nuee

    # hydrogen-electron
    vTHe=np.sqrt(2*(CH**2+Ce**2))
    mHe=mH*me/(mH+me)
    bm_He=qe*qe/(4*np.pi*eps0)/mHe/vTHe**2
    log_He=np.log(debp/bm_He)
    nuHe=Ne*qe**2*qe**2*log_He/(3*np.pi**(3/2)*eps0**2*mH*mHe*vTHe**3)
    # hydrogen-hydrogen
    vTHH=np.sqrt(2.*(CH**2+CH**2))
    mHH=mH*mH/(mH+mH)
    bm_HH=qe*qe/(4*np.pi*eps0)/mHH/vTHH**2
    log_HH=np.log(debp/bm_HH)
    nuHH=NH*qe**2*qe**2*log_HH/(3*np.pi**(3/2)*eps0**2*mH*mHH*vTHH**3) 
    # hydrogen-helium
    vTH4=np.sqrt(2*(CH**2+C4**2))
    mH4=mH*m4/(mH+m4)
    bm_H4=qe*qe/(4*np.pi*eps0)/mH4/vTH4**2
    log_H4=np.log(debp/bm_H4)
    nuH4=N4*qe**2*qe**2*log_H4/(3*np.pi**(3/2)*eps0**2*mH*mH4*vTH4**3)
    # hydrogen-oxygen
    vTHO=np.sqrt(2*(CH**2+CO**2))
    mHO=mH*mO/(mH+mO)
    bm_HO=qe*qe/(4*np.pi*eps0)/mHO/vTHO**2
    log_HO=np.log(debp/bm_HO)
    nuHO=NO*qe**2*qe**2*log_HO/(3*np.pi**(3/2)*eps0**2*mH*mHO*vTHO**3)
    # hydrogen Coulomb collision frequency
    nuH=nuHe+nuHH+nuH4+nuHO

    # helium-electron
    vT4e=np.sqrt(2*(C4**2+Ce**2))
    m4e=m4*me/(m4+me)
    bm_4e=qe*qe/(4*np.pi*eps0)/m4e/vT4e**2
    log_4e=np.log(debp/bm_4e)
    nu4e=Ne*qe**2*qe**2*log_4e/(3*np.pi**(3/2)*eps0**2*m4*m4e*vT4e**3)
    # helium-hydrogen
    vT4H=np.sqrt(2.*(C4**2+CH**2))
    m4H=m4*mH/(m4+mH)
    bm_4H=qe*qe/(4*np.pi*eps0)/m4H/vT4H**2
    log_4H=np.log(debp/bm_4H)
    nu4H=NH*qe**2*qe**2*log_4H/(3*np.pi**(3/2)*eps0**2*m4*m4H*vT4H**3) 
    # helium-helium
    vT44=np.sqrt(2*(C4**2+C4**2))
    m44=m4*m4/(m4+m4)
    bm_44=qe*qe/(4*np.pi*eps0)/m44/vT44**2
    log_44=np.log(debp/bm_44)
    nu44=N4*qe**2*qe**2*log_44/(3*np.pi**(3/2)*eps0**2*m4*m44*vT44**3)
    # helium-oxygen
    vT4O=np.sqrt(2*(C4**2+CO**2))
    m4O=m4*mO/(m4+mO)
    bm_4O=qe*qe/(4*np.pi*eps0)/m4O/vT4O**2
    log_4O=np.log(debp/bm_4O)
    nu4O=NO*qe**2*qe**2*log_4O/(3*np.pi**(3/2)*eps0**2*m4*m4O*vT4O**3)
    # helium Coulomb collision frequency
    nu4=nu4e+nu4H+nu44+nu4O

    # oxygen-electron
    vTOe=np.sqrt(2*(CO**2+Ce**2))
    mOe=mO*me/(mO+me)
    bm_Oe=qe*qe/(4*np.pi*eps0)/mOe/vTOe**2
    log_Oe=np.log(debp/bm_Oe)
    nuOe=Ne*qe**2*qe**2*log_Oe/(3*np.pi**(3/2)*eps0**2*mO*mOe*vTOe**3)
    # oxygen-hydrogen
    vTOH=np.sqrt(2*(CO**2+CH**2))
    mOH=mO*mH/(mO+mH)
    bm_OH=qe*qe/(4*np.pi*eps0)/mOH/vTOH**2
    log_OH=np.log(debp/bm_OH)
    nuOH=NH*qe**2*qe**2*log_OH/(3*np.pi**(3/2)*eps0**2*mO*mOH*vTOH**3)
    # oxygen-helium
    vTO4=np.sqrt(2*(CO**2+C4**2))
    mO4=mO*m4/(mO+m4)
    bm_O4=qe*qe/(4*np.pi*eps0)/mO4/vTO4**2
    log_O4=np.log(debp/bm_O4)
    nuO4=N4*qe**2*qe**2*log_O4/(3*np.pi**(3/2)*eps0**2*mO*mO4*vTO4**3)
    # oxygen-osxygen
    vTOO=np.sqrt(2.*(CO**2+CO**2))
    mOO=mO*mO/(mO+mO)
    bm_OO=qe*qe/(4*np.pi*eps0)/mOO/vTOO**2
    log_OO=np.log(debp/bm_OO)
    nuOO=NO*qe**2*qe**2*log_OO/(3*np.pi**(3/2)*eps0**2*mO*mOO*vTOO**3) 
    # oxygen Coulomb collision frequency
    nuO=nuOe+nuOH+nuO4+nuOO

    fradar=430.0e6 # Radar Frequency (Hz)
    lam=c/fradar/2.
    kB=2*np.pi/lam # Bragg wavenumber kB = 2*ko
    aspect=45.*np.pi/180. # Aspect angle (rad) with 0 perp to Bs
    
    t=np.arange(N)*dt 
    #varel=(Ce*t)**2; varep=((2*Ce/Omge)*sin(Omge*t/2))**2 # collisionless
    varel=((2.*Ce**2)/nuel**2)*(nuel*t-1+np.exp(-nuel*t)) # collisional
    gam=np.arctan(nuep/Omge)
    varep=((2.*Ce**2)/(nuep**2+Omge**2))*(np.cos(2*gam)+nuep*t-np.exp(-nuep*t)*np.cos(Omge*t-2*gam))
    acfe=np.exp(-((kB*np.sin(aspect))**2)*varel/2.)*np.exp(-((kB*np.cos(aspect))**2)*varep/2.)
    Ge=chirpz_gpu(acfe,N,dt,dw,wo) # Electron Gordeyev Integral
    
    # Oxygen Gordeyev integral (Brownian)
    dtO=dt*100
    t=np.arange(N)*dtO #adjust dt such that full range of acfi is covered by range t
    #varil=(CO*t)**2; varip=((2*CO/OmgO)*sin(OmgO*t/2))**2 # collisionless
    varil=((2.*CO**2)/nuO**2)*(nuO*t-1+np.exp(-nuO*t)) # collisional
    gam=np.arctan(nuO/OmgO)
    varip=((2.*CO**2)/(nuO**2+OmgO**2))*(np.cos(2*gam)+nuO*t-np.exp(-nuO*t)*np.cos(OmgO*t-2*gam))
    acfO=np.exp(-((kB*np.sin(aspect))**2)*varil/2.)*np.exp(-((kB*np.cos(aspect))**2)*varip/2.) 
    GO=chirpz_gpu(acfO,N,dtO,dw,wo) # Ion Gordeyev Integral
    
    # Helium Gordeyev integral (Brownian) 
    dt4=dt*50
    t=np.arange(N)*dt4 #adjust dt such that full range of acfi is covered by range t
    #varil=(C4*t)**2; varip=((2*C4/Omg4)*sin(Omg4*t/2))**2 # collisionless
    varil=((2.*C4**2)/nu4**2)*(nu4*t-1+np.exp(-nu4*t)) # collisional
    gam=np.arctan(nu4/Omg4)
    varip=((2.*C4**2)/(nu4**2+Omg4**2))*(np.cos(2*gam)+nu4*t-np.exp(-nu4*t)*np.cos(Omg4*t-2*gam))
    acf4=np.exp(-((kB*np.sin(aspect))**2)*varil/2.)*np.exp(-((kB*np.cos(aspect))**2)*varip/2.) # page-337 in ppr-II equa-42
    G4=chirpz_gpu(acf4,N,dt4,dw,wo) # Ion Gordeyev Integral
   
    # Hydrogen Gordeyev integral (Brownian) 
    dtH=dt*25
    t=np.arange(N)*dtH #adjust dt such that full range of acfi is covered by range t

    #varil=(CH*t)**2; varip=((2*CH/OmgH)*sin(OmgH*t/2))**2
    varil=((2.*CH**2)/nuH**2)*(nuH*t-1+np.exp(-nuH*t)) # page-337 in ppr-II equa-43
    gam=np.arctan(nuH/OmgH)
    varip=((2.*CH**2)/(nuH**2+OmgH**2))*(np.cos(2*gam)+nuH*t-np.exp(-nuH*t)*np.cos(OmgH*t-2*gam))
    acfH=np.exp(-((kB*np.sin(aspect))**2)*varil/2.)*np.exp(-((kB*np.cos(aspect))**2)*varip/2.) # page-337 in ppr-II equa-42
    GH=chirpz_gpu(acfH,N,dtH,dw,wo) # Ion Gordeyev Integral
    # Total ISR Spectrum
    yO=(1-1j*w*GO)/(kB**2*debO**2) # oxygen admittance  
    y4=(1-1j*w*G4)/(kB**2*deb4**2) # helium admittance
    yH=(1-1j*w*GH)/(kB**2*debH**2) # hydrogen admittance
    ye=(1-1j*w*Ge)/(kB**2*debe**2) # electron admittance
    
    spec=np.real(Ne*2*Ge)*abs((1+yH+y4+yO)/(1+ye+yH+y4+yO))**2+ \
            np.real(NH*2*GH+N4*2*G4+NO*2*GO)*abs((ye)/(1+ye+yH+y4+yO))**2
    
    return spec
    
    
    

