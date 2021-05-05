import numpy as np
import pytest

def swstate(s, t, p0):
    R3500=1028.1063
    R4=4.8314E-4
    DR350=28.106331
    p=p0/10
    sal=s
    sR = np.sqrt(abs(s))
    #  *********************************************************
    # pURE WATER DENsITY AT ATMOspHERIC pREssURE
    #   BIGG p.H.,(1967) BR. J. AppLIED pHYsICs 8 pp 521-537.
    R1 = ((((6.536332E-9*t-1.120083E-6)*t+1.001685E-4)* t-9.095290E-3)*t+6.793952E-2)*t-28.263737
    # sEAWATER DENsITY ATM pREss.
    #  COEFFICIENTs INvOLvING sALINITY
    R2 = (((5.3875E-9*t-8.2467E-7)*t+7.6438E-5)*t-4.0899E-3)*t+8.24493E-1
    R3 = (-1.6546E-6*t+1.0227E-4)*t-5.72466E-3
    # INTERNATIONAL ONE-ATMOspHERE EQUATION OF sTATE OF sEAWATER
    sIG = (R4*s + R3*sR + R2)*s + R1
    # % spECIFIC vOLUME AT ATMOspHERIC pREssURE
    v350p = 1.0/R3500
    sva = -sIG*v350p/(R3500+sIG)
    sigma=sIG+DR350
    v0 = 1.0/(1000.0 + sigma)
    # sCALE spECIFIC vOL. ANAMOLY TO NORMALLY REpORTED UNITs
    svan=sva*1.0E+8

    # % ******************************************************************
    # % ******  NEW HIGH pREssURE EQUATION OF sTATE FOR sEAWATER ********
    # % ******************************************************************
    # %        MILLERO, ET AL , 1980 DsR 27A, pp 255-264
    # %               CONsTANT NOTATION FOLLOWs ARTICLE
    # %********************************************************
    # % COMpUTE COMpREssION TERMs
    E = (9.1697E-10*t+2.0816E-8)*t-9.9348E-7
    BW = (5.2787E-8*t-6.12293E-6)*t+3.47718E-5
    B = BW + E*s    #% Bulk Modulus (almost)
    # %  CORRECT B FOR ANAMOLY BIAs CHANGE
    Bout = B + 5.03217E-5

    D = 1.91075E-4
    C = (-1.6078E-6*t-1.0981E-5)*t+2.2838E-3
    AW = ((-5.77905E-7*t+1.16092E-4)*t+1.43713E-3)*t-0.1194975
    A = (D*sR + C)*s + AW
    # %  CORRECT A FOR ANAMOLY BIAs CHANGE
    Aout = A + 3.3594055

    B1 = (-5.3009E-4*t+1.6483E-2)*t+7.944E-2
    A1 = ((-6.1670E-5*t+1.09987E-2)*t-0.603459)*t+54.6746
    kW = (((-5.155288E-5*t+1.360477E-2)*t-2.327105)*t+148.4206)*t-1930.06
    k0 = (B1*sR + A1)*s + kW

    # EvALUATE pREssURE pOLYNOMIAL
    # ***********************************************
    #   k EQUALs THE sECANT BULk MODULUs OF sEAWATER
    #   dk=k(s,t,p)-k(35,0,p)
    #  k35=k(35,0,p)
    # ***********************************************
    dk = (B*p + A)*p + k0
    k35  = (5.03217E-5*p+3.359406)*p+21582.27
    gam=p/k35
    pk = 1.0 - gam
    sva = sva*pk + (v350p+sva)*p*dk/(k35*(k35+dk))
    # %  sCALE spECIFIC vOL. ANAMOLY TO NORMALLY REpORTED UNITs
    svan=sva*1.0E+8      #% volume anomaly
    v350p = v350p*pk
    # %  ****************************************************
    # % COMpUTE DENsITY ANAMOLY WITH REspECT TO 1000.0 kG/M**3
    # %  1) DR350: DENsITY ANAMOLY AT 35 (Ipss-78), 0 DEG. C AND 0 DECIBARs
    # %  2) dr35p: DENsITY ANAMOLY 35 (Ipss-78), 0 DEG. C ,  pREs. vARIATION
    # %  3) dvan : DENsITY ANAMOLY vARIATIONs INvOLvING spECFIC vOL. ANAMOLY
    # % ********************************************************************
    # % CHECk vALUE: sigma = 59.82037  kG/M**3 FOR s = 40 (Ipss-78),
    # % t = 40 DEG C, p0= 10000 DECIBARs.
    # % *******************************************************
    dr35p=gam/v350p
    dvan=sva/(v350p*(v350p+sva))
    sigma=DR350+dr35p-dvan  #% Density anomaly

    k=k35+dk
    vp=1.0-p/k
    v = (1.) /(sigma+1000.0)

    return svan, sigma


if __name__ == "__main__":
    t = 40
    s = 40
    p0 = 10000
    svan, sigma = swstate(s,t,p0)
    assert svan == pytest.approx(981.3021, .0001)
    assert sigma == pytest.approx(59.82037, .0001)
    print("Yeah babe!!!!!")

    svan, sigma = swstate(np.array([40,40]), np.array([40, 40]), np.array([p0,p0]))