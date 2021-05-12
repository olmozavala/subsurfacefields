import numpy as np

# ============== Page 34 of the ISOP Validation test report ========================
def MLD(s, t, depths):
    """Computes the Mixed Layer Depth (MLD) from the T/S profiles and the corresponding depth levels.
    MLD is the thickness of a surface layer that has nearly constant temperature, salinity, and density
    https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2000JC900072
    http://www.ifremer.fr/cerweb/deboyer/mld/Surface_Mixed_Layer_Depth.php
    :param T:
    :param S:
    :param depths
    :return:
    """
    # From page 12 or 5 of ISOP
    # Compute the potential density
    th_sigma = [0.15, 0.05, 0.025, 0.01, 0.001]  # km/m^3
    svan, density = swstate(s, t, depths)

    ref_density = np.interp(4, depths, density)  # Density at 4 mts (reference density)
    MLD = 500  # Just start with a number that will not trigger the while
    th_i = 0   # Threshold index
    while MLD > 400:
        c_th = th_sigma[th_i] # Current threshold
        for i in range(2, len(depths)):
            c_density = density[i] # Current density
            if c_density - ref_density > c_th:
                MLD = depths[i]
                break
            else:
                ref_density = density[i]
        th_i += 1

    return MLD


def SLD(gt_t, gt_s, pred_t, pred_s, d):
    """Computes the Sonic Layer Depth (SLD) from the T/S profiles and the corresponding depth levels.
    SLD is the vertical distance from the ocean surface to the depth of a sound speed maximum.
    :param t:
    :param s:
    :param d:
    :return:
    """
    return 1


def MB(obs, syn):
    """Mean Bias (MB) between observation and synthetic profile"""
    return np.nanmean(syn-obs)

def RMSE(obs, syn):
    """RMSE between observation and synthetic profile"""
    return np.sqrt(np.nanmean((syn-obs)**2))


def swstate(s, t, depths):
    R3500=1028.1063
    R4=4.8314E-4
    DR350=28.106331
    p=depths/10
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
    # % t = 40 DEG C, depths= 10000 DECIBARs.
    # % *******************************************************
    dr35p=gam/v350p
    dvan=sva/(v350p*(v350p+sva))
    sigma=DR350+dr35p-dvan  #% Density anomaly

    k=k35+dk
    vp=1.0-p/k
    v = (1.) /(sigma+1000.0)

    return svan, sigma

def hycom2sigma(t, s, nterm=17):
    if (nterm==17):
        c001= 9.9984085444849347E+02    #!num. constant    coefficent
        c002= 7.3471625860981584E+00    #!num.    T        coefficent
        c003=-5.3211231792841769E-02    #!num.    T^2      coefficent
        c004= 3.6492439109814549E-04    #!num.    T^3      coefficent
        c005= 2.5880571023991390E+00    #!num.       S     coefficent
        c006= 6.7168282786692355E-03    #!num.    T  S     coefficent
        c007= 1.9203202055760151E-03    #!num.       S^2   coefficent
        c008= 1.0000000000000000E+00    #!den. constant    coefficent
        c009= 7.2815210113327091E-03    #!den.    T        coefficent
        c010=-4.4787265461983921E-05    #!den.    T^2      coefficent
        c011= 3.3851002965802430E-07    #!den.    T^3      coefficent
        c012= 1.3651202389758572E-10    #!den.    T^4      coefficent
        c013= 1.7632126669040377E-03    #!den.       S     coefficent
        c014= 8.8066583251206474E-06    #!den.    T  S     coefficent
        c015= 1.8832689434804897E-10    #!den.    T^3S     coefficent
        c016= 5.7463776745432097E-06    #!den.    T  S^1.5 coefficent
        c017= 1.4716275472242334E-09    #!den.    T^3S^1.5 coefficent
        #
        c018= 1.1798263740430364E-02    #!num. P           coefficent
        c019= 9.8920219266399117E-08    #!num. P  T^2      coefficent
        c020= 4.6996642771754730E-06    #!num. P     S     coefficent
        c021= 2.5862187075154352E-08    #!num. P^2         coefficent
        c022= 3.2921414007960662E-12    #!num. P^2T^2      coefficent
        c023= 6.7103246285651894E-06    #!den. P           coefficent
        c024= 2.4461698007024582E-17    #!den. P^2T^3      coefficent
        c025= 9.1534417604289062E-18    #!den. P^3T        coefficent
        #
        prs2pdb=1.E-4       #!Pascals to dbar
        pref=2000.E4        #!ref. pressure in Pascals, sigma2
        rpdb=pref*prs2pdb   #!ref. pressure in dbar
        #
        c101=c001+(c018-c021*rpdb)*rpdb #num. constant    coefficent
        c103=c003+(c019-c022*rpdb)*rpdb #num.    T^2      coefficent
        c105=c005+c020*rpdb             #num.       S     coefficent
        c108=c008+c023*rpdb             #den. constant    coefficent
        c109=c009-c025*rpdb^3           #den.    T        coefficent
        c111=c011-c024*rpdb^2           #den.    T^3      coefficent
        #
        sig_n = c101 + t*(c002+t*(c103+t*c004)) + s*(c105-t*c006+s*c007)
        sig_d = c108 + t*(c109+t*(c010+t*(c111+t*c012))) + s*(c013-t*(c014+t*t*c015) + max(0,s)^0.5*(c016+t*t*c017))
        aout = sig_n/sig_d - 1000.0
    elif (nterm==9):
        c1= 9.903308E+00  #const. coefficent
        c2=-1.618075E-02  #T      coefficent
        c3= 7.819166E-01  #   S   coefficent
        c4=-6.593939E-03  #T^2    coefficent
        c5=-2.896464E-03  #T  S   coefficent
        c6= 3.038697E-05  #T^3    coefficent
        c7= 3.266933E-05  #T^2S   coefficent
        c8= 1.180109E-04  #   S^2 coefficent
        c9= 3.399511E-06  #T  S^2 coefficent
        aout = c1+s*(c3+s* c8)+ t*(c2+s*(c5+s*c9)+t*(c4+s*c7+t*c6))
    else:
        print('unknown nterm (either 9 or 17)')
