import glob
from scipy import integrate
import numpy as np
from scipy.signal import argrelmax
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pylab as plt

def relaxation_T1_no_noise(x, M0, T1inv): #x(ms), M0(V), T1inv(1/s)
    return M0 * (1.0 - np.exp(-x/1000*T1inv))

def relaxation_T1(x, A, M0, T1inv): #x(ms), A(V), M0(V), T1inv(1/s)
    return A + M0* (1.0 - np.exp(-x/1000*T1inv))

def relaxation_T1_2com_no_noise(x, M0, T1inv1, r, T1inv2): #x(ms), M0(V), T1inv1(1/s), r, T1inv2(1/s)
    return M0 * (1.0 - np.exp(-x/1000*T1inv1) + r * (1.0 - np.exp(-x/1000*T1inv2)))

def relaxation_T1_2com(x, A, M0, T1inv1, r, T1inv2): #x(ms), A(V), M0(V), T1inv1(1/s), r, T1inv2(1/s)
    return A + M0 * (1.0 - np.exp(-x/1000*T1inv1) + r * (1.0 - np.exp(-x/1000*T1inv2)))

def relaxation_T1_3com_no_noise(x, M0, T1inv1, r, T1inv2, s, T1inv3): #x(ms), M0(V), T1inv1(1/s), r = M01/M02,  T1inv2(1/s), s = M02/M03, T1inv3(1/s)
    return M0 * (1.0 - np.exp(-x/1000*T1inv1) + r * (1.0 - np.exp(-x/1000*T1inv2) + s * (1.0 - np.exp(-x/1000*T1inv3))))

def relaxation_T1_3com(x, A, M0, T1inv1, r, T1inv2, s, T1inv3): #x(ms), A(V), M0(V), T1inv1(1/s), r = M01/M02,  T1inv2(1/s), s = M02/M03, T1inv3(1/s)
    return A + M0 * (1.0 - np.exp(-x/1000*T1inv1) + r * (1.0 - np.exp(-x/1000*T1inv2) + s * (1.0 - np.exp(-x/1000*T1inv3))))

def relaxation_T1_beta(x, a, b, T1inv, beta): #x(ms), a(V), b(V), T1inv(1/s)
    return a + b*np.exp(-(x/1000*T1inv)**beta)

def gaussian_fit(x, a, b, FWHM): #a(V), b(MHz), FWHM(MHz)
    return a*np.exp(-4*np.log(2)*((x-b)/FWHM)**2)

def dblgaussian_fit(x, a1, b1, FWHM1,a2, b2, FWHM2): #a(V), b(MHz), FWHM(MHz)
    return a1*np.exp(-4*np.log(2)*((x-b1)/FWHM1)**2)+a2*np.exp(-4*np.log(2)*((x-b2)/FWHM2)**2)

def tplgaussian_fit(x, a1, b1, FWHM1, a2, b2, FWHM2, a3, b3, FWHM3): #a(V), b(MHz), FWHM(MHz)
    return a1*np.exp(-4*np.log(2)*((x-b1)/FWHM1)**2) + a2*np.exp(-4*np.log(2)*((x-b2)/FWHM2)**2) + a3*np.exp(-4*np.log(2)*((x-b3)/FWHM3)**2)

def TempCX(T, B):
    ##CX-1010-SD-HT-0.1M #100722
    return T - 0.04 * B / (T ** 0.3)

 

##### constants ######
I = 3.0/2.0 #7Li, I=3/2 for 63Cu
gammaLi = 16.5468008 #MHz/T, 7Li
gammaCu = 11.2893305 #MHz/T, 63Cu

##### file loading #####

Li_num = 'Li2'

T1_ended = 'yes'
fssum_ended = 'yes'
gauss = 'd'## s = single gaussian, d = double gaussian, t = triple gaussian 


header = '20180601-betaLi2IrO3_S47hishi2-AP-b-4T'

if T1_ended=='yes':
    for name in glob.glob(header+'7'+str(Li_num)+'-*-T1-auto.dat'):
        T1 = name
##for name in glob.glob('*-T2-rect50.dat'):
##    T2 = name

for name in glob.glob(header+'63Cu-*-td.dat'):
    tdCu = name
skip_tdCu = 1#50003#only header->1

for name in glob.glob(header+'63Cu-*-ft.dat'):
    ftCu = name
skip_ftCu = 1#65539#only header->1


for name in glob.glob(header+'7'+str(Li_num)+'-*-td.dat'):
    tdLi = name
##for name in glob.glob(header+'7'+str(Li_num)+'-*-ft.dat'):
##    ftLi = name
if fssum_ended == 'yes':
    for name in glob.glob(header+'7'+str(Li_num)+'*-fssum-autotune.dat'):
       fssum = name

#fssum = '20180117-betaLi2IrO3_S47hishi1-run1-4T7Li-rep10000-tau20-5.0K-67.0MHz-fssum-asis.dat'


##### parameters to change ######
P = 0.0 ##GPa
T = 40.0 ##K
B_nominal = 4 #T

T_err = T/40.0

rep = 1000.0#ms of fssum measurement
tau = 20.0#us of fssum measurement


cfreqCu = 45.61#MHz
if Li_num =='Li0':
    cfreqLi = 66.26#MHz    ##cfreq is not pfreq.
if Li_num =='Li1':
    cfreqLi = 66.914#MHz    ##cfreq is not pfreq.
if Li_num =='Li2':
    cfreqLi = 67.008#MHz    ##cfreq is not pfreq.
if Li_num =='Li3':
    cfreqLi = 68.071#MHz    ##cfreq is not pfreq.

pfreqCu =  cfreqCu#MHz
pfreqLi =  cfreqLi#MHz

#### plot range of td #####
xCu = [-0.1,0.2]
yCu = [-0.01,0.01]

if Li_num == 'Li0':
    xLi = [-0.02, 0.05]
    yLi = [-0.02, 0.02]
if Li_num == 'Li1':
    xLi = [-0.02, 0.05]
    yLi = [-0.2, 0.2]
if Li_num == 'Li2':
    xLi = [-0.02, 0.05]
    yLi = [-0.2, 0.2]
if Li_num == 'Li3':
    xLi = [-0.02, 0.05]
    yLi = [-0.2, 0.2]

#### T1 fitting parameters ######
fit_option = 's'

Lcut_P1 = 0.0
Ucut_P1 = 10**(10)

#s
iniT1inv = 0.1#1/s

#d
ini_r = 0.5
iniT1inv1 = 0.1#1/s
iniT1inv2 = 10.0

#t
ini_r = 1.0
ini_s = 1.0
iniT1inv1 = 0.1#1/s
iniT1inv2 = 1.0
iniT1inv3 = 10.0

iniBeta = 1.0

#### T2 fitting parameters ######
iniT2 = [0.02,7]
iniT2inv = 7 #1/(ms)

#iniT2_2com = [0.003,7.0,0.003,10.0]

#### Cu FT spectrum fitting parameters ######
iniCu = [0.002, pfreqCu, 0.02] #[Intens(V), pfreq(MHz), FWHM(MHz)]
widthCu = 0.5 # width = 1 => fitting on FWHM

#### Li FT spectrum fitting parameters ######
iniLi = [0.025, pfreqLi, 0.03] #[Intens(V), pfreq(MHz), FWHM(MHz)]
widthLi = 0.5 # width = 1 => fitting on FWHM

#### fssum dbl & tpl gaussian fit ####
sglg_ini = [0.0031796, 68.127, 0.038605]
dblg_ini = [0.01097, 66.629, 0.364,\
            0.005822, 67.903, 0.36232]
tplg_ini = [0.0055496, 66.834, 0.05, \
            0.017265, 66.911, 0.070877, \
            0.0039502, 67.063, 0.056152]
Lcut = 66.0
Ucut = 68.5

#### spesctrum hosei ######
width = 3.0 #width = 1 => integration on FWHM
k = 12 #2**k+1 = 2**12+1 = 4097 = the number of sampling in the integration region

#################################
if T1_ended == 'yes':
    filename = T1
    data = np.loadtxt(filename, skiprows=1, usecols=(0,1,2))
    P1 = data[:,0]#ms
    Intens = data[:,1]#V
    Weight = data[:,2]#1/V
    
    P1_lim = []
    Intens_lim = []
    sigma_lim = []
    for i in range(len(P1)):
        if P1[i] > Lcut_P1 and P1[i] < Ucut_P1:
            P1_lim.append(P1[i])
            Intens_lim.append(Intens[i])
            sigma_lim.append(1.0 / Weight[i])

    if fit_option == 's':
        ini = [Intens_lim[0], Intens[len(Intens_lim)-1], iniT1inv]#A(V), M0(V), T1inv(1/s)
        param, cov = curve_fit(relaxation_T1, P1_lim, Intens_lim, p0 = ini, sigma = sigma_lim)
        error = np.sqrt(np.diag(cov))

        plt.figure()
        plt.xscale("log")
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1(P1, *param))
        Noise = np.full_like(P1, param[0])
        plt.plot (P1, Noise)
        plt.plot (P1, relaxation_T1_no_noise(P1, param[1], param[2]))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-' + str(T) +'K_'+str(Li_num)+'.eps')

        plt.figure()
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1(P1, *param))
        plt.plot (P1, Noise)
        plt.plot (P1, relaxation_T1_no_noise(P1, param[1], param[2]))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-' + str(T) +'K-linear_'+str(Li_num)+'.eps')

        results = np.zeros(11)
        results[0] = param[0]#A
        results[1] = error[0]
        results[2] = param[1]#M0
        results[3] = error[1]
        results[4] = param[2]#T1inv
        results[5] = error[2]
        results[6] = 1/param[2]*1000 #T1 (ms)
        results[7] = 0.0
        results[8] = error[0] / param[0] * 100.0
        results[9] = error[1] / param[1] * 100.0
        results[10] = error[2] / param[2] * 100.0


        filename='T1_fit-'+str(fit_option)+'-' + str(T) +'K_'+str(Li_num)+'.csv'
        np.savetxt(filename, results)

        ##fitting without noise ##
        ini = [Intens_lim[len(Intens_lim)-1], iniT1inv]#M0(V), T1inv(1/s)
        param, cov = curve_fit(relaxation_T1_no_noise, P1_lim, Intens_lim, p0 = ini, sigma = sigma_lim)
        error = np.sqrt(np.diag(cov))

        plt.figure()
        plt.xscale("log")
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_no_noise(P1, *param))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-no_noise-' + str(T) +'K_'+str(Li_num)+'.eps')

        plt.figure()
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_no_noise(P1, *param))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-no_noise-' + str(T) +'K-linear_'+str(Li_num)+'.eps')

        results = np.zeros(8)
        results[0] = param[0]#M0
        results[1] = error[0]
        results[2] = param[1]#T1inv
        results[3] = error[1]
        results[4] = 1/param[1]*1000 #T1 (ms)
        results[5] = 0.0
        results[6] = error[0] / param[0] * 100.0
        results[7] = error[1] / param[1] * 100.0


        filename='T1_fit-'+str(fit_option)+'-no_noise-' + str(T) +'K_'+str(Li_num)+'.csv'
        np.savetxt(filename, results)

    if fit_option == 'd':
        ini = [Intens_lim[0], Intens_lim[len(Intens_lim)-1], iniT1inv1, ini_r, iniT1inv2]#A, M0, T1inv1, r, T1inv2
        param, cov = curve_fit(relaxation_T1_2com, P1_lim, Intens_lim, p0 = ini, sigma = sigma_lim)
        error = np.sqrt(np.diag(cov))

        plt.figure()
        plt.xscale("log")
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_2com(P1, *param))
        Noise = np.full_like(P1, param[0])
        plt.plot (P1, Noise)
        plt.plot (P1, relaxation_T1_no_noise(P1, param[1], param[2]))
        plt.plot (P1, relaxation_T1_no_noise(P1, param[1] * param[3], param[4]))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-' + str(T) +'K_'+str(Li_num)+'.eps')

        plt.figure()
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_2com(P1, *param))
        Noise = np.full_like(P1, param[0])
        plt.plot (P1, Noise)
        plt.plot (P1, relaxation_T1_no_noise(P1, param[1], param[2]))
        plt.plot (P1, relaxation_T1_no_noise(P1, param[1] * param[3], param[4]))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-' + str(T) +'K-linear_'+str(Li_num)+'.eps')

        results = np.zeros(20)
        results[0] = param[0]#A
        results[1] = error[0]
        results[2] = param[1]#M0
        results[3] = error[1]
        results[4] = param[2]#T1inv1
        results[5] = error[2]
        results[6] = 1/param[2]*1000 #T11 (ms)
        results[7] = 0.0
        results[8] = param[3]#r
        results[9] = error[3]
        results[10] = 0.0
        results[11] = param[4]#T1inv2
        results[12] = error[4]
        results[13] = 1/param[4]*1000 #T12 (ms)
        results[14] = 0.0
        results[15] = error[0] / param[0] * 100.0
        results[16] = error[1] / param[1] * 100.0
        results[17] = error[2] / param[2] * 100.0
        results[18] = error[3] / param[3] * 100.0
        results[19] = error[4] / param[4] * 100.0
        

        filename='T1_fit-'+str(fit_option)+'-' + str(T) +'K_'+str(Li_num)+'.csv'
        np.savetxt(filename, results)

        ## fitting without noise ###
        ini = [Intens_lim[len(Intens_lim)-1], iniT1inv1, ini_r, iniT1inv2]#M0, T1inv1, r, T1inv2
        param, cov = curve_fit(relaxation_T1_2com_no_noise, P1_lim, Intens_lim, p0 = ini, sigma = sigma_lim)
        error = np.sqrt(np.diag(cov))

        plt.figure()
        plt.xscale("log")
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_2com_no_noise(P1, *param))
        plt.plot (P1, relaxation_T1_no_noise(P1, param[0], param[1]))
        plt.plot (P1, relaxation_T1_no_noise(P1, param[0] * param[2], param[3]))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-no_noise-' + str(T) +'K_'+str(Li_num)+'.eps')

        plt.figure()
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_2com_no_noise(P1, *param))
        plt.plot (P1, relaxation_T1_no_noise(P1, param[0], param[1]))
        plt.plot (P1, relaxation_T1_no_noise(P1, param[0] * param[2], param[3]))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-no_noise-' + str(T) +'K-linear_'+str(Li_num)+'.eps')

        results = np.zeros(17)
        results[0] = param[0]#M0
        results[1] = error[0]
        results[2] = param[1]#T1inv1
        results[3] = error[1]
        results[4] = 1/param[1]*1000 #T11 (ms)
        results[5] = 0.0
        results[6] = param[2]#r
        results[7] = error[2]
        results[8] = 0.0
        results[9] = param[3]#T1inv2
        results[10] = error[3]
        results[11] = 1/param[3]*1000 #T12 (ms)
        results[12] = 0.0
        results[13] = error[0] / param[0] * 100.0
        results[14] = error[1] / param[1] * 100.0
        results[15] = error[2] / param[2] * 100.0
        results[16] = error[3] / param[3] * 100.0
        

        filename='T1_fit-'+str(fit_option)+'-no_noise-' + str(T) +'K_'+str(Li_num)+'.csv'
        np.savetxt(filename, results)

    if fit_option == 't':
        ini = [Intens_lim[0], Intens_lim[len(Intens_lim)-1], iniT1inv1, ini_r, iniT1inv2, ini_s, iniT1inv3]#A(V), M0(V), T1inv1(1/s), r = M01/M02, T1inv2(1/s), s = M02/M03, T1inv3(1/s)
        param, cov = curve_fit(relaxation_T1_3com, P1_lim, Intens_lim, p0 = ini, sigma = sigma_lim)
        error = np.sqrt(np.diag(cov))
        A = param[0]
        M0 = param[1]
        T1inv1 = param[2]
        r = param[3]
        T1inv2 = param[4]
        s = param[5]
        T1inv3 = param[6]

        dA = error[0]
        dM0 = error[1]
        dT1inv1 = error[2]
        dr = error[3]
        dT1inv2 = error[4]
        ds = error[5]
        dT1inv3 = error[6]
        
        plt.figure()
        plt.xscale("log")
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_3com(P1, *param))
        Noise = np.full_like(P1, A)
        plt.plot (P1, Noise)
        plt.plot (P1, relaxation_T1_no_noise(P1, M0, T1inv1))
        plt.plot (P1, relaxation_T1_no_noise(P1, M0 * r, T1inv2))
        plt.plot (P1, relaxation_T1_no_noise(P1, M0 * r * s, T1inv3))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-' + str(T) +'K_'+str(Li_num)+'.eps')

        plt.figure()
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_2com(P1, *param))
        Noise = np.full_like(P1, A)
        plt.plot (P1, Noise)
        plt.plot (P1, relaxation_T1_no_noise(P1, M0, T1inv1))
        plt.plot (P1, relaxation_T1_no_noise(P1, M0 * r, T1inv2))
        plt.plot (P1, relaxation_T1_no_noise(P1, M0 * r * s, T1inv3))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-' + str(T) +'K-linear_'+str(Li_num)+'.eps')

        T11 = 1.0 / T1inv1 * 1000.0
        T12 = 1.0 / T1inv2 * 1000.0
        T13 = 1.0 / T1inv3 * 1000.0
        results = [A, dA, '### comp1 ###', M0, dM0, T1inv1, dT1inv, T11, '### comp2 ###', r, dr, T1inv2, dT1inv2, T12, '### comp3 ###', s, ds, T1inv3, T13, \
                   '### error ratio (%) ###', error[0] / param[0] * 100.0, error[1] / param[1] * 100.0, error[2] / param[2] * 100.0, error[3] / param[3] * 100.0,\
                   error[4] / param[4] * 100.0, error[5] / param[5] * 100.0, error[6] / param[6] * 100.0]

        filename='T1_fit-'+str(fit_option)+'-' + str(T) +'K_'+str(Li_num)+'.csv'
        np.savetxt(filename, results)

        ## fitting without noise ###
        ini = [Intens_lim[len(Intens_lim)-1], iniT1inv1, ini_r, iniT1inv2, ini_s, iniT1inv3]#M0(V), T1inv1(1/s), r = M01/M02, T1inv2(1/s), s = M02/M03, T1inv3(1/s)
        param, cov = curve_fit(relaxation_T1_3com_no_noise, P1_lim, Intens_lim, p0 = ini, sigma = sigma_lim)
        error = np.sqrt(np.diag(cov))
        [M0, T1inv1, r, T1inv2, s, T1inv3] = param
        [dM0, dT1inv1, dr, dT1inv2, ds, dT1inv3] = error
        
        plt.figure()
        plt.xscale("log")
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_3com_no_noise(P1, *param))
        Noise = np.full_like(P1, A)
        plt.plot (P1, Noise)
        plt.plot (P1, relaxation_T1_no_noise(P1, M0, T1inv1))
        plt.plot (P1, relaxation_T1_no_noise(P1, M0 * r, T1inv2))
        plt.plot (P1, relaxation_T1_no_noise(P1, M0 * r * s, T1inv3))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-' + str(T) +'K_'+str(Li_num)+'.eps')

        plt.figure()
        plt.plot (P1, Intens, 'x')
        plt.plot (P1, relaxation_T1_2com_no_noise(P1, *param))
        Noise = np.full_like(P1, A)
        plt.plot (P1, Noise)
        plt.plot (P1, relaxation_T1_no_noise(P1, M0, T1inv1))
        plt.plot (P1, relaxation_T1_no_noise(P1, M0 * r, T1inv2))
        plt.plot (P1, relaxation_T1_no_noise(P1, M0 * r * s, T1inv3))
        plt.xlabel ('P1 (ms)')
        plt.ylabel ('Intens (V)')
        plt.savefig('T1_fit-'+str(fit_option)+'-' + str(T) +'K-linear_'+str(Li_num)+'.eps')

        T11 = 1.0 / T1inv1 * 1000.0
        T12 = 1.0 / T1inv2 * 1000.0
        T13 = 1.0 / T1inv3 * 1000.0
        results = ['### comp1 ###', M0, dM0, T1inv1, dT1inv, T11, '### comp2 ###', r, dr, T1inv2, dT1inv2, T12, '### comp3 ###', s, ds, T1inv3, T13, \
                   '### error ratio (%) ###', error[0] / param[0] * 100.0, error[1] / param[1] * 100.0, error[2] / param[2] * 100.0, error[3] / param[3] * 100.0,\
                   error[4] / param[4] * 100.0, error[5] / param[5] * 100.0]

        filename='T1_fit-'+str(fit_option)+'-' + str(T) +'K_'+str(Li_num)+'.csv'
        np.savetxt(filename, results)



######## plot of tdCu #########
##filename = tdCu
##data = np.loadtxt(filename, skiprows=skip_tdCu, usecols=(0,1,2,3,4))
##ms = data[:,0]#kHz
##Re = data[:,1]#V
##Im = data[:,2]#V
##DSOch1 = data[:,3]#V
##DSOch2 = data[:,4]#V
##
##plt.figure()
##plt.plot (ms, DSOch1)
##plt.plot (ms, DSOch2)
##plt.plot (ms, Re, lw=4)
##plt.plot (ms, Im, lw=4)
##
##plt.xlabel ('time (ms)')
##plt.ylabel ('Intensity (V)')
##plt.xlim(xCu[0], xCu[1])
##plt.ylim(yCu[0], yCu[1])
##plt.savefig('Cu_td-' + str(T) +'K.eps')

######## plot of tdLi #########
##filename = tdLi
##data = np.loadtxt(filename, skiprows=1, usecols=(0,1,2,3,4))
##ms = data[:,0]#kHz
##Re = data[:,1]#V
##Im = data[:,2]#V
##DSOch1 = data[:,3]#V
##DSOch2 = data[:,4]#V
##
##plt.figure()
##plt.plot (ms, DSOch1)
##plt.plot (ms, DSOch2)
##plt.plot (ms, Re, lw=4)
##plt.plot (ms, Im, lw=4)
##
##plt.xlabel ('time (ms)')
##plt.ylabel ('Intensity (V)')
##plt.xlim(xLi[0], xLi[1])
##plt.ylim(yLi[0], yLi[1])
##plt.savefig('Li_td-' + str(T) +'K_'+str(Li_num)+'.eps')

##### peak search for ftCu #####
filename = ftCu
data = np.loadtxt(filename, skiprows=skip_ftCu, usecols=(0,3))
dfreq = data[:,0]#kHz
absV = data[:,1]#V

freq = np.zeros(dfreq.size)
freq = dfreq/1000.0 + cfreqCu #MHz

pfreqCu = freq[np.argmax(absV)]
KCu = 0.19955 + (0.20084 - 0.19955)/(300.0 - 1.0)*(T - 1.0) - 0.001*P #%

##ini=iniCu
##param, cov = curve_fit(gaussian_fit, freq, absV, p0=ini)#fitting only for getting xlim
##error = np.sqrt(np.diag(cov))

plt.figure()
plt.plot (freq, absV)
plt.plot (pfreqCu, absV[np.argmax(absV)],'o')

plt.xlabel ('freq (MHz)')
plt.ylabel ('Intensity (V)')
#plt.xlim(param[1] - param[2] * 5.0, param[1] + param[2] * 5.0)
plt.savefig('ftCu_peak' + str(T) +'K.eps')

B = pfreqCu/gammaCu/(1.0 + KCu/100.0)
Bdev = (B-B_nominal)/B_nominal*100.0

results = np.zeros(4)
results[0]=pfreqCu
results[1]=KCu
results[2]=B
results[3]=Bdev

filename='ftCu_peak-'+str(T)+'K.csv'
np.savetxt(filename, results)

##### field calibratioin of thermometer ####
Tcalib = TempCX(T,B)
Tdev = (Tcalib - T)/T*100.0
Tcalib_err = T_err*np.absolute(1.0 + 0.04 * 0.3 * B * T**(-1.3))

### T & P -> KCu -> B, Tcalib

results = np.zeros(3)
results[0] = Tcalib
results[1] = Tcalib_err
results[2] = Tdev

filename='calibrated_temp-'+str(T)+'K.csv'
np.savetxt(filename, results)

##if T1_ended=='yes':
##    ####### T1T #######
##    T1Tinv = T1inv/Tcalib #s-1K-1
##    T1Tinv_err = T1Tinv*np.sqrt((dT1inv/T1inv)**2+(Tcalib_err/Tcalib)**2) #s-1K-1
##
##    results = np.zeros(2)
##    results[0] = T1Tinv
##    results[1] = T1Tinv_err
##
##    filename='T1T-'+str(T)+'K_'+str(Li_num)+'.csv'
##    np.savetxt(filename, results)

if fssum_ended == 'yes' and gauss == 's':
    ###### fssum plot #####
    filename = fssum
    data = np.loadtxt(filename, skiprows=1, usecols=(0,4,5))
    freq = data[:,0]#MHz
    absV = data[:,1]#V
    darkV = data[:,2]#V

    plt.figure()
    plt.plot(freq, absV)
    plt.xlabel ('freq (MHz)')
    plt.ylabel ('spectrum (V)')
    plt.savefig('fssum' + str(T) +'K_'+str(Li_num)+'.eps')
    
    filename='fssum_absV'+str(T)+'K_'+str(Li_num)+'.csv'
    np.savetxt(filename, absV)

    filename='fssum_freq'+str(T)+'K_'+str(Li_num)+'.csv'#MHz
    np.savetxt(filename, freq)
    
    Ks = (freq/gammaLi/B-1.0)*100.0 #%
    filename='fssum_Ks'+str(T)+'K_'+str(Li_num)+'.csv'
    np.savetxt(filename, Ks)

    ini=sglg_ini

    freq_lim = []
    absV_lim = []
    for i in range(len(freq)):
        if freq[i] > Lcut and freq[i] < Ucut:
            freq_lim.append(freq[i])
            absV_lim.append(absV[i])

    Weight = np.ones_like(freq_lim)
    for i in range(len(freq_lim)):
        if freq_lim[i] > ini[1]-ini[2]/2.0 and freq_lim[i] < ini[1]+ini[2]/2.0:
            Weight[i] = 10.0
            
    param, cov = curve_fit(gaussian_fit, freq_lim, absV_lim, p0=ini, sigma=1/Weight)
    error = np.sqrt(np.diag(cov))

    plt.figure()
    plt.plot (freq, absV)
    plt.plot (freq, gaussian_fit(freq, *param))

    plt.xlim(param[1] - param[2] * 5.0, param[1] + param[2] * 5.0)
    plt.xlabel ('freq (MHz)')
    plt.ylabel ('Intensity (V)')
    plt.savefig('fssum_sglgaussian_fit-' + str(T) +'K_'+str(Li_num)+'.eps')

    pfreqLi_fit1 = param[1] #MHz
    pfreqLi_fit1_err = error[1] #MHz
    K1 = (pfreqLi_fit1/gammaLi/B-1.0)*100 #%
    K1_err = (1.0+K1/100.0)*100.0*np.absolute(pfreqLi_fit1_err/pfreqLi_fit1)#%
    FWHM1_kHz = param[2]*1000 #kHz 
    FWHM1_kHz_err = error[2]*1000 #kHz

    results = np.zeros(12)
    results[0] = pfreqLi_fit1
    results[1] = pfreqLi_fit1_err
    results[2] = K1
    results[3] = K1_err
    results[4] = param[0]
    results[5] = error[0]
    results[6] = FWHM1_kHz
    results[7] = FWHM1_kHz_err
    results[8] = 0.0
    results[9] = error[0]/param[0]*100.0
    results[10] = error[1]/param[1]*100.0
    results[11] = error[2]/param[2]*100.0


    filename='fssum_sglgaussian_fit-'+str(T)+'K_'+str(Li_num)+'_result.csv'
    np.savetxt(filename, results)

    filename='fssum_'+str(T)+'K_sgl_gauss.csv'
    np.savetxt(filename, gaussian_fit(freq, *param))


if fssum_ended == 'yes' and gauss == 'd':
    ###### fssum plot #####
    filename = fssum
    data = np.loadtxt(filename, skiprows=1, usecols=(0,4,5))
    freq = data[:,0]#MHz
    absV = data[:,1]#V
    darkV = data[:,2]#V

    plt.figure()
    plt.plot(freq, absV)
    plt.xlabel ('freq (MHz)')
    plt.ylabel ('spectrum (V)')
    plt.savefig('fssum' + str(T) +'K_'+str(Li_num)+'.eps')
    
    filename='fssum_absV'+str(T)+'K_'+str(Li_num)+'.csv'
    np.savetxt(filename, absV)

    filename='fssum_freq'+str(T)+'K_'+str(Li_num)+'.csv'#MHz
    np.savetxt(filename, freq)
    
    Ks = (freq/gammaLi/B-1.0)*100.0 #%
    filename='fssum_Ks'+str(T)+'K_'+str(Li_num)+'.csv'
    np.savetxt(filename, Ks)

    ini=dblg_ini

    freq_lim = []
    absV_lim = []
    for i in range(len(freq)):
        if freq[i] > Lcut and freq[i] < Ucut:
            freq_lim.append(freq[i])
            absV_lim.append(absV[i])

    Weight = np.ones_like(freq_lim)
    for i in range(len(freq_lim)):
        if freq_lim[i] > ini[1]-ini[2]/4.0 and freq_lim[i] < ini[1]+ini[2]/4.0:
            Weight[i] = 10.0
        if freq_lim[i] > ini[4]-ini[5]/4.0 and freq_lim[i] < ini[4]+ini[5]/4.0:
            Weight[i] = 10.0
            
    param, cov = curve_fit(dblgaussian_fit, freq_lim, absV_lim, p0=ini, sigma=1/Weight)
    error = np.sqrt(np.diag(cov))

    plt.figure()
    plt.plot (freq, absV)
    plt.plot (freq, dblgaussian_fit(freq, *param))
    plt.plot (freq, gaussian_fit(freq, param[0], param[1], param[2]))
    plt.plot (freq, gaussian_fit(freq, param[3], param[4], param[5]))

    plt.xlabel ('freq (MHz)')
    plt.ylabel ('Intensity (V)')
    plt.savefig('fssum_dblgaussian_fit-' + str(T) +'K_'+str(Li_num)+'.eps')

    pfreqLi_fit1 = param[1] #MHz
    pfreqLi_fit1_err = error[1] #MHz
    K1 = (pfreqLi_fit1/gammaLi/B-1.0)*100 #%
    K1_err = (1.0+K1/100.0)*100.0*np.absolute(pfreqLi_fit1_err/pfreqLi_fit1)#%
    FWHM1_kHz = param[2]*1000 #kHz 
    FWHM1_kHz_err = error[2]*1000 #kHz

    pfreqLi_fit2 = param[4] #MHz
    pfreqLi_fit2_err = error[4] #MHz
    K2 = (pfreqLi_fit2/gammaLi/B-1.0)*100 #%
    K2_err = (1.0+K2/100.0)*100.0*np.absolute(pfreqLi_fit2_err/pfreqLi_fit2)#%
    FWHM2_kHz = param[5]*1000 #kHz 
    FWHM2_kHz_err = error[5]*1000 #kHz

    results = np.zeros(24)
    results[0] = pfreqLi_fit1
    results[1] = pfreqLi_fit1_err
    results[2] = K1
    results[3] = K1_err
    results[4] = param[0]
    results[5] = error[0]
    results[6] = FWHM1_kHz
    results[7] = FWHM1_kHz_err
    results[8] = 0.0
    results[9] = pfreqLi_fit2
    results[10] = pfreqLi_fit2_err
    results[11] = K2
    results[12] = K2_err
    results[13] = param[3]
    results[14] = error[3]
    results[15] = FWHM2_kHz
    results[16] = FWHM2_kHz_err
    results[17] = 0.0
    results[18] = error[0]/param[0]*100.0
    results[19] = error[1]/param[1]*100.0
    results[20] = error[2]/param[2]*100.0
    results[21] = error[3]/param[3]*100.0
    results[22] = error[4]/param[4]*100.0
    results[23] = error[5]/param[5]*100.0


    filename='fssum_dblgaussian_fit-'+str(T)+'K_'+str(Li_num)+'_result.csv'
    np.savetxt(filename, results)

    filename='fssum_'+str(T)+'K_dbl_gauss.csv'
    np.savetxt(filename, dblgaussian_fit(freq, *param))
    
    filename='fssum_'+str(T)+'K_gauss1.csv'
    np.savetxt(filename, gaussian_fit(freq, param[0], param[1], param[2]))

    filename='fssum_'+str(T)+'K_gauss2.csv'
    np.savetxt(filename, gaussian_fit(freq, param[3], param[4], param[5]))

if fssum_ended == 'yes' and gauss == 't':
    ###### fssum plot #####
    filename = fssum
    data = np.loadtxt(filename, skiprows=1, usecols=(0,4,5))
    freq = data[:,0]#MHz
    absV = data[:,1]#V
    darkV = data[:,2]#V

    plt.figure()
    plt.plot(freq, absV)
    plt.xlabel ('freq (MHz)')
    plt.ylabel ('spectrum (V)')
    plt.savefig('fssum' + str(T) +'K_'+str(Li_num)+'.eps')
    
    filename='fssum_absV'+str(T)+'K_'+str(Li_num)+'.csv'
    np.savetxt(filename, absV)

    filename='fssum_freq'+str(T)+'K_'+str(Li_num)+'.csv'#MHz
    np.savetxt(filename, freq)
    
    Ks = (freq/gammaLi/B-1.0)*100.0#%
    filename='fssum_Ks'+str(T)+'K_'+str(Li_num)+'.csv'
    np.savetxt(filename, Ks)

    ini=tplg_ini

    freq_lim = []
    absV_lim = []
    for i in range(len(freq)):
        if freq[i] > Lcut and freq[i] < Ucut:
            freq_lim.append(freq[i])
            absV_lim.append(absV[i])

    Weight = np.ones_like(freq_lim)
    for i in range(len(freq_lim)):
        if freq_lim[i] > ini[1]-ini[2]/2.0 and freq_lim[i] < ini[1]+ini[2]/2.0:
            Weight[i] = 10.0
        if freq_lim[i] > ini[4]-ini[5]/2.0 and freq_lim[i] < ini[4]+ini[5]/2.0:
            Weight[i] = 10.0
        if freq_lim[i] > ini[7]-ini[8]/2.0 and freq_lim[i] < ini[7]+ini[8]/2.0:
            Weight[i] = 10.0
            
    param, cov = curve_fit(tplgaussian_fit, freq_lim, absV_lim, p0=ini, sigma=1/Weight)
    error = np.sqrt(np.diag(cov))

    plt.figure()
    plt.plot (freq, absV)
    plt.plot (freq, tplgaussian_fit(freq, *param))
    plt.plot (freq, gaussian_fit(freq, param[0], param[1], param[2]))
    plt.plot (freq, gaussian_fit(freq, param[3], param[4], param[5]))
    plt.plot (freq, gaussian_fit(freq, param[6], param[7], param[8]))

    plt.xlabel ('freq (MHz)')
    plt.ylabel ('Intensity (V)')
    plt.savefig('fssum_tplgaussian_fit-' + str(T) +'K_'+str(Li_num)+'.eps')

    pfreqLi_fit1 = param[1] #MHz
    pfreqLi_fit1_err = error[1] #MHz
    K1 = (pfreqLi_fit1/gammaLi/B-1.0)*100 #%
    K1_err = (1.0+K1/100.0)*100.0*np.absolute(pfreqLi_fit1_err/pfreqLi_fit1)#%
    FWHM1_kHz = param[2]*1000.0 #kHz 
    FWHM1_kHz_err = error[2]*1000.0 #kHz


    pfreqLi_fit2 = param[4] #MHz
    pfreqLi_fit2_err = error[4] #MHz
    K2 = (pfreqLi_fit2/gammaLi/B-1.0)*100 #%
    K2_err = (1.0+K2/100.0)*100.0*np.absolute(pfreqLi_fit2_err/pfreqLi_fit2)#%
    FWHM2_kHz = param[5]*1000.0 #kHz 
    FWHM2_kHz_err = error[5]*1000.0 #kHz

    pfreqLi_fit3 = param[7] #MHz
    pfreqLi_fit3_err = error[7] #MHz
    K3 = (pfreqLi_fit3/gammaLi/B-1.0)*100 #%
    K3_err = (1.0+K3/100.0)*100.0*np.absolute(pfreqLi_fit3_err/pfreqLi_fit3)#%
    FWHM3_kHz = param[8]*1000.0 #kHz 
    FWHM3_kHz_err = error[8]*1000.0 #kHz

    results = np.zeros(36)
    results[0] = pfreqLi_fit1
    results[1] = pfreqLi_fit1_err
    results[2] = K1
    results[3] = K1_err
    results[4] = param[0]
    results[5] = error[0]
    results[6] = FWHM1_kHz
    results[7] = FWHM1_kHz_err
    results[8] = 0.0
    results[9] = pfreqLi_fit2
    results[10] = pfreqLi_fit2_err
    results[11] = K2
    results[12] = K2_err
    results[13] = param[3]
    results[14] = error[3]
    results[15] = FWHM2_kHz
    results[16] = FWHM2_kHz_err
    results[17] = 0.0
    results[18] = pfreqLi_fit3
    results[19] = pfreqLi_fit3_err
    results[20] = K3
    results[21] = K3_err
    results[22] = param[6]
    results[23] = error[6]
    results[24] = FWHM3_kHz
    results[25] = FWHM3_kHz_err
    results[26] = 0.0
    results[27] = error[0]/param[0]*100.0
    results[28] = error[1]/param[1]*100.0
    results[29] = error[2]/param[2]*100.0
    results[30] = error[3]/param[3]*100.0
    results[31] = error[4]/param[4]*100.0
    results[32] = error[5]/param[5]*100.0
    results[33] = error[6]/param[6]*100.0
    results[34] = error[7]/param[7]*100.0
    results[35] = error[8]/param[8]*100.0

    filename='fssum_tplgaussian_fit-'+str(T)+'K_'+str(Li_num)+'_result.csv'
    np.savetxt(filename, results)

    filename='fssum_'+str(T)+'K_tpl_gauss.csv'
    np.savetxt(filename, tplgaussian_fit(freq, *param))
    
    filename='fssum_'+str(T)+'K_gauss1.csv'
    np.savetxt(filename, gaussian_fit(freq, param[0], param[1], param[2]))

    filename='fssum_'+str(T)+'K_gauss2.csv'
    np.savetxt(filename, gaussian_fit(freq, param[3], param[4], param[5]))

    filename='fssum_'+str(T)+'K_gauss3.csv'
    np.savetxt(filename, gaussian_fit(freq, param[6], param[7], param[8]))

