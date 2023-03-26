import strax
import numpy as np
import numba
import straxen
from numba import njit

export, __all__ = strax.exporter()

# A.P. Colijn -  Feb. 2021
# Parameters for the fit
lpf_iter_max = 100
lpf_min_position_cor = 0.01 # convergence criterium for position minimizer
lpf_npar = 4 # number of fit parameters
#                       x      y      r0     gamma
lpf_fix_par = np.array([False, False, False, False]) # which parameters to fix

@export
class PeakPositionsLPF(strax.Plugin):
    """
    LPF = Linearized Position Fit
    Computes S2 position with linearized log-likelihood fit. 
    Returns xy position and the parameters of the fit.
    See LinPosFit https://github.com/XAMS-nikhef/LinPosFit/ 
    from A.P. Colijn (Feb 2021)
    """
    depends_on =('peaks','peak_basics')
    provides = 'peak_positions_lpf'
    rechunk_on_save = False

    n_top_pmts = straxen.URLConfig(default=straxen.n_top_pmts, infer_type=False,
                                   help="Number of top PMTs")

    pmt_to_lxe = straxen.URLConfig(default=7, infer_type=False,
                                   help="Distance between the PMTs and the liquid interface")
    
    def infer_dtype(self):

        dtype = [('x_lpf',np.float32, 'Reconstructed x (cm) position, LinPosFit'),
        ('y_lpf',np.float32, 'Reconstructed y (cm) position, LinPosFit'),
        ('lpf_logl',np.float32, 'LinPosFit LogLikelihood value'), 
        ('lpf_r0',np.float32, 'LinPosFit r0 parameter, reduced rate'), 
        ('lpf_gamma', np.float32, 'LinPosFit gamma parameter, reflection'),
        ('lpf_n',np.float32, 'LinPosFit parameter n, number of photons generated'),
        ('lpf_err_x',np.float32, 'LinPosFit error on x (cm), 95p conficence interval'),
        ('lpf_err_y',np.float32, 'LinPosFit error on y (cm), 95p conficence interval')
        ]

        dtype += strax.time_fields
        return dtype

    def setup(self,):
        inch = 2.54 # cm
        pmt_pos = straxen.pmt_positions()
        self.pmt_pos = list(zip(pmt_pos['x'].values,pmt_pos['y'].values,np.repeat(self.pmt_to_lxe, self.n_top_pmts)))
        self.pmt_surface=(3*inch)**2*np.pi/4.

    def compute(self,peaks):
        
        result = np.ones(len(peaks), dtype=self.dtype)

        result['time'], result['endtime'] = peaks['time'], strax.endtime(peaks)
        result['x_lpf'] *= float('nan')
        result['y_lpf'] *= float('nan')
        result['lpf_r0'] *= float('nan')
        result['lpf_logl'] *= float('nan')
        result['lpf_n'] *= float('nan')
        result['lpf_gamma'] *= float('nan')
        result['lpf_err_x'] *= float('nan')
        result['lpf_err_y'] *= float('nan')

        for ix, p in enumerate(peaks):
            if p['type']==2:
                # Only reconstruct s2 peaks. We do need to set the time of the peaks
                try:
                    # Execute linearised position reconstruction fit
                    fit_result, xiter, user_hits = lpf_execute(self.pmt_pos[:self.n_top_pmts],
                                                    p['area_per_channel'][:self.n_top_pmts],
                                                    self.pmt_surface)

                    # Error estimation function
                    err_x, err_y = lpf_deriv_95(self.pmt_pos[:self.n_top_pmts],
                                            p['area_per_channel'][:self.n_top_pmts],
                                            fit_result,
                                            used_hits)

                    result[ix]['x_lpf'] = fit_result[0]
                    result[ix]['y_lpf'] = fit_result[1]
                    result[ix]['lpf_r0'] = fit_result[2]
                    result[ix]['lpf_logl'] = fit_result[3]
                    result[ix]['lpf_n'] = fit_result[4]
                    result[ix]['lpf_gamma'] = fit_result[5]
                    result[ix]['lpf_err_x'] = err_x
                    result[ix]['lpf_err_y'] = err_y
                    
                except Exception as e:
                    # Sometimes inverting the matrix fails.. 
                    # Let's just leave it as a NaN
                    pass

        return result



def lpf_execute(xhit, nhit, area_sensor):
    #%snakeviz
    """
    Execute the fitter
    :param xhit:
    :param nhit:
    :param area_sensor:
    :param kwargs:
    :return:
    """
    xhit = np.array(xhit)
    nhit = np.array(nhit)
    # minimize the likelihood
    fit_result, xiter, hit_is_used = lpf_minimize(xhit, nhit, area_sensor)
    return fit_result, xiter, hit_is_used

@njit
def lpf_factorial(x):
    n = 1
    for i in range(2, x+1):
        n *= i
    return n

@njit
def lpf_lnlike(xhit, nhit, xf, nuv, gamma, area_sensor):
    r0 = nuv * area_sensor * xhit[0][2] / 4 / np.pi
    logl = 0

    # print('next')
    for ih in range(len(nhit)):
        nexp = lpf_nexp(xhit[ih], xf, r0, gamma)
        # print(xhit[ih], nexp, xf, r0)

        logl = logl + nexp - nhit[ih] * np.log(nexp)

    return logl

@njit
def lpf_lnlike_plot(xhit, nhit, hit_is_used, xf, r0, gamma):
    logl = 0

    for ih in range(len(nhit)):
        nexp = lpf_nexp(xhit[ih], xf, r0, gamma)
        d = lpf_dist(xhit[ih],xf)
        if hit_is_used[ih] == 1:
            logl = logl + nexp - nhit[ih] * np.log(nexp)

    return logl

@njit
def lpf_lnlike_r(xhit, nhit, xf, r0, gamma):
    logl = 0

    for ih in range(len(nhit)):
        nexp = lpf_nexp(xhit[ih], xf, r0, gamma)
        logl = logl + nexp - nhit[ih] * np.log(nexp)
 
    return logl

@njit
def lpf_initialize(xhit, nhit):
    """
    Initialize the lpf_fitter. Estimate the initial position and uv photons

    :param xhit:
    :param nhit:
    :return:
    """
    nmax = -1

    xhit_max = np.zeros(3)
    # sort hits in descending order. helps when looping over the hits
    #
    indices = np.argsort(-nhit) # descending order
    
    # find a cluster of hits
    #
    # 1. we start with the highest hit in the event
    # 2. if it is surrrounded by other hits then we found a cluster and we are done
    # 3. if not: continue at 1. with the second highest hit in the event
    # 4. repeat until we checked all possibilities
    # 5. if no cluster was found we use the highest hit as the initial fit position
    #
    
    # only consider hits within distance dist_min
    #
    dist_min = 25.
    #  - maximum fraction of the cluster signal allowed to be inside the leading hit
    #  - below this fraction we assume that we are dealing with a hit cluster
    hit_frac_cut = 0.75
    found_cluster = False
    
    # loop starting from the highest hit
    #
    xcluster = np.zeros(3)
    for i in range(len(indices)-1):
        index = indices[i]
        xhit_max = xhit[index]
        # the seed hit has signal xmax
        #
        nmax = nhit[index]
        
        # we have reached the zero-signal PMTs..... no cluster was found, so leave this loop
        #
        if nmax == 0:
            break
        
        # n_around indicates the amount of signal that is found around the seed hit
        #
        n_around = 0
        
        # check the other hits....
        #
        xcluster = nhit[index]*xhit[index]
        for j in range(i+1, len(indices)):
            jindex = indices[j]
            xtest = xhit[jindex]
            dist = lpf_dist(xhit_max, xtest)
            # consider the hit if it is near the seed hit
            #
            if dist<dist_min:
                n_around = n_around + nhit[jindex]
                xcluster = xcluster + nhit[jindex]*xhit[jindex]
        # calculate the fraction of the cluster signal in the seed hit
        #
        hit_frac = 1.*nmax/(nmax+n_around)
        # if the fraction is below hit_frac_cut we are dealing with a cluster 
        # 
        if hit_frac<hit_frac_cut:
            found_cluster = True
            break
            
    # no cluster was found... best guess is to use the hit with the maximum signal as initial position
    #
    if not found_cluster: # use maximum hit
        nmax = nhit[indices[0]]
        xhit_max = xhit[indices[0]]
    else: # we calculate the center-of-gravity as initial position
        xhit_max = xcluster / (nmax+n_around)
    

    # refine the determination of the initial position
    # logl_min = 1e12
    # xtemp = np.zeros(3)
    xfit = np.zeros(3)
    xfit[0] = xhit_max[0]
    xfit[1] = xhit_max[1]
    xfit[2] = 0.
    # estimate the fitted rate
    #
    r0 = nmax * xhit_max[2] ** 3

    return xfit, r0

@njit
def lpf_hit_prob(xi, ni, xf, r0, gamma):
    nexp = lpf_nexp(xi, xf, r0, gamma)
    if ni<10: # explicit calculation of ln(n!)
        nfac = lpf_factorial(ni)
        log_nfac = np.log(nfac)
    else: # use Stirlings formula to approximate ln(n!)
        log_nfac = ni*np.log(ni) - ni
    log_prob = ni*np.log(nexp) - nexp - log_nfac
    return log_prob

@njit
def lpf_minimize(xhit, nhit, area_sensor):
    """
    Linearized -log(L) fit for S2 position

    :param xhit: positions of the sensors (list with 3D arrays with length n-sensor)
    :param nhit: number of p.e. on each sensor (length n-sensor)
    :param area_sensor: area of teh sensor
    :return: fit_result: [0] = x, [1] = y, [2] = nuv [3] = lnlike [4] = n_in_fit [5] = gamma
    :return: xiter: intermediate fit results
             nuv: number of emitted photons
    """
    # initialize
    #
    xfit, r0 = lpf_initialize(xhit, nhit)
    gamma = 0.0

    # arrays to store teh fit resuults for each iteration
    xiter = np.zeros((lpf_iter_max + 1, 6))
    xstart = np.array([xfit[0],xfit[1],0])
    xiter[0][0] = xfit[0]
    xiter[0][1] = xfit[1]
    xiter[0][2] = r0
    xiter[0][3] = lpf_lnlike_r(xhit, nhit, xfit, r0, gamma)
    xiter[0][4] = 0
    xiter[0][5] = 0
    
    r_max = 10.
    # iterate & minimize
    #
    hit_is_used = np.zeros(len(nhit))
    n_in_fit = 0
    # log_pmin: minimal probability for a hit to be used in teh position fit
    #
    log_pmin = np.log(1e-10)
    nmax = np.amax(nhit)
    for lpf_iter in range(lpf_iter_max):
        # initialize error matrix and vector
        #
        g = np.zeros(lpf_npar)
        m = np.zeros((lpf_npar, lpf_npar))

        # calculate the sums
        #
        # print(lpf_iter,' xin =',xfit,' r0=',r0)
        n_in_fit = 0 # reset after each iteration
        
        if lpf_iter < 2:
            lpf_fix_par = np.array([False, False, False, False]) # which parameters to fix
        else:
            lpf_fix_par = np.array([False, False, False, False]) # which parameters to fix
        # make a list of the active parameters
        active_parameters = np.arange(lpf_npar)[lpf_fix_par == False]

        
        for isensor in range(len(nhit)):
            if lpf_iter==0:
                hit_is_used[isensor] = 0
            # calculate the probability that a hit comes from the assumed xhit and r0
            #
            log_prob = lpf_hit_prob(xhit[isensor], nhit[isensor], xfit, r0, gamma)
            # print(isensor,' xhit =',xhit[isensor],' logP =',np.exp(log_prob),' ni =', nhit[isensor])
            #if (log_prob > log_pmin) and (nhit[isensor]>0):
            if ((lpf_iter == 0)  and (nhit[isensor] > nmax*0) and (lpf_dist(xhit[isensor],xfit) < 150.)) or (hit_is_used[isensor] == 1):
                n_in_fit = n_in_fit+1
                if lpf_iter == 0:
                    hit_is_used[isensor] = 1
                for i in active_parameters: # only fill the error matrix for parameters you want to fit 
                    g[i] = g[i] + lpf_f(i, xhit[isensor], nhit[isensor], xfit, r0, gamma)
                    for j in active_parameters:
                        m[i][j] = m[i][j] + lpf_deriv_f(i, j, xhit[isensor], nhit[isensor], xfit, r0, gamma)

        for i in np.arange(lpf_npar)[lpf_fix_par == True]:
            m[i][i] = 1

        # invert the matrix
        #
        if np.linalg.det(m) == 0.:
            # print('lpf_minimize:: singular error matrix')
            break
        
        minv = np.linalg.inv(m)
        # multiply with vector to get corrections to the current fit parameters
        #
        result = np.dot(minv, g)

        # if abs(result[1])>1000:
        #    print('WARNING:: result = ',result)
        r_new = np.sqrt((xstart[0]-(xfit[0] - result[0]))**2 + (xstart[1] - (xfit[1] - result[1]))**2)
        
        weight = 0.5*(1 + np.math.erf((r_max-r_new)/1.0))
        # weight = 1
        # update fit result
        #if weight<1e-3:
        #    print('ping ping ping.... weight =',weight)
        ###weight = 1.0
        xfit[0] = xfit[0] - result[0]*weight
        xfit[1] = xfit[1] - result[1]*weight
        
        #weight = np.exp(-lpf_dist(xstart,xfit)/25.)
        r0 = r0 - result[2]*weight
            
        gamma = gamma - result[3]*weight

        # and store the intermediate results
        #
        xiter[lpf_iter + 1][0] = xfit[0]
        xiter[lpf_iter + 1][1] = xfit[1]
        xiter[lpf_iter + 1][2] = r0
        xiter[lpf_iter + 1][3] = lpf_lnlike_r(xhit, nhit, xfit, r0, gamma)
        xiter[lpf_iter + 1][4] = n_in_fit
        xiter[lpf_iter + 1][5] = gamma


        # if (lpf_iter>=5) and (abs(result[0]) < 0.01) and (abs(result[1]) < 0.01) or (r0<0):  # if position no longer changes -> terminate loop
        if ((lpf_iter>1) and abs(result[0]) < lpf_min_position_cor) and (abs(result[1]) < lpf_min_position_cor) or (r0<0):  # if position no longer changes -> terminate loop
            break

    # calculate the number of uv photons
    #
    nuv = 4 * np.pi * r0 / area_sensor / xhit[0][2]

    # store the fit results
    #
    fit_result = np.zeros(7)
    fit_result[0] = xfit[0]
    fit_result[1] = xfit[1]
    fit_result[2] = nuv
    
    logl = 0
    for ih in range(len(nhit)):
        if nhit[ih] > 0:
            logprob = lpf_hit_prob(xhit[ih], nhit[ih], xfit, r0, gamma)
            if logprob>log_pmin:
                logl = logl + lpf_hit_prob(xhit[ih], nhit[ih], xfit, r0, gamma)

    fit_result[3] = logl # lpf_lnlike_r(xhit, nhit, xfit, r0)
    fit_result[4] = n_in_fit
    fit_result[5] = gamma
    fit_result[6] = r0


    return fit_result, xiter, hit_is_used


@njit
def lpf_dist(x0, x1):
    d2 = (x0[0] - x1[0]) ** 2 + (x0[1] - x1[1]) ** 2 + (x0[2] - x1[2]) ** 2
    return np.sqrt(d2)


@njit
def lpf_nexp(xi, xf, r0, gamma):
    """
    Calculate the expected number of p.e. for a lightsensor
    
    :param xi: sensor position
    :param xf: assumed hit position
    :param r0: assumed nuumber of UV photons (re-normalized)
    :param gamma: assumed constant background
    :return nexp: number of expected photons
    
    A.P. Colijn
    """
    delta = lpf_dist(xi, xf)
    nexp = r0 / delta ** 3 + gamma

    return nexp

@njit
def lpf_f(i, xi, ni, xf, r0, gamma):
    """
    Calculate the minimizer functions
    :param i:
    0=F0 (x)
    1=F1 (y)
    2=F2 (r0)
    3=F3 (gamma) 
    :param xi: sensor position
    :param ni: hits for the sensor
    :param xf: assumed fit position
    :param r0: assumed number of UV photons (re-normalized)
    :param gamma: assumed constant background
    :return f: function value
    """

    f = 0
    if i < 2:
        f = -3 * (xf[i] - xi[i]) * (lpf_nexp(xi, xf, r0, gamma) - ni) / lpf_dist(xi, xf) ** 2
    elif i == 2:
        f = (lpf_nexp(xi, xf, r0, gamma) - ni) / r0
    elif i == 3:
        f = 1. - ni / lpf_nexp(xi, xf, r0, gamma)

    return f

@njit
def lpf_deriv_f(i, j, xi, ni, xf, r0, gamma):
    """
    Derivatives of the minimizer functions

    :param i:
    0=F0
    1=F1
    2=F2
    :param j:
    0=x
    1=y
    z=r0
    :param xi: hit position
    :param ni: number of hits
    :param xf: fit position
    :param r0: number of photons
    :return:
    """

    d = lpf_dist(xi, xf)
    n0 = lpf_nexp(xi, xf, r0, gamma)

    deriv = 0
    if i == 0:
        dx = xf[0] - xi[0]
        if j == 0:  # dF0/dx
            deriv = -3 * (n0 - ni) / d ** 2
            deriv = deriv - 3 * dx * (n0 - ni) * lpf_deriv_dist_min2(0, xi, xf)
            deriv = deriv - 3 * dx * lpf_deriv_n(0, xi, xf, r0, gamma) / d ** 2
        elif j == 1:  # dF0/dy
            deriv = - 3 * dx * (n0 - ni) * lpf_deriv_dist_min2(1, xi, xf)
            deriv = deriv - 3 * dx * lpf_deriv_n(1, xi, xf, r0, gamma) / d ** 2
        elif j == 2:  # dF0/dr0
            deriv = -3 * dx * lpf_deriv_n(2, xi, xf, r0, gamma) / d ** 2
        elif j == 3: #dF0/dgamma
            deriv = -3 * dx * lpf_deriv_n(3, xi, xf, r0, gamma) / d ** 2        
    elif i == 1:
        dy = xf[1] - xi[1]
        if j == 0:  # dF1/dx
            deriv = - 3 * dy * (n0 - ni) * lpf_deriv_dist_min2(0, xi, xf)
            deriv = deriv - 3 * dy * lpf_deriv_n(0, xi, xf, r0, gamma) / d ** 2
        elif j == 1:  # dF1/dy
            deriv = -3 * (n0 - ni) / d ** 2
            deriv = deriv - 3 * dy * (n0 - ni) * lpf_deriv_dist_min2(1, xi, xf)
            deriv = deriv - 3 * dy * lpf_deriv_n(1, xi, xf, r0, gamma) / d ** 2
        elif j == 2:  # dF1/dr0
            deriv = -3 * dy * lpf_deriv_n(2, xi, xf, r0, gamma) / d ** 2
        elif j == 3: #dF0/dgamma
            deriv = -3 * dy * lpf_deriv_n(3, xi, xf, r0, gamma) / d ** 2 
    elif i == 2:
        #if j == 0:
        #    deriv = lpf_deriv_n(0, xi, xf, r0)
        #elif j == 1:
        #    deriv = lpf_deriv_n(1, xi, xf, r0)
        #elif j == 2:
        #    deriv = lpf_deriv_n(2, xi, xf, r0)
        if j == 0:
            deriv = lpf_deriv_n(0, xi, xf, r0, gamma) / r0
        elif j == 1:
            deriv = lpf_deriv_n(1, xi, xf, r0, gamma) / r0
        elif j == 2:
            deriv = lpf_deriv_n(2, xi, xf, r0, gamma) / r0 - (n0 - ni) / r0 ** 2
        elif j == 3:
            deriv = lpf_deriv_n(3, xi, xf, r0, gamma) / r0
    elif i == 3:
        deriv = ni * lpf_deriv_n(j, xi, xf, r0, gamma) / n0 ** 2
    else:
        deriv = 0.
        

    return deriv


@njit
def lpf_deriv_n(i, xi, xf, r0, gamma):
    """
    Derivative of n wrt to fit parameters

    :param i:
    0=x
    1=y
    2=r0
    3=gamma
    :param xi: hit position
    :param xf: fit position
    :param r0: number of photons
    :param gamma:  constant background term
    :return: dn/di
    """

    if i < 2:
        deriv = -3 * lpf_nexp(xi, xf, r0, gamma) * (xf[i] - xi[i]) / lpf_dist(xi, xf) ** 2
    elif i == 2:
        deriv = lpf_nexp(xi, xf, r0, gamma) / r0
    elif i == 3:
        deriv = 1.
    else:
        deriv = 0.

    return deriv


@njit
def lpf_deriv_dist(i, xi, xf):
    """
    Derivative of distance wrt fit parameters

    :param i: 0=x
              1=y
              2=r0
    :param xi: hit position
    :param xf: fit position
    :return: dDist/di
    """

    if i < 2:
        deriv = (xf[i] - xi[i]) / lpf_dist(xi, xf)
    else:
        deriv = 0.0

    return deriv


@njit
def lpf_deriv_dist_min2(i, xi, xf):
    """
    Derivative of 1/dist**2

     :param i: 0=x
               1=y
               2=r0
    :param xi: hit position
    :param xf: fit position
    :return: d(1/Dist**2)/di
    """

    deriv = 0.0
    if i < 2:
        d = lpf_dist(xi, xf)
        deriv = -(2 / d ** 3) * lpf_deriv_dist(i, xi, xf)

    return deriv

@njit
def lpf_deriv_95(xhit, nhit, fit_result, hit_is_used, **kwargs):
    """
    Error estimation for LinPosFit
    Bachelor thesis: Ezra de Cleen, Nikhef, June 2022
    """

    deriv_x = 0
    deriv_y = 0
    
    for isensor in range(len(nhit)):

        xff = np.array([fit_result[0],fit_result[1],0])
        
        deriv_x += lpf_deriv_f(0, 0, xhit[isensor], nhit[isensor], xff, fit_result[2], fit_result[5])

        xff = np.array([fit_result[0],fit_result[1],0])

        deriv_y += lpf_deriv_f(1, 1, xhit[isensor], nhit[isensor], xff, fit_result[2], fit_result[5])

    x_95 = 1*(1/np.sqrt(deriv_x))
    y_95 = 1*(1/np.sqrt(deriv_y))

        
    return x_95, y_95