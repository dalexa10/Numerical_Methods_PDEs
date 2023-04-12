__author__ = 'Dario Rodriguez'

# ----------------------------------------------------------------
#                       PROBLEM DESCRIPTION
# ----------------------------------------------------------------
# Solve:
#               u_t + (u^2 / 2)_x = 0
#
#               on [-1, 6] in space domain and
#               on [0, 2] time domain
#
# with periodic boundary conditions.
# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def fun_rarefaction(x):
    """ Defines the initial speed state of a rarefaction case """
    u = np.zeros(x.shape)
    for j in range(len(x)):
        if x[j] < 0:
            u[j] = 1.
        elif 0 <= x[j] <= 1:
            u[j] = 1 + x[j]
        elif x[j] > 1:
            u[j] = 2.
    return u

def fun_shock(x):
    """ Defines the initial speed state of a shock case """
    u = np.zeros(x.shape)
    for j in range(len(x)):
        if x[j] < 0:
            u[j] = 2.
        elif 0 <= x[j] <= 1:
            u[j] = 2 - x[j]
        elif x[j] > 1:
            u[j] = 1.
    return u

def f(u):
    """ Defines f(u) in Burger's equation """
    return u**2/2

def fprime(u):
    """ Derivative of f(u) in Burger's equation """
    return u

def minmod(r):
    phi = np.array([np.maximum(0., np.minimum(r_i, 1.)) for r_i in r])
    return phi

def exact_position_rarefaction(x0, t, u0):
    """ Computes the final x position (xf) given an initial position (x0)
    and a speed profile u0 after time t has passed
    Note: this works specifically for the simple case of u0 shown in this
    script
    """
    xf = x0 + u0 * t
    return xf

def exact_position_shock(x0, t, u0):
    """ Computes the final x position (xf) given an initial position (x0)
    and a speed profile u0 after time t has passed
    Note: this works specifically for the simple case of u0 shown in this
    script
    """
    if t >= 1:
        x_sh = 1.5 * (t - 1) + 2  # 1.5 is the shock speed - Computed with RH relations (VALID for this specific example!)
        idx_0 = np.where(u0 == 2)[0][-1]
        idx_f = np.where(u0 == 1)[0][0]
        xf = np.empty(x0.shape[0], dtype=float)
        xf[:idx_0] = np.linspace(-1, x_sh, idx_0)
        xf[idx_0: idx_f] = x_sh
        xf[idx_f:] = np.linspace(x_sh, 6, u0.shape[0] - idx_f)
    else:
        xf = x0 + u0 * t
    return xf

def filter_u_data(u, BC_l, BC_r, eps=1e-3):
    """ This function filters the u vector and get the position (x)
     of the filtered u vector """
    lb = np.minimum(BC_l, BC_r)
    ub = np.maximum(BC_l, BC_r)
    idx = np.where((lb < u - eps) & (u + eps < ub))
    x_filt = x[idx]
    u_filt = u[idx]
    return x_filt, u_filt


def compute_exact_position(x_filt, case, T):
    try:
        if case == 'Rarefaction':
            if T == 1:
                x_exact = np.linspace(1, 3, x_filt.shape[0])
            elif T == 2:
                x_exact = np.linspace(2, 5, x_filt.shape[0])
            return x_exact

        elif case == 'Shock':
            if T == 1:
                x_exact = 2 * np.ones(x_filt.shape[0])
            elif T == 2:
                x_exact = 3.5 * np.ones(x_filt.shape[0])
            return x_exact

    except ValueError:
        print('Case not found or not correct time provided')
        return None


def compute_error(u_num, case, T, hx, BC):
    """ Compute the L2 norm of the error """
    x_filt, u_filt = filter_u_data(u_num, BC[0], BC[1])
    x_exact = compute_exact_position(x_filt, case, T)
    error = x_filt - x_exact
    l2err = np.sqrt(hx * np.sum(error**2))
    return x_filt, u_filt, x_exact, l2err


if __name__ == '__main__':
    import time
    import pickle
    import os

    # --------------------------------------
    #               Instructions
    # --------------------------------------
    # Comment / Uncomment variables in User Inputs section and hit run
    # If you want to save data for visualization, change boolean value of 'write_results' variable to True
    # This will save a dictionary  that contains almost all results from a single
    # simulation as .pkl file. You can then use a visualization tool to see results
    #
    # Warning: DO NOT CHANGE the code if you are unsure of what it does

    # -------------------------------------
    #               User Inputs
    # --------------------------------------
    # -----  Time (period) ----
    T = 1.0
    # T = 2.0

    # ------  Method --------
    # method = 'LF'
    # method = 'Gudonov'
    method = 'RK2'

    # ----- Boundary Conditions -----
    case = 'Rarefaction'
    # case = 'Shock'

    # ---- Space discretization -----
    nx = 128
    # nx = 1000

    # ------- Save results? -------
    write_results = False

    # ----------------------------------------
    #               Preprocessing
    # ---------------------------------------

    start_time = time.time()
    x, hx = np.linspace(-1, 6, nx, endpoint=False, retstep=True)  # Domain in space

    try:
        if case == 'Rarefaction':
            u = fun_rarefaction(x)
            u0 = u.copy()
            xf = exact_position_rarefaction(x, t=T, u0=u0)

        elif case == 'Shock':
            u = fun_shock(x)
            u0 = u.copy()
            xf = exact_position_shock(x, t=T, u0=u0)

    except ValueError:
        print('Wrong case given')


    ht = hx / (2 * np.max(np.abs(fprime(u0))))   # ht computed to ensure estable simulation
    nt = int(np.ceil(T/ht))

    K = np.arange(0, nx)    # 0, ..., nx-1
    Km1 = np.roll(K, 1)     # nx-1, 0, 1, ..., nx-2
    Kp1 = np.roll(K, -1)    # 1, ..., nx
    Kp2 = np.roll(K, -2)    # 2, ..., nx + 1



    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(x, u0, 'r-', linewidth=1)
    ax.plot(xf, u0, 'g-', lw=1)
    uline, = ax.plot(x, u, '-', linewidth=3)
    txt = ax.text(1.0 / 3.0, 0.8, 't=%g, i=%g' % (0.0, 0), fontsize=12)


    print('T = %g' % T)
    print('tsteps = %d' % nt)
    print('    hx = %g' % hx)
    print('    ht = %g' % ht)

    # ----------------------------------------
    #       Processing and live plotting
    # ---------------------------------------

    for n in range(1, nt+1):

        if method == 'LF':
            uplus = u[Kp1]
            uminus = u[K]

            alpha = np.maximum(np.abs(fprime(uminus)), np.abs(fprime(uplus)))
            flux = (f(uminus) + f(uplus)) / 2 - (alpha / 2) * (uplus - uminus)

            u[:] = u - ht / hx * (flux[K] - flux[Km1])

            if case == 'Rarefaction':
                u[0] = 1
                u[-1] = 2
            elif case == 'Shock':
                u[0] = 2
                u[-1] = 1
            
        elif method == 'Gudonov':
            uplus = u[Kp1]
            uminus = u[K]

            flux = np.empty(uminus.shape[0], dtype=float)
            rar_ls = np.where(uminus <= uplus)
            flux[rar_ls] = np.minimum(f(uplus[rar_ls]), f(uminus[rar_ls]))
            shock_ls = np.where(uminus > uplus)
            flux[shock_ls] = np.maximum(f(uplus[shock_ls]), f(uminus[shock_ls]))

            u[:] = u - ht / hx * (flux[K] - flux[Km1])

            if case == 'Rarefaction':
                u[0] = 1
                u[-1] = 2
            elif case == 'Shock':
                u[0] = 2
                u[-1] = 1

        elif method == 'RK2':
            # -----------------------------------
            # First Stage RK-2
            # -----------------------------------
            # Mimmod factor
            u_km1 = u[Km1]
            u_kp1 = u[Kp1]
            u_kp2 = u[Kp2]

            r_k = (u[K] - u_km1) / (u_kp1 - u[K] + 1e-15)
            phi_k = minmod(r_k)

            # U reconstruction in space with minmod limiter
            uminus = u[K] + 0.5 * phi_k[K] * (u_kp1 - u[K])
            uplus = u_kp1 - 0.5 * phi_k[Kp1] * (u_kp2 - u_kp1)

            # U reconstruction with RK 2 order and LLF
            alpha = np.maximum(np.abs(fprime(uminus)), np.abs(fprime(uplus)))
            flux = (f(uminus) + f(uplus)) / 2 - (alpha / 2) * (uplus - uminus)
            u_half = u - (ht / 2) * ((flux[K] - flux[Km1]) / hx)

            # --------------------------------------------
            # Second stage RK-2
            # -------------------------------------------
            if case == 'Rarefaction':
                u_half[0] = 1.
            elif case == 'Shock':
                u_half[0] = 2.

            u_km1 = u_half[Km1]
            u_kp1 = u_half[Kp1]
            u_kp2 = u_half[Kp2]

            # U reconstruction in space with minmod limiter
            r_k = (u_half[K] - u_km1) / (u_kp1 - u_half[K] + 1e-15)
            phi_k = minmod(r_k)

            # U reconstruction with RK 2 order and LLF
            uminus = u_half[K] + 0.5 * phi_k[K] * (u_kp1 - u_half[K])
            uplus = u_kp1 - 0.5 * phi_k[Kp1] * (u_kp2 - u_kp1)
            alpha = np.maximum(np.abs(fprime(uminus)), np.abs(fprime(uplus)))
            flux = (f(uminus) + f(uplus)) / 2 - (alpha / 2) * (uplus - uminus)
            u[:] = u - ht * ((flux[K] - flux[Km1]) / hx)

            if case == 'Rarefaction':
                u[0] = 1
                u[-1] = 2
                # u[-2] = 2

            elif case == 'Shock':
                u[0] = 2
                u[-1] = 1
                # u[-2] = 1


        uline.set_ydata(u)
        txt.set_text('t=%g, i=%g' % (n * ht, n))
        ax.axis([x.min(), x.max(), 0.9, 2.1])
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.show(block=True)

    x_filt, u_filt, x_exact, err = compute_error(u, case, T, hx, [u[0], u[-1]])

    comp_time = time.time() - start_time



    if write_results:

        results = {'u': u,
                   'x': x,
                   'u0': u0,
                   'xf': xf,
                   'x_filt': x_filt,
                   'u_filt': u_filt,
                   'x_exact': x_exact,
                   'error': err,
                   'case': case,
                   'hx': hx,
                   'nx': nx,
                   'ht': ht,
                   'nt': nt,
                   'T': T,
                   'comp_time': comp_time,
                   'method': method}

        file_name = case + '_' + method + '_' + 'T' + str(int(T)) + '_' + 'nx' + str(int(nx)) + '.pkl'

        if os.path.exists(os.getcwd() + '/results'):
            print('Directory already exits')
            pass
        else:
            os.mkdir(os.getcwd() + '/results')
            print("New directory 'results' created in current path")


        print('Writing your results ...')
        with open('results/' + file_name, 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

        print('Done')


    print('Error computed at time t = {:.0f} is {:.4f}'.format(T, err))
    print('Simulation time {:.3f} seconds'.format(comp_time))
