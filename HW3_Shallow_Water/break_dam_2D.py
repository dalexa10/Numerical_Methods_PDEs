__author__ = 'Dario Rodriguez'

# ----------------------------------------------------------------
#                     2D DAM-BREAKING PROBLEM
# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def speed2mom(u, h):
    """ Compute momentum from given speed (u or v) and height """
    return u * h

def mom2speed(m, h):
    """ Compute speed from given momentum and height """
    return m / h

def flux_fun(vec, g=1):
    """
    Flux vector function
    :param vec: (np.array 3 x nx)
    :return: f(u) (np.array 3 x nx)

     h: water height
     mx: x-momentum
     my: y-momentum
     g: (float) gravity. Default value 1

    """
    h, mx, my = vec[0, :], vec[1, :], vec[2, :]
    f_u = np.array([mx,
                    (mx**2 / h) + 0.5 * g * h**2,
                    mx * my / h])

    return f_u

def abs_lambda_max(u, h, g=1):
    """ Compute the absolute value of the greatest eigenvalue of the
    Jacobian matrix of the hyperbolic system"""
    alamb = np.amax(np.abs(u + np.sqrt(g * h)))
    return alamb

def initial_fun(x):
    """ Sets the initial value of u as function of x """
    u = np.zeros([3, x.shape[0]])
    for j in range(len(x)):
        if x[j] < 0:
            u[:, j] = [2, 0, 0]
        else:
            u[:, j] = [1, 0, 0]
    return u



if __name__ == '__main__':

    from itertools import cycle

    # -------------------------------------
    #               User Inputs
    # --------------------------------------
    nx = 100    # Space discretization points
    c = 0.5     # Constant to determine time discretization
    T = 0.5     # Total simulation time
    T_store = [0.25, 0.5]

    # -------------------------------------
    #               Processing
    # --------------------------------------
    # Note that the time step is chosen dynamically and the first
    # discretization is computed based on the initial values

    x, hx = np.linspace(-1, 1, nx, endpoint=False, retstep=True)
    u_vec = initial_fun(x)                        # Vector of conserved variables
    u_v_0 = u_vec.copy()                          # Vector of conserved variables at time 0
    u0 = mom2speed(u_vec[1, :], u_vec[0, :])      # u speed vector
    l_max = abs_lambda_max(u0, u_vec[0, :])       # lambda maximum (initial value)
    l_max_0 = l_max.copy()
    ht = c * np.min(hx / l_max_0)                 # time step

    K = np.arange(0, nx)    # 0, ..., nx-1
    Km1 = np.roll(K, 1)     # nx-1, 0, 1, ..., nx-2
    Kp1 = np.roll(K, -1)    # 1, ..., nx

    u_vec_st = {}           # Desired stored values

    # -------------------------------------
    #               Computing
    # --------------------------------------
    t_c = 0.
    st_idx = 0
    T_store_unique = []

    while t_c < T:
        u_v_plus = u_vec[:, Kp1]
        u_v_minus = u_vec[:, K]

        l_max_plus = abs_lambda_max(mom2speed(u_v_plus[1, :], u_v_plus[0, :]), u_v_plus[0, :])
        l_max_minus = abs_lambda_max(mom2speed(u_v_minus[1, :], u_v_minus[0, :]), u_v_minus[0, :])

        l_max_bar = np.maximum(l_max_plus, l_max_minus)
        flux = 0.5 * (flux_fun(u_v_minus) + flux_fun(u_v_plus)) - 0.5 * l_max_bar * (u_v_plus - u_v_minus)

        # Vector v computation
        u_vec = u_vec - (ht / hx) * (flux[:, K] - flux[:, Km1])   # u vector

        # Boundary conditions
        u_vec[:, 0] = [2, 0, 0]
        u_vec[:, -1] = [1, 0, 0]

        if (round(t_c, 2) in T_store) and (round(t_c, 2) not in T_store_unique):
            u_vec_st[st_idx] = u_vec
            st_idx += 1
            T_store_unique.append(round(t_c, 2))

        # Update variables
        t_c += ht
        ht = c * np.min(hx / l_max_minus)  # time step



    fig, ax = plt.subplots(1, 3)
    plt.tight_layout()
    cycol = cycle('bgrcmk')


    for k, v in u_vec_st.items():

        c = next(cycol)
        ax[0].plot(x, u_vec_st[k][0, :], c=c, label='t = {:.2f}'.format(T_store_unique[k]))
        ax[0].set_title('Water height', fontsize=16)
        ax[0].set_ylabel(r'$h(x, t)$', fontsize=16)
        ax[0].set_xlabel('x', fontsize=16)
        ax[0].legend(loc='best', fontsize=16)
        ax[0].tick_params(labelsize=14)

        ax[1].plot(x, mom2speed(u_vec_st[k][1, :], u_vec_st[k][0, :]), c=c, label='t = {:.2f}'.format(T_store_unique[k]))
        ax[1].set_title('Water speed u', fontsize=16)
        ax[1].set_ylabel(r'$u(x ,t)$', fontsize=16)
        ax[1].set_xlabel('x', fontsize=16)
        ax[1].tick_params(labelsize=14)

        ax[2].plot(x, mom2speed(u_vec_st[k][2, :], u_vec_st[k][0, :]), c=c, label='t = {:.2f}'.format(T_store_unique[k]))
        ax[2].set_title('Water speed v', fontsize=16)
        ax[2].set_ylabel(r'$v(x ,t)$', fontsize=16)
        ax[2].set_xlabel('x', fontsize=16)
        ax[2].tick_params(labelsize=14)

