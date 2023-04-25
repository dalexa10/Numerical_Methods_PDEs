__author__ = 'Dario Rodriguez'
__author__ = 'Elena Fernandez'

import numpy as np
from scipy.optimize import fsolve
from sklearn.metrics import mean_squared_error
from matplotlib import ticker

# -----------------------------------------------------
#               Thermodynamic functions
# -----------------------------------------------------
def compute_speed_of_sound(P, rho, gam=7/5):
    """ Compute speed of sound as function of pressure and density"""
    c = np.sqrt(gam * P / rho)
    return c

def compute_pressure(rho, rho_u, E, gam=7/5):
    """ Compute pressure as function of primitive variables """
    P = (gam - 1) * (E - 0.5 * (rho_u**2) / rho)
    return P


# -----------------------------------------------------
#               Initial condition function
#                   Sod's Problem
# -----------------------------------------------------
def initial_fun(x, gam=7/5):
    """ Sets the initial value of multidimensional u vector as function of x
    as a Sod's problem instance """
    u_vec = np.zeros([3, x.shape[0]])
    P_vec = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] < 0.5:
            rho = 1.
            rho_u = 0.
            E = 1./(gam - 1)
            u_vec[:, i] = [rho, rho_u, E]
            P_vec[i] = compute_pressure(rho, rho_u, E)
        else:
            rho = 0.125
            rho_u = 0.
            E = 0.1 / (gam - 1)
            u_vec[:, i] = [rho, rho_u, E]
            P_vec[i] = compute_pressure(rho, rho_u, E)
    return u_vec, P_vec

# -----------------------------------------------------
#                 Boundary conditions
#                   Sod's Problem
# -----------------------------------------------------
def left_BC():
    rho = 1.
    rho_u = 0.
    E = 1./ (gam - 1)
    u_l = np.array([rho, rho_u, E]).reshape([-1, 1])
    return u_l

def right_BC():
    rho = 0.125
    rho_u = 0.
    E = 0.1 / (gam - 1)
    u_r = np.array([rho, rho_u, E]).reshape([-1, 1])
    return u_r


# -----------------------------------------------------
#                   Exact Solution
#                   Sod's Problem
# -----------------------------------------------------
def pressureEqn(p2p1, p4p1, g, a1a4):
    num = (g - 1) * a1a4 * (p2p1 - 1)
    den = np.sqrt(2 * g * (2 * g + (g + 1) * (p2p1 - 1)))
    powterm = np.power(1 - num / den, -2 * g / (g - 1))
    return p2p1 * powterm - p4p1

def Sod_Exact_solution(x, T, gam=7/5):
    """
    Sod's problem has 4 zones:
        Lelft BC [4] Expansion [3] Contact surface [2] Shock [1] Right BC
    """
    # Left boundary condition
    rho4, rho_u4, E4 = left_BC()
    u4 = rho_u4 / rho4
    p4 = compute_pressure(rho4, rho_u4, E4)

    # Right boundary condition
    rho1, rho_u1, E1 = right_BC()
    u1 = rho_u1 / rho1
    p1 = compute_pressure(rho1, rho_u1, E1)

    x0 = 0.5
    gpm = (gam + 1) / (gam - 1)
    p4p1 = p4 / p1                    # Left / right pressure ratio
    a4 = np.sqrt(gam * p4 / rho4)   # Left speed of sound
    a1 = np.sqrt(gam * p1 / rho1)   # Right speed of sound
    a4a1 = a4/a1                    # Ratio of the speed of sounds

    p2p1 = fsolve(pressureEqn, 1.0, args=(p4p1, gam, 1.0 / a4a1))[0]  # This gives the pressure ratio of the shock
    ushock = a1 * np.sqrt((gam + 1) / (2 * gam) * (p2p1 - 1) + 1)

    # Region between shock and contact surface
    p2 = p2p1 * p1
    rho2rho1 = (1 + gpm * p2p1) / (gpm + p2p1)
    rho2 = rho2rho1 * rho1
    u2 = a1 / gam * (p2p1 - 1) * np.sqrt(((2 * gam) / (gam + 1)) / (p2p1 + 1.0/gpm))

    # Expansion wave
    p3p4 = p2p1 / p4p1
    p3 = p3p4 * p4
    rho3rho4 = np.power(p3p4, 1 / gam)
    rho3 = rho3rho4 * rho4
    u3 = u2     # Velocity is unchanged across the contact surface

    # Regions are based on time
    x1 = x0 - a4 * T            # Location of the left part of the expansion fan
    a3 = np.sqrt(gam * p3 / rho3)
    x2 = x0 + (u3 - a3) * T     # Location of the right part of the expansion fan
    x3 = x0 + u2 * T            # Location of the contact surface
    x4 = x0 + ushock * T        # Location of the shock wave

    # Expansion region
    u_vec = np.zeros((3, len(x)))
    for i in np.arange(0, len(x)):
        x_i = x[i]
        if x_i <= x1:               # Left region
            u_vec[0, i] = rho4
            u_vec[1, i] = rho4 * u4
            u_vec[2, i] = p4 / (gam - 1) + rho4 * np.power(u4, 2.0) / 2
        elif x_i > x1 and x_i <= x2:    # Expansion fan
            u = 2 / (gam + 1) * (a4 + (x_i - x0) / T)
            p = p4 * np.power(1 - (gam - 1) / 2 * (u/a4), (2 * gam) / (gam - 1))
            rho = rho4 * np.power(1 - (gam - 1) / 2 * (u/a4), 2 / (gam - 1))
            u_vec[0, i] = rho
            u_vec[1, i] = rho*u
            u_vec[2, i] = p / (gam - 1) + rho * np.power(u, 2.0) / 2
        elif x_i > x2 and x_i <= x3: # Between the expansion and contact surface
            u_vec[0, i] = rho3
            u_vec[1, i] = rho3 * u3
            u_vec[2, i] = p3 / (gam - 1) + rho3 * np.power(u3, 2.0) / 2
        elif x_i > x3 and x_i <= x4: # Between the contact surface and the shock
            u_vec[0, i] = rho2
            u_vec[1, i] = rho2 * u2
            u_vec[2, i] = p2 / (gam - 1) + rho2 * np.power(u2, 2.0) / 2
        elif x_i > x4: # The right region
            u_vec[0, i] = rho1
            u_vec[1, i] = rho1 * u1
            u_vec[2, i] = p1 / (gam - 1) + rho1 * np.power(u1, 2.0) / 2

    return u_vec

# -----------------------------------------------------
#     Coefficients for WENO scheme for p = 2 & p = 3
# -----------------------------------------------------
def cr_coefficients(p):
    if p == 2:
        c_rj = np.array([[3/2, -1/2],
                         [1/2, 1/2],
                         [-1/2, 3/2]])
    else:
        c_rj = np.array([[11/6, -7/6, 1/3],
                         [1/3, 5/6, -1/6],
                         [-1/6, 5/6, 1/3],
                         [1/3, -7/6, 11/6]])
    return c_rj


def dr_coefficients(p):
    if p == 2:
        d_r = np.array([2/3, 1/3])
    else:
        d_r = np.array([3/10, 6/10, 1/10])
    return d_r

def add_BC_conditions_to_ghost_cells(u_vec, p):
    """ Add boundary conditions to edge cells (ghost) depending on p degree of
    interpolating polynomial """
    if p == 2:
        u_vec[:, :2] = left_BC() * np.ones([3, 2])
        u_vec[:, -2:] = right_BC() * np.ones([3, 2])
    else:
        u_vec[:, :3] = left_BC() * np.ones([3, 3])
        u_vec[:, -3:] = right_BC() * np.ones([3, 3])

    return u_vec


# -----------------------------------------------------
#               Numerical functions
# -----------------------------------------------------
def flux_fun(u_vec):
    """ Compute the flux vector from given u vector (states) """
    rho, rho_u, E = u_vec[0, :], u_vec[1, :], u_vec[2, :]
    P = compute_pressure(rho, rho_u, E)
    f_u = np.stack((rho_u,
                    rho_u**2 / rho + P,
                    (E + P) * rho_u / rho))
    return f_u

def compute_LF_flux(u_minus, u_plus, alpha_LF):
    """ Computes the Lax-Friedrich flux value """
    flux = 0.5 * (flux_fun(u_minus) + flux_fun(u_plus)) - 0.5 * alpha_LF * (u_plus - u_minus)
    return flux

def compute_eigenvalues(u_vec, gam=7/5):
    """ Compute the eigenvalues of the mass matrix of the hyperbolic system
    of the 1D Euler equations """
    rho, rho_u, E = u_vec[0, :], u_vec[1, :], u_vec[2, :]
    u = rho_u / rho
    P = compute_pressure(rho, rho_u, E)
    c = compute_speed_of_sound(P, rho, gam)
    lamb_v = np.stack((u - c, u, u + c))
    return lamb_v

def compute_lambda_max(lamb_v):
    """ Compute the maximum value of the eigenvalues """
    max_l = np.amax(np.abs(lamb_v))
    return max_l

def compute_alfa_LF(l_v_minus, l_v_plus):
    """ Compute the factor alpha k + 1/2 for the Lax-Fridrich flux function"""
    max_l_minus = compute_lambda_max(l_v_minus)
    max_l_plus = compute_lambda_max(l_v_plus)
    alpha = np.maximum(max_l_minus, max_l_plus)
    return alpha

def compute_polynomials(u_vec, p, side):
    """
    Compute the interpolating polynomials phi for the cell interfaces
        right: k + 1/2
        left: k - 1/2
    """
    c_rj = cr_coefficients(p)
    global K, Km2, Km1, Kp2, Kp1

    if p == 2:
        if side == 'right':
            p0 = c_rj[1, 0] * u_vec[:, K] + c_rj[1, 1] * u_vec[:, Kp1]
            p1 = c_rj[2, 0] * u_vec[:, Km1] + c_rj[2, 1] * u_vec[:, K]
        else:
            p0 = c_rj[0, 0] * u_vec[:, K] + c_rj[0, 1] * u_vec[:, Kp1]
            p1 = c_rj[1, 0] * u_vec[:, Km1] + c_rj[1, 1] * u_vec[:, K]

        return p0, p1

    else:
        if side == 'right':
            p0 = c_rj[1, 0] * u_vec[:, K] + c_rj[1, 1] * u_vec[:, Kp1] + c_rj[1, 2] * u_vec[:, Kp2]
            p1 = c_rj[2, 0] * u_vec[:, Km1] + c_rj[2, 1] * u_vec[:, K] + c_rj[2, 2] * u_vec[:, Kp1]
            p2 = c_rj[3, 0] * u_vec[:, Km2] + c_rj[3, 1] * u_vec[:, Km1] + c_rj[3, 2] * u_vec[:, K]
        else:
            p0 = c_rj[0, 0] * u_vec[:, K] + c_rj[0, 1] * u_vec[:, Kp1] + c_rj[0, 2] * u_vec[:, Kp2]
            p1 = c_rj[1, 0] * u_vec[:, Km1] + c_rj[1, 1] * u_vec[:, K] + c_rj[1, 2] * u_vec[:, Kp1]
            p2 = c_rj[2, 0] * u_vec[:, Km2] + c_rj[2, 1] * u_vec[:, Km1] + c_rj[2, 2] * u_vec[:, K]

        return p0, p1, p2

def compute_beta(u_vec, p):
    """ Calculate smoothness indicators beta """

    global K, Km2, Km1, Kp2, Kp1
    if p == 2:
        beta0 = (u_vec[:, Kp1] - u_vec[:, K])**2
        beta1 = (u_vec[:, K] - u_vec[:, Km1])**2
        return [beta0, beta1]
    else:
        beta0 = (13/12) * (u_vec[:, K] - 2 * u_vec[:, Kp1] + u_vec[:, Kp2])**2 + \
                (1/4) * (3 * u_vec[:, K] - 4 * u_vec[:, Kp1] + u_vec[:, Kp2])**2
        beta1 = (13/12) * (u_vec[:, Km1] - 2 * u_vec[:, K] + u_vec[:, Kp1])**2 + \
                (1/4) * (u_vec[:, Km1] - u_vec[:, Kp1])**2
        beta2 = (13/12) * (u_vec[:, Km2] - 2 * u_vec[:, Km1] + u_vec[:, K])**2 + \
                (1/4) * (u_vec[:, Km2] - 4 * u_vec[:, Km1] + 3 * u_vec[:, K])**2
        return [beta0, beta1, beta2]

def compute_weights(u_vec, p, eps, side):
    """ Compute weights for the WENO scheme """
    dr = dr_coefficients(p)
    beta = compute_beta(u_vec, p)
    if p == 2:
        if side == 'right':
            alfa_0 = dr[0] / (eps + beta[0])**2
            alfa_1 = dr[1] / (eps + beta[1])**2
        else:
            alfa_0 = dr[1] / (eps + beta[0])**2
            alfa_1 = dr[0] / (eps + beta[1])**2
        alfa_sum = alfa_0 + alfa_1
        w = [alfa_0/alfa_sum, alfa_1/alfa_sum]
        return w
    else:
        if side == 'right':
            alfa_0 = dr[0] / (eps + beta[0])**2
            alfa_1 = dr[1] / (eps + beta[1])**2
            alfa_2 = dr[2] / (eps + beta[2])**2
        else:
            alfa_0 = dr[2] / (eps + beta[0])**2
            alfa_1 = dr[1] / (eps + beta[1])**2
            alfa_2 = dr[0] / (eps + beta[2])**2

        alfa_sum = alfa_0 + alfa_1 + alfa_2
        w = [alfa_0/alfa_sum, alfa_1/alfa_sum, alfa_2/alfa_sum]

        return w

# -----------------------------------------------------
#                   WENO3 - RK2
# -----------------------------------------------------
def WENO3(u_vec, eps, p=2):
    """
    WENO scheme computes:
        u_hat_minus
        u_hat_plus
        for a third order polynomial (p = 3)
    """
    global K, Km2, Km1, Kp2, Kp1

    # ------ \hat_{u}_{k + 1/2} ---------
    p0_r, p1_r = compute_polynomials(u_vec, p, side='right')
    w0_r, w1_r = compute_weights(u_vec, p, eps, side='right')
    u_hat_plus = w0_r * p0_r + w1_r * p1_r

    # ------ \hat_{u}_{k + 1 - 1/2} --------
    u_vec_kp1 = u_vec[:, Kp1]  # Roll u_vec plus 1
    p0_l, p1_l = compute_polynomials(u_vec_kp1, p, side='left')
    w0_l, w1_l = compute_weights(u_vec_kp1, p, eps, side='left')
    u_hat_minus = w0_l * p0_l + w1_l * p1_l
    return u_hat_minus, u_hat_plus


def RK2_Integration(u_vec, ht, p, eps):
    """ Three-stage Runge-Kutta integration of constructed u vectors """
    global Km1, Km2, K, Kp1, Kp2

    # --------- First stage -----------
    u_hat_minus_1, u_hat_plus_1 = WENO3(u_vec, eps)
    u_v_minus_1 = u_hat_plus_1
    u_v_plus_1 = u_hat_minus_1
    lamb_v_minus_1, lamb_v_plus_1 = compute_eigenvalues(u_v_minus_1), compute_eigenvalues(u_v_plus_1)
    alfa_LF_1 = compute_alfa_LF(lamb_v_minus_1, lamb_v_plus_1)
    flux_1 = compute_LF_flux(u_v_minus_1, u_v_plus_1, alfa_LF_1)
    u_vec_1 = u_vec - (ht / hx) * (flux_1[:, K] - flux_1[:, Km1])
    u_vec_1 = add_BC_conditions_to_ghost_cells(u_vec_1, p)

    # --------- Second stage -----------
    u_hat_minus_2, u_hat_plus_2 = WENO5(u_vec_1, eps)
    u_v_minus_2 = u_hat_plus_2
    u_v_plus_2 = u_hat_minus_2
    lamb_v_minus_2, lamb_v_plus_2 = compute_eigenvalues(u_v_minus_2), compute_eigenvalues(u_v_plus_2)
    alfa_LF_2 = compute_alfa_LF(lamb_v_minus_2, lamb_v_plus_2)
    flux_2 = compute_LF_flux(u_v_minus_2, u_v_plus_2, alfa_LF_2)
    u_vec_2 = 0.5 * u_vec + 0.5 * u_vec_1 - 0.5 * (ht / hx) * (flux_2[:, K] - flux_2[:, Km1])
    u_vec_2 = add_BC_conditions_to_ghost_cells(u_vec_2, p)

    u_vec = u_vec_2
    P_vec = compute_pressure(u_vec[0, :], u_vec[1, :], u_vec[2, :])

    return u_vec, P_vec


# -----------------------------------------------------
#                   WENO5 - RK3
# -----------------------------------------------------
def WENO5(u_vec, eps, p=3):
    """
    WENO scheme computes:
        u_hat_minus
        u_hat_plus
        for a third order polynomial (p = 3)
    """
    global K, Km2, Km1, Kp2, Kp1

    # ------ \hat_{u}_{k + 1/2} ---------
    p0_r, p1_r, p2_r = compute_polynomials(u_vec, p, side='right')
    w0_r, w1_r, w2_r = compute_weights(u_vec, p, eps, side='right')
    u_hat_plus = w0_r * p0_r + w1_r * p1_r + w2_r * p2_r

    # ------ \hat_{u}_{k + 1 - 1/2} --------
    u_vec_kp1 = u_vec[:, Kp1]  # Roll u_vec plus 1
    p0_l, p1_l, p2_l = compute_polynomials(u_vec_kp1, p, side='left')
    w0_l, w1_l, w2_l = compute_weights(u_vec_kp1, p, eps, side='left')
    u_hat_minus = w0_l * p0_l + w1_l * p1_l + w2_l * p2_l
    return u_hat_minus, u_hat_plus


def RK3_Integration(u_vec, ht, p, eps):
    """ Three-stage Runge-Kutta integration of constructed u vectors """
    global Km1, Km2, K, Kp1, Kp2

    # --------- First stage -----------
    u_hat_minus_1, u_hat_plus_1 = WENO5(u_vec, eps)
    u_v_minus_1 = u_hat_plus_1
    u_v_plus_1 = u_hat_minus_1
    lamb_v_minus_1, lamb_v_plus_1 = compute_eigenvalues(u_v_minus_1), compute_eigenvalues(u_v_plus_1)
    alfa_LF_1 = compute_alfa_LF(lamb_v_minus_1, lamb_v_plus_1)
    flux_1 = compute_LF_flux(u_v_minus_1, u_v_plus_1, alfa_LF_1)
    u_vec_1 = u_vec - (ht / hx) * (flux_1[:, K] - flux_1[:, Km1])
    u_vec_1 = add_BC_conditions_to_ghost_cells(u_vec_1, p)

    # --------- Second stage -----------
    u_hat_minus_2, u_hat_plus_2 = WENO5(u_vec_1, eps)
    u_v_minus_2 = u_hat_plus_2
    u_v_plus_2 = u_hat_minus_2
    lamb_v_minus_2, lamb_v_plus_2 = compute_eigenvalues(u_v_minus_2), compute_eigenvalues(u_v_plus_2)
    alfa_LF_2 = compute_alfa_LF(lamb_v_minus_2, lamb_v_plus_2)
    flux_2 = compute_LF_flux(u_v_minus_2, u_v_plus_2, alfa_LF_2)
    u_vec_2 = (0.75 * u_vec) + (0.25 * u_vec_1) - 0.25 * (ht / hx) * (flux_2[:, K] - flux_2[:, Km1])
    u_vec_2 = add_BC_conditions_to_ghost_cells(u_vec_2, p)

    # --------- Third stage -----------
    u_hat_minus_3, u_hat_plus_3 = WENO5(u_vec_2, eps)
    u_v_minus_3 = u_hat_plus_3
    u_v_plus_3 = u_hat_minus_3
    lamb_v_minus_3, lamb_v_plus_3 = compute_eigenvalues(u_v_minus_3), compute_eigenvalues(u_v_plus_3)
    alfa_LF_3 = compute_alfa_LF(lamb_v_minus_3, lamb_v_plus_3)
    flux_3 = compute_LF_flux(u_v_minus_3, u_v_plus_3, alfa_LF_3)
    u_vec_3 = (1/3) * u_vec + (2/3) * u_vec_2 - (2/3) * (ht / hx) * (flux_3[:, K] - flux_3[:, Km1])
    u_vec_3 = add_BC_conditions_to_ghost_cells(u_vec_3, p)

    u_vec = u_vec_3
    P_vec = compute_pressure(u_vec[0, :], u_vec[1, :], u_vec[2, :])

    return u_vec, P_vec


# -----------------------------------------------------
#                   Aux. Functions
# -----------------------------------------------------
def get_mark(k):
    p = int(k[-1])
    if p == 2:
        m = 'v'
    else:
        m = 's'
    return m

def compute_error(u_num, u_exact, hx):
    """ Compute the L2 norm of the error """
    # error = u_num - u_exact
    # l2err = np.sqrt(hx * np.sum(error**2, axis=1))
    mse = [mean_squared_error(u_exact[0, :], u_num[0, :]),
           mean_squared_error(u_exact[1, :], u_num[1, :]),
           mean_squared_error(u_exact[2, :], u_num[2, :])]
    return mse

def formater_scientific(ax):
    """ Formats the axis in scientific notation """
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.get_offset_text().set_fontsize(14)
    return ax.yaxis.set_major_formatter(formatter)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from itertools import cycle

    import matplotlib
    matplotlib.rcParams['animation.embed_limit'] = 2**28
    from matplotlib import animation


    # -------------------------------------
    #               User Inputs
    # --------------------------------------
    nx_ls = [100, 500]  # Discretization nodes in x
    gam = 7/5           # Specific heat ratio
    CFL = 0.5           # CFL condition
    T = 0.2             # Final time of the simulation
    eps = 1e-6
    anim = True

    # -------------------------------------
    #               Processing
    # --------------------------------------
    out = {}
    error_dict = {}
    if anim:
        out_an = {}

    for nx in nx_ls:
        x, hx = np.linspace(0, 1, nx, endpoint=True, retstep=True)
        u_vec, P_vec = initial_fun(x)
        u_vec_0, P_vec_0 = u_vec.copy(), P_vec.copy()
        u_vec_2, P_vec_2 = u_vec.copy(), P_vec.copy()

        l_v = compute_eigenvalues(u_vec)
        max_l_0 = compute_lambda_max(l_v)
        ht = CFL * hx / max_l_0
        ht_2 = ht.copy()

        # Indexes arrays
        K = np.arange(0, nx)    # 0, ..., nx-1
        Km1 = np.roll(K, 1)     # nx-1, 0, 1, ..., nx-2
        Km2 = np.roll(K, 2)     # nx-2, 0, 1, ..., nx-3
        Kp1 = np.roll(K, -1)    # 1, ..., nx
        Kp2 = np.roll(K, -2)    # 2, ..., nx + 1

        # -------------------------------------
        #               Computation
        # --------------------------------------
        t_c = 0
        t_c_2 = 0
        p_w5 = 3  # Degree of interpolating polynomial
        p_w3 = 2  # Degree of interpolating polynomial

        k1 = str(nx) + '_' + str(p_w5)
        k2 = str(nx) + '_' + str(p_w3)

        error_dict[k1] = {'t': [], 'e': [[], [], []]}
        error_dict[k2] = {'t': [], 'e': [[], [], []]}

        out_an[k1] = {'t': [],
                      'u_ex': [[], [], []],
                      'u': [[], [], []]}

        out_an[k2] = {'t': [],
                      'u_ex': [[], [], []],
                      'u': [[], [], []]}

        # -------------------------------------
        #              WENO5 - RK3
        # --------------------------------------

        while t_c < T:
            # Numerical tensor update
            u_vec, P_vec = RK3_Integration(u_vec, ht, p_w5, eps)

            # Exact solution
            u_vec_ex = Sod_Exact_solution(x, t_c)

            # Error calculation
            e_i = compute_error(u_vec, u_vec_ex, hx)
            [error_dict[k1]['e'][i].append(e_i[i]) for i in range(3)]
            error_dict[k1]['t'].append(t_c)

            # Animation section
            if anim:
                out_an[k1]['t'].append(t_c_2)
                [out_an[k1]['u_ex'][i].append(u_vec_ex[i]) for i in range(3)]
                [out_an[k1]['u'][i].append(u_vec[i]) for i in range(3)]

            # Time update
            t_c += ht

            # Dynamic time step
            l_v = compute_eigenvalues(u_vec, P_vec)
            max_l = compute_lambda_max(l_v)
            ht = CFL * hx / max_l

        out[k1] = {'u': u_vec,
                   'P': P_vec,
                   'x': x}

        # -------------------------------------
        #              WENO3 - RK2
        # --------------------------------------

        while t_c_2 < T:
            # Tensor update
            u_vec_2, P_vec_2 = RK2_Integration(u_vec_2, ht, p_w3, eps)

            # Animation section
            u_vec_ex_2 = Sod_Exact_solution(x, t_c_2)

            # Error calculation
            e_i_2 = compute_error(u_vec_2, u_vec_ex_2, hx)
            [error_dict[k2]['e'][i].append(e_i_2[i]) for i in range(3)]
            error_dict[k2]['t'].append(t_c_2)

            # Animation section
            if anim:
                out_an[k2]['t'].append(t_c_2)
                [out_an[k2]['u_ex'][i].append(u_vec_ex_2[i]) for i in range(3)]
                [out_an[k2]['u'][i].append(u_vec_2[i]) for i in range(3)]

            # Time update
            t_c_2 += ht_2

            # Dynamic time step
            l_v_2 = compute_eigenvalues(u_vec_2, P_vec_2)
            max_l_2 = compute_lambda_max(l_v_2)
            ht_2 = CFL * hx / max_l_2

        out[k2] = {'u': u_vec_2,
                   'P': P_vec_2,
                   'x': x}

    x_ex = np.linspace(0, 1, 500, endpoint=True)
    u_vec_ex = Sod_Exact_solution(x_ex, T)
    P_vec_ex = compute_pressure(u_vec_ex[0, :], u_vec_ex[1, :], u_vec_ex[2, :])

    # -------------------------------------
    #               Plotting
    # --------------------------------------

    fig, ax = plt.subplots(2, 2)
    plt.tight_layout()
    cycol = cycle('bgrc')

    ax[0, 0].plot(x_ex, u_vec_ex[0, :], ls='--', lw=1, c='r')
    ax[0, 1].plot(x_ex, u_vec_ex[1, :] / u_vec_ex[0, :], ls='--', lw=1, c='r')
    ax[1, 0].plot(x_ex, u_vec_ex[2, :], ls='--', lw=1, c='r')
    ax[1, 1].plot(x_ex, P_vec_ex, c='r', ls='--', lw=1, label='Exact')

    for k in out.keys():
        c = next(cycol)
        m = get_mark(k)
        ax[0, 0].scatter(out[k]['x'], out[k]['u'][0, :], marker=m, c=c, s=5)
        ax[0, 1].scatter(out[k]['x'], out[k]['u'][1, :] / out[k]['u'][0, :], marker=m, c=c, s=5)
        ax[1, 0].scatter(out[k]['x'], out[k]['u'][2, :], marker=m, c=c, s=5)
        ax[1, 1].scatter(out[k]['x'], out[k]['P'], marker=m, c=c, s=5,
                         label=r'$n_x$ = {} - p = {}'.format(k[:3], k[-1]))

    ax[0, 0].set_ylabel(r'$\rho$', fontsize=16)
    ax[0, 1].set_ylabel('u', fontsize=16)
    ax[1, 0].set_xlabel('x', fontsize=16)
    ax[1, 0].set_ylabel('E', fontsize=16)
    ax[1, 1].legend(loc='best', fontsize=16)
    [ax[i, j].tick_params(labelsize=16) for i in range(2) for j in range(2)]


    fig, ax = plt.subplots(1, 3)
    plt.tight_layout()
    plt.ticklabel_format(axis="y", style="sci")

    for k in error_dict.keys():
        ax[0].plot(error_dict[k]['t'], error_dict[k]['e'][0])
        ax[1].plot(error_dict[k]['t'], error_dict[k]['e'][1])
        ax[2].plot(error_dict[k]['t'], error_dict[k]['e'][2], label=r'$n_x$ = {} - p = {}'.format(k[:3], k[-1]))

    ax[0].set_ylabel(r'$L_{2}$ error $\rho$', fontsize=16)
    ax[1].set_ylabel(r'$L_{2}$ error $u$', fontsize=16)
    ax[2].set_ylabel(r'$L_{2}$ error $E$', fontsize=16)

    ax[0].set_xlabel('t', fontsize=16)
    ax[1].set_xlabel('t', fontsize=16)
    ax[2].set_xlabel('t', fontsize=16)
    ax[2].legend(loc='best', fontsize=16)
    [ax[i].tick_params(labelsize=16) for i in range(3)]
    [formater_scientific(ax[i]) for i in range(3)]


# %%

    # -------------------------------------
    #               Animation
    # --------------------------------------

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    [ax[i].set_box_aspect(1) for i in range(3)]
    plt.tight_layout()

    # -------------------------------------
    #              WENO5 - RK3
    # --------------------------------------
    u_vec_ex_an = out_an['100_3']['u_ex']
    u_vec_num_an = out_an['100_3']['u']

    rho_ex_an, rho_u_ex_an, E_ex_an = [np.array(u_vec_ex_an[j]) for j in range(3)]
    u_ex_an = rho_u_ex_an / rho_ex_an

    rho_an, rho_u_an, E_an = [np.array(u_vec_num_an[j]) for j in range(3)]
    u_an = rho_u_an / rho_an

    x_an = np.linspace(0, 1, rho_an.shape[1], endpoint=True)

    # -------------------------------------
    #              WENO3 - RK2
    # --------------------------------------
    u_vec_ex_an_2 = out_an['100_2']['u_ex']
    u_vec_num_an_2 = out_an['100_2']['u']

    rho_ex_an_2, rho_u_ex_an_2, E_ex_an_2 = [np.array(u_vec_ex_an_2[j]) for j in range(3)]
    u_ex_an_2 = rho_u_ex_an_2 / rho_ex_an_2

    rho_an_2, rho_u_an_2, E_an_2 = [np.array(u_vec_num_an_2[j]) for j in range(3)]
    u_an_2 = rho_u_an_2 / rho_an_2

    x_an_2 = np.linspace(0, 1, rho_an_2.shape[1], endpoint=True)

    line0, = ax[0].plot(x_an, rho_ex_an[0, :], c='red', lw=2, clip_on=False)
    line1, = ax[0].plot(x_an, rho_an[0, :], c='blue', lw=2, clip_on=False)
    line2, = ax[0].plot(x_an_2, rho_an_2[0, :], c='green', lw=2, clip_on=False)

    ax[1].set_ylim([- 0.1, np.max(u_ex_an) + 0.1])
    line3, = ax[1].plot(x_an, u_ex_an[0, :], c='red', lw=2, clip_on=False)
    line4, = ax[1].plot(x_an, u_an[0, :], c='blue', lw=2, clip_on=False)
    line5, = ax[1].plot(x_an_2, u_an_2[0, :], c='green', lw=2, clip_on=False)

    line6, = ax[2].plot(x_an, E_ex_an[0, :], c='red', lw=2, clip_on=False, label='Exact Solution')
    line7, = ax[2].plot(x_an, E_an[0, :], c='blue', lw=2, clip_on=False, label=r'$n_{x} = 100, p = 3$')
    line8, = ax[2].plot(x_an_2, E_an_2[0, :], c='green', lw=2, clip_on=False, label=r'$n_{x} = 100, p = 2$')

    ax[2].legend(loc='upper left')

    def init():
        pass
    def time_stepper(n):
        line0.set_data(x_an, rho_ex_an[n, :])
        line1.set_data(x_an, rho_an[n, :])
        line2.set_data(x_an_2, rho_an_2[n, :])

        line3.set_data(x_an, u_ex_an[n, :])
        line4.set_data(x_an, u_an[n, :])
        line5.set_data(x_an_2, u_an_2[n, :])

        line6.set_data(x_an, E_ex_an[n, :])
        line7.set_data(x_an, E_an[n, :])
        line8.set_data(x_an_2, E_an_2[n, :])

        return line0, line1, line2, line3, line4, line5, line6, line7, line8

    nt_an = rho_an_2.shape[0]  # Set this value as the lowest size (in time) between the vectors you want to animate
    # In this case rho_an_2.shape[0] = 59 and rho_an.shape[0] = 60

    ani = animation.FuncAnimation(fig, time_stepper,
                                  frames=nt_an, interval=30,
                                  init_func=init)
    ani.save('anim.gif', writer='PillowWriter', fps=30)



