__author__ = 'Dario Rodriguez'
__author__ = 'Elena Fernandez'

import numpy as np

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

def flux_fun(u_vec):
    """ Compute the flux vector from given u vector (states) """
    rho, rho_u, E = u_vec[0, :], u_vec[1, :], u_vec[2, :]
    P = compute_pressure(rho, rho_u, E)
    f_u = np.stack((rho_u,
                    rho_u**2 / rho + P,
                    (E + P) * rho_u / rho))
    return f_u


# -----------------------------------------------------
#               Numerical functions
# -----------------------------------------------------
def add_spatial_ghost_cells(x, p, hx):
    """ Add left and right ghost cells to the 1D mesh
    The number of ghost cells depends on the polynomial order for
    reconstruction """
    start_x = np.array([x[0] - p_i * hx for p_i in range(1, p)])
    end_x = np.array([x[-1] + p_i * hx for p_i in range(1, p)])
    x = np.insert(x, 0, start_x)
    x = np.insert(x, -1, end_x)
    return x

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
    """ Compute the factor alpha k + 1/2 """
    max_l_minus = compute_lambda_max(l_v_minus)
    max_l_plus = compute_lambda_max(l_v_plus)
    alpha = np.maximum(max_l_minus, max_l_plus)
    return alpha

def cr_coefficiens(p):
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

def compute_polynomials(u_vec, p, side):
    c_rj = cr_coefficiens(p)
    global K, Km2, Km1, Kp2, Kp1
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


def left_BC():
    rho = 1.
    rho_u = 0.
    E = 1./(gam - 1)
    u_l = np.array([rho, rho_u, E]).reshape([-1, 1])
    return u_l

def right_BC():
    rho = 0.125
    rho_u = 0.
    E = 0.1 /(gam - 1)
    u_r = np.array([rho, rho_u, E]).reshape([-1, 1])
    return u_r

def add_BC_conditions_to_ghost_cells(u_vec, p):
    """ Creates ghost cells on u vector depending on p degree of
    interpolating polynomial """
    if p == 2:
        u_vec[:, 0] = left_BC()
        u_vec[:, -1] = right_BC()
    else:
        u_vec[:, :3] = left_BC() * np.ones([3, 3])
        u_vec[:, -3:] = right_BC() * np.ones([3, 3])

    return u_vec

def WENO5(u_vec, eps, p=3):
    global K, Km2, Km1, Kp2, Kp1

    # \hat_{u}_{k + 1/2}
    p0_r, p1_r, p2_r = compute_polynomials(u_vec, p, side='right')
    w0_r, w1_r, w2_r = compute_weights(u_vec, p, eps, side='right')
    u_hat_plus = w0_r * p0_r + w1_r * p1_r + w2_r * p2_r

    # \hat_{u}_{k + 1 - 1/2}
    u_vec_kp1 = u_vec[:, Kp1]
    p0_l, p1_l, p2_l = compute_polynomials(u_vec_kp1, p, side='left')
    w0_l, w1_l, w2_l = compute_weights(u_vec_kp1, p, eps, side='left')
    u_hat_minus = w0_l * p0_l + w1_l * p1_l + w2_l * p2_l
    return u_hat_minus, u_hat_plus

def compute_LF_flux(u_minus, u_plus, alpha_LF):
    """ Computes the Lax-Friedrich flux value """
    flux = 0.5 * (flux_fun(u_minus) + flux_fun(u_plus)) - 0.5 * alpha_LF * (u_plus - u_minus)
    return flux

def RK3_Integration(u_vec, ht, p, eps):
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



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # -------------------------------------
    #               User Inputs
    # --------------------------------------
    nx = 500        # Discretization nodes in x
    gam = 7/5       # Specific heat ratio
    CFL = 0.5       # CFL condition
    T = 0.2         # Final time of the simulation
    p = 3           # Degree of interpolating polynomial
    eps = 1e-6

    # -------------------------------------
    #               Processing
    # --------------------------------------
    x, hx = np.linspace(0, 1, nx, endpoint=True, retstep=True)
    # x = add_spatial_ghost_cells(x, p, hx)
    # nx = x.shape[0]
    u_vec, P_vec = initial_fun(x)
    l_v = compute_eigenvalues(u_vec)
    max_l_0 = compute_lambda_max(l_v)
    ht = CFL * hx / max_l_0

    # Indexes arrays
    K = np.arange(0, nx)    # 0, ..., nx-1
    Km1 = np.roll(K, 1)     # nx-1, 0, 1, ..., nx-2
    Km2 = np.roll(K, 2)     # nx-2, 0, 1, ..., nx-3
    Kp1 = np.roll(K, -1)    # 1, ..., nx
    Kp2 = np.roll(K, -2)    # 2, ..., nx + 1

    # WENO simulation
    t_c = 0
    while t_c < T:
        # Tensor update
        u_vec, P_vec = RK3_Integration(u_vec, ht, p, eps)

        # Time update
        t_c += ht

        # Dynamic time step
        l_v = compute_eigenvalues(u_vec, P_vec)
        max_l = compute_lambda_max(l_v)
        ht = CFL * hx / max_l

    fig, ax = plt.subplots()
    ax.plot(x, u_vec[0, :])



