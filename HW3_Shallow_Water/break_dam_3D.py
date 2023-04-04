__author__ = 'Dario Rodriguez'

# ----------------------------------------------------------------
#                    3D DAM-BREAKING PROBLEM
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


def mom2speed(m, h):
    """ Compute speed from given momentum and height """
    return m / h

def abs_lambda_max_u(u, h, g=1):
    """ Compute the absolute value of the greatest eigenvalue of the
    Jacobian matrix of the hyperbolic system"""
    alamb_u = np.amax(np.abs(u + np.sqrt(g * h)))
    return alamb_u

def abs_lambda_max_v(v, h, g=1):
    """ Compute the absolute value of the greatest eigenvalue of the
    Jacobian matrix of the hyperbolic system"""
    alamb_v = np.amax(np.abs(v + np.sqrt(g * h)))
    return alamb_v

def initial_fun(x, y):
    """ Sets the initial value of u as function of x """
    u = np.zeros([3, x.shape[0], y.shape[0]])
    for i in range(len(x)):
        for j in range(len(y)):
            if (-0.5 <= x[i] <= 0.5) and (-0.5 <= y[j] <= 0.5):
                u[:, i, j] = [2, 0, 0]
            else:
                u[:, i, j] = [1, 0, 0]
    return u

def flux_fun_f(vec, g=1):
    """
    Flux vector function f(u)
    :param vec: (np.array 3 x nx x ny)
    :return: f(u) (np.array 3 x nx x ny)

     h: water height
     mx: x-momentum
     my: y-momentum
     g: (float) gravity. Default value 1

    """
    h, mx, my = vec[0, :, :], vec[1, :, :], vec[2, :, :]
    f_u = np.array([mx,
                    (mx**2 / h) + 0.5 * g * h**2,
                    mx * my / h])

    return f_u

def flux_fun_g(vec, g=1):
    """
    Flux vector function g(u)
    :param vec: (np.array 3 x nx x ny)
    :return: f(u) (np.array 3 x nx)

     h: water height
     mx: x-momentum
     my: y-momentum
     g: (float) gravity. Default value 1

    """
    h, mx, my = vec[0, :, :], vec[1, :, :], vec[2, :, :]
    g_u = np.array([my,
                    mx * my / h,
                    (my**2 / h) + 0.5 * g * h**2])

    return g_u

def compute_mass(h, hx, hy):
    """ Computes total mass of water """
    m = np.sum(h) * hx * hy
    return m


if __name__ == '__main__':

    # -------------------------------------
    #               User Inputs
    # --------------------------------------
    nx = 60     # Space discretization points in x
    ny = nx     # Space discretization points in y
    c = 0.8     # Constant to determine time discretization
    T = 3.1       # Total simulation time
    T_store = [1.35, 3]
    # T_store = [0.25, 0.5, 1.]

    # -------------------------------------
    #               Processing
    # --------------------------------------
    # Note that the time step is chosen dynamically and the first
    # discretization is computed based on the initial values

    x, hx = np.linspace(-1, 1, nx, endpoint=False, retstep=True)
    y, hy = np.linspace(-1, 1, ny, endpoint=False, retstep=True)
    u_vec = initial_fun(x, y)                     # Vector of conserved variables
    u_v_0 = u_vec.copy()                          # Vector of conserved variables at time 0
    m_0 = compute_mass(u_vec[0, :, :], hx, hy)

    u0 = mom2speed(u_vec[1, :, :], u_vec[0, :, :])
    l_max_u = abs_lambda_max_u(u0, u_vec[0, :, :])

    v0 = mom2speed(u_vec[2, :, :], u_vec[0, :, :])
    l_max_v = abs_lambda_max_v(v0, u_vec[0, :, :])

    ht = (c / 2) * np.min([np.min(hx / l_max_u), np.min(hy / l_max_v)])

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
    m_vec = [m_0]
    t_vec = [t_c]

    while t_c < T:
        u_vec_u_plus = u_vec[:, Kp1, :]
        u_vec_u_minus = u_vec[:, K, :]

        u_vec_v_plus = u_vec[:, :, Kp1]
        u_vec_v_minus = u_vec[:, :, K]

        l_max_u_minus = abs_lambda_max_u(mom2speed(u_vec_u_minus[1, :, :], u_vec_u_minus[0, :, :]),
                                         u_vec_u_minus[0, :, :])
        l_max_u_plus = abs_lambda_max_u(mom2speed(u_vec_u_plus[1, :, :], u_vec_u_plus[0, :, :]),
                                         u_vec_u_plus[0, :, :])
        l_max_u_bar = np.maximum(l_max_u_minus, l_max_u_plus)

        l_max_v_minus = abs_lambda_max_v(mom2speed(u_vec_v_minus[2, :, :], u_vec_v_minus[0, :, :]),
                                         u_vec_v_minus[0, :, :])
        l_max_v_plus = abs_lambda_max_u(mom2speed(u_vec_v_plus[2, :, :], u_vec_v_plus[0, :, :]),
                                         u_vec_v_plus[0, :, :])
        l_max_v_bar = np.maximum(l_max_v_minus, l_max_v_plus)

        flux_f = 0.5 * (flux_fun_f(u_vec_u_minus) + flux_fun_f(u_vec_u_plus)) - \
                 0.5 * l_max_u_bar * (u_vec_u_plus - u_vec_u_minus)

        flux_g = 0.5 * (flux_fun_g(u_vec_v_minus) + flux_fun_g(u_vec_v_plus)) - \
                 0.5 * l_max_v_bar * (u_vec_v_plus - u_vec_v_minus)

        # Tensor v computation
        u_vec = u_vec - (ht / hx) * (flux_f[:, K, :] - flux_f[:, Km1, :]) - \
                (ht / hy) * (flux_g[:, :, K] - flux_g[:, :, Km1])  # u vector

        # Mass calculation
        m_vec.append(compute_mass(u_vec[0, :, :], hx, hy))

        # Boundary conditions (Perfect walls) Normal speed is nullified at the four walls
        u_vec[2, :, 0] = - u_vec[2, :, 1]       # Bottom wall u_y[0] = - u_y[1]
        u_vec[2, :, -1] = - u_vec[2, :, -2]     # Top wall u_y[-1] = - u_y[-2]
        u_vec[1, 0, :] = - u_vec[1, 1, :]       # Left wall u_x[0] = - u_x[1]
        u_vec[1, -1, :] = - u_vec[1, -2, :]     # Right wall u_x[-1] = - u_x[-2]

        # Store tensor if listed
        if (round(t_c, 2) in T_store) and (round(t_c, 2) not in T_store_unique):
            u_vec_st[st_idx] = u_vec
            st_idx += 1
            T_store_unique.append(round(t_c, 2))

        # Add time and update time step dynamically
        t_c += ht
        t_vec.append(t_c)
        ht = (c / 2) * np.min([np.min(hx / l_max_u_minus), np.min(hy / l_max_v_minus)])

    for k, v in u_vec_st.items():

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, u_vec_st[k][0, :, :], cmap='viridis', edgecolor='none')
        ax.set_zlim3d(0, 2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'Water height $h(x,y,t)$ at time $t = {:.2f}$'.format(T_store[k]))
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, mom2speed(u_vec_st[k][1, :, :], u_vec_st[k][0, :, :]), cmap='viridis', edgecolor='none')
        ax.set_zlim3d(-0.5, 0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'$x$ flow speed $u(x,y,t)$ at time $t = {:.2f}$'.format(T_store[k]))
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, mom2speed(u_vec_st[k][2, :, :], u_vec_st[k][0, :, :]), cmap='viridis', edgecolor='none')
        ax.set_zlim3d(-0.5, 0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'$y$ flow speed $v(x,y,t)$ at time $t = {:.2f}$'.format(T_store[k]))
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)

    fig, ax = plt.subplots()
    ax.plot(t_vec, m_vec)
    ax.set_xlabel('t')
    ax.set_ylabel('mass')
