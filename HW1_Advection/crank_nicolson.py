__author__ = 'Dario Rodriguez'

import numpy as np
from numpy.linalg import solve
# from scipy.linalg import solve

def compute_P(nx, lmbda):
    """ Compute P matrix from CN scheme"""
    P = np.zeros([nx, nx])
    P += np.diag([1] * nx, 0)
    P += np.diag([-lmbda] * (nx - 1), -1)
    P += np.diag([lmbda] * (nx - 1), 1)
    return P

def compute_Q(nx, lmbda):
    """ Compute Q matrix from CN scheme"""
    Q = np.zeros([nx, nx])
    Q += np.diag([1] * nx, 0)
    Q += np.diag([lmbda] * (nx - 1), -1)
    Q += np.diag([-lmbda] * (nx - 1), 1)
    return Q

def apply_periodic_BC(P, Q, lmbda):
    """ Applies periodic BC conditions to P and Q operators by
    adding 'ghost points' """
    P[0, -1] = -lmbda
    P[-1, 0] = lmbda
    Q[0, -1] = lmbda
    Q[-1, 0] = - lmbda
    return P, Q

def fun_u(x):
    """ Initial function """
    return np.exp(-100 * (x - 0.5)**2)

def compute_uk_plus_1_(u_k, nx, lmbda):
    """ Solve linear system in the CN scheme """
    P = compute_P(nx, lmbda)
    Q = compute_Q(nx, lmbda)
    u_k_T = u_k.reshape(-1, 1).copy()
    P, Q = apply_periodic_BC(P, Q, lmbda)
    u_k_1_T = solve(P, Q@u_k_T)
    u_k_1 = u_k_1_T.T
    return u_k_1

def solve_PDE(fun0, x_vec, T, c, lmbda=0.9):
    """ Solve the advection partial differential equation with
    periodic boundary conditions """
    hx = x_vec[1] - x_vec[0]
    ht = 4 * hx * lmbda / c
    nt = int(T / ht)
    u_0 = fun0(x_vec)
    u_k_vec = np.zeros([nt, x_vec.shape[0]])
    u_k_vec[0, :] = u_0
    u_k_vec_exact = np.zeros([nt, x_vec.shape[0]])
    u_k_vec_exact[0, :] = u_0

    for t in range(1, nt - 1):
        # Numerical solution CN scheme
        u_k_i = compute_uk_plus_1_(u_k_vec[t - 1, :], x_vec.shape[0], lmbda)
        u_k_vec[t, :] = u_k_i
        # Exact solution CN scheme
        u_k_vec_ex_i = fun0((x_vec - c * t * ht) % 1.0)
        u_k_vec_exact[t, :] = u_k_vec_ex_i

    return u_k_vec, u_k_vec_exact, nt

def compute_error(u_num, u_exact, hx):
    """ Compute the L2 norm of the error """
    error = u_num - u_exact
    l2err = np.sqrt(hx * np.sum(error**2))
    return l2err

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['animation.embed_limit'] = 2**28
    from matplotlib import animation

    # -----------------------------------------------
    #               Input parameters
    # -----------------------------------------------
    c = 1.0
    T = 1.0 / c

    # ---------------------------------------
    #               Inputs lists
    # ---------------------------------------
    nx = [50, 100, 300]
    lmbda = [0.2, 0.5, 1.2]

    results = {}

    for nx_i in nx:
        results[str(nx_i)] = {}
        for i, lb_i in enumerate(lmbda):
            x = np.linspace(0, 1, nx_i, endpoint=False)
            hx = x[1] - x[0]
            u_k_vec, u_k_vec_exact, nt = solve_PDE(fun_u, x, T, c, lmbda=lb_i)
            error_v = [compute_error(u_k_vec[:, i], u_k_vec_exact[:, i], hx) for i in range(u_k_vec.shape[1])]
            results[str(nx_i)][i] = {'x': x,
                                     'u_num': u_k_vec,
                                     'u_ex': u_k_vec_exact,
                                     'err': error_v}

    # -----------------------------------------------
    #               Plotting section
    # -----------------------------------------------
    #%%
    fig, ax = plt.subplots(3, 3)
    plt.rcParams.update({'font.size': 16})
    plt.tick_params(labelsize=14)

    fig.suptitle('Crank-Nicolson Analysis applied to Advection Equation', fontsize=22)

    ax[0, 0].set_title(r'$N_x$=50 $\lambda$ = 0.2')
    ax[0, 0].plot(results['50'][0]['x'], results['50'][0]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[0, 0].plot(results['50'][0]['x'], results['50'][0]['u_num'][20, :], 'r--', lw=3)
    ax[0, 0].plot(results['50'][0]['x'], results['50'][0]['u_num'][50, :], 'r--', lw=3)
    ax[0, 0].plot(results['50'][0]['x'], results['50'][0]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[0, 0].plot(results['50'][0]['x'], results['50'][0]['u_ex'][20, :], 'b', lw=2)
    ax[0, 0].plot(results['50'][0]['x'], results['50'][0]['u_ex'][50, :], 'b', lw=2)
    ax[0, 0].set_ylabel('u', fontsize=20)
    ax[0, 0].legend(loc='upper left')


    ax[0, 1].set_title(r'$N_x$=100 $\lambda$ = 0.2')
    ax[0, 1].plot(results['100'][0]['x'], results['100'][0]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[0, 1].plot(results['100'][0]['x'], results['100'][0]['u_num'][40, :], 'r--', lw=3)
    ax[0, 1].plot(results['100'][0]['x'], results['100'][0]['u_num'][100, :], 'r--', lw=3)
    ax[0, 1].plot(results['100'][0]['x'], results['100'][0]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[0, 1].plot(results['100'][0]['x'], results['100'][0]['u_ex'][40, :], 'b', lw=2)
    ax[0, 1].plot(results['100'][0]['x'], results['100'][0]['u_ex'][100, :], 'b', lw=2)
    ax[0, 1].legend(loc='upper left')


    ax[0, 2].set_title(r'$N_x$=300 $\lambda$ = 0.2')
    ax[0, 2].plot(results['300'][0]['x'], results['300'][0]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[0, 2].plot(results['300'][0]['x'], results['300'][0]['u_num'][100, :], 'r--', lw=3)
    ax[0, 2].plot(results['300'][0]['x'], results['300'][0]['u_num'][250, :], 'r--', lw=3)
    ax[0, 2].plot(results['300'][0]['x'], results['300'][0]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[0, 2].plot(results['300'][0]['x'], results['300'][0]['u_ex'][100, :], 'b', lw=2)
    ax[0, 2].plot(results['300'][0]['x'], results['300'][0]['u_ex'][250, :], 'b', lw=2)
    ax[0, 2].legend(loc='upper left')



    ax[1, 0].set_title(r'$N_x$=50 $\lambda$ = 0.5')
    ax[1, 0].plot(results['50'][1]['x'], results['50'][1]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[1, 0].plot(results['50'][1]['x'], results['50'][1]['u_num'][10, :], 'r--', lw=3)
    ax[1, 0].plot(results['50'][1]['x'], results['50'][1]['u_num'][20, :], 'r--', lw=3)
    ax[1, 0].plot(results['50'][1]['x'], results['50'][1]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[1, 0].plot(results['50'][1]['x'], results['50'][1]['u_ex'][10, :], 'b', lw=2)
    ax[1, 0].plot(results['50'][1]['x'], results['50'][1]['u_ex'][20, :], 'b', lw=2)
    ax[1, 0].set_ylabel('u', fontsize=20)
    ax[1, 0].legend(loc='upper left')


    ax[1, 1].set_title(r'$N_x$=100 $\lambda$ = 0.5')
    ax[1, 1].plot(results['100'][1]['x'], results['100'][1]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[1, 1].plot(results['100'][1]['x'], results['100'][1]['u_num'][20, :], 'r--', lw=3)
    ax[1, 1].plot(results['100'][1]['x'], results['100'][1]['u_num'][40, :], 'r--', lw=3)
    ax[1, 1].plot(results['100'][1]['x'], results['100'][1]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[1, 1].plot(results['100'][1]['x'], results['100'][1]['u_ex'][20, :], 'b', lw=2)
    ax[1, 1].plot(results['100'][1]['x'], results['100'][1]['u_ex'][40, :], 'b', lw=2)
    ax[1, 1].legend(loc='upper left')


    ax[1, 2].set_title(r'$N_x$=300 $\lambda$ = 0.5')
    ax[1, 2].plot(results['300'][1]['x'], results['300'][1]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[1, 2].plot(results['300'][1]['x'], results['300'][1]['u_num'][50, :], 'r--', lw=3)
    ax[1, 2].plot(results['300'][1]['x'], results['300'][1]['u_num'][100, :], 'r--', lw=3)
    ax[1, 2].plot(results['300'][1]['x'], results['300'][1]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[1, 2].plot(results['300'][1]['x'], results['300'][1]['u_ex'][50, :], 'b', lw=2)
    ax[1, 2].plot(results['300'][1]['x'], results['300'][1]['u_ex'][100, :], 'b', lw=2)
    ax[1, 2].legend(loc='upper left')


    ax[2, 0].set_title(r'$N_x$=50 $\lambda$ = 1.2')
    ax[2, 0].plot(results['50'][2]['x'], results['50'][2]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[2, 0].plot(results['50'][2]['x'], results['50'][2]['u_num'][2, :], 'r--', lw=3)
    ax[2, 0].plot(results['50'][2]['x'], results['50'][2]['u_num'][6, :], 'r--', lw=3)
    ax[2, 0].plot(results['50'][2]['x'], results['50'][2]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[2, 0].plot(results['50'][2]['x'], results['50'][2]['u_ex'][2, :], 'b', lw=2)
    ax[2, 0].plot(results['50'][2]['x'], results['50'][2]['u_ex'][6, :], 'b', lw=2)
    ax[2, 0].set_ylabel('u', fontsize=20)
    ax[2, 0].legend(loc='upper left')
    ax[2, 0].set_xlabel('x')

    ax[2, 1].set_title(r'$N_x$=100 $\lambda$ = 1.2')
    ax[2, 1].plot(results['100'][2]['x'], results['100'][2]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[2, 1].plot(results['100'][2]['x'], results['100'][2]['u_num'][8, :], 'r--', lw=3)
    ax[2, 1].plot(results['100'][2]['x'], results['100'][2]['u_num'][16, :], 'r--', lw=3)
    ax[2, 1].plot(results['100'][2]['x'], results['100'][2]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[2, 1].plot(results['100'][2]['x'], results['100'][2]['u_ex'][8, :], 'b', lw=2)
    ax[2, 1].plot(results['100'][2]['x'], results['100'][2]['u_ex'][16, :], 'b', lw=2)
    ax[2, 1].legend(loc='upper left')
    ax[2, 1].set_xlabel('x', fontsize=20)


    ax[2, 2].set_title(r'$N_x$=300 $\lambda$ = 1.2')
    ax[2, 2].plot(results['300'][2]['x'], results['300'][2]['u_num'][0, :], 'r--', lw=3, label='Numeric CN')
    ax[2, 2].plot(results['300'][2]['x'], results['300'][2]['u_num'][25, :], 'r--', lw=3)
    ax[2, 2].plot(results['300'][2]['x'], results['300'][2]['u_num'][40, :], 'r--', lw=3)
    ax[2, 2].plot(results['300'][2]['x'], results['300'][2]['u_ex'][0, :], 'b', lw=2, label='Exact')
    ax[2, 2].plot(results['300'][2]['x'], results['300'][2]['u_ex'][25, :], 'b', lw=2)
    ax[2, 2].plot(results['300'][2]['x'], results['300'][2]['u_ex'][40, :], 'b', lw=2)
    ax[2, 2].legend(loc='upper left')
    ax[2, 2].set_xlabel('x', fontsize=20)


    fig, ax = plt.subplots(3, 3)
    plt.rcParams.update({'font.size': 16})
    plt.tick_params(labelsize=14)

    fig.suptitle(r'$L_{2}$ Norm Error of CN scheme applied to Advection Equation', fontsize=22)

    ax[0, 0].set_title(r'$N_x$=50 $\lambda$ = 0.2')
    ax[0, 0].plot(results['50'][0]['err'], 'r-', lw=3)
    ax[0, 0].set_ylabel(r'$||e||^2 _{2}$', fontsize=20)

    ax[0, 1].set_title(r'$N_x$=100 $\lambda$ = 0.2')
    ax[0, 1].plot(results['100'][0]['err'], 'r-', lw=3)

    ax[0, 2].set_title(r'$N_x$=300 $\lambda$ = 0.2')
    ax[0, 2].plot(results['300'][0]['err'], 'r-', lw=3)


    ax[1, 0].set_title(r'$N_x$=50 $\lambda$ = 0.5')
    ax[1, 0].plot(results['50'][1]['err'], 'r-', lw=3)
    ax[1, 0].set_ylabel(r'$||e||^2 _{2}$', fontsize=20)

    ax[1, 1].set_title(r'$N_x$=100 $\lambda$ = 0.5')
    ax[1, 1].plot(results['100'][1]['err'], 'r-', lw=3)

    ax[1, 2].set_title(r'$N_x$=300 $\lambda$ = 0.5')
    ax[1, 2].plot(results['300'][1]['err'], 'r-', lw=3)


    ax[2, 0].set_title(r'$N_x$=50 $\lambda$ = 1.2')
    ax[2, 0].plot(results['50'][2]['err'], 'r-', lw=3)
    ax[2, 0].set_ylabel(r'$||e||^2 _{2}$')
    ax[2, 0].set_xlabel('Sample', fontsize=20)

    ax[2, 1].set_title(r'$N_x$=100 $\lambda$ = 1.2')
    ax[2, 1].plot(results['100'][2]['err'], 'r-', lw=3)
    ax[2, 1].set_ylabel(r'$||e||^2 _{2}$')
    ax[2, 1].set_xlabel('Sample', fontsize=20)

    ax[2, 2].set_title(r'$N_x$=300 $\lambda$ = 1.2')
    ax[2, 2].plot(results['300'][2]['err'], 'r-', lw=3)
    ax[2, 2].set_ylabel(r'$||e||^2 _{2}$')
    ax[2, 2].set_xlabel('Sample', fontsize=20)

#%%

    # -----------------------------------------------
    #               Animation section
    #   (Only activated for nx=300, lambda=0.5)
    # -----------------------------------------------
    x_an = np.linspace(0, 1, 300, endpoint=False)
    hx_an = x_an[1] - x_an[0]
    u_k_vec_an, u_k_vec_exact_an, nt_an = solve_PDE(fun_u, x_an, T, c, lmbda=0.5)

    fig, ax = plt.subplots()
    ax.set_title('u vs x')
    line1, = ax.plot(x_an, u_k_vec_an[0, :], c='red', lw=3, clip_on=False, label='Numerical Solution')
    line2, = ax.plot(x_an, u_k_vec_exact_an[0, :], c='blue', lw=3, clip_on=False, label='Exact Solution')
    ax.legend(loc='upper left')

    def init():
        pass

    def time_stepper(n):
        line1.set_data(x_an, u_k_vec_an[n, :])
        line2.set_data(x_an, u_k_vec_exact_an[n, :])
        return line1, line2

    ani = animation.FuncAnimation(fig, time_stepper,
                                  frames=nt_an, interval=30,
                                  init_func=init)
    ani.save('anim.gif', writer='PillowWriter', fps=30)


