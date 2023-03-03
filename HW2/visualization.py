
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

files = os.listdir('results/')
temp_file = {}

for file in files:
    try:
        with open('results/' + file, 'rb') as inp:
            file_name = file[:-4]
            temp_file[file_name] = pickle.load(inp)
    except NameError:
        print('No files found')

for k, v in temp_file.items():
    globals()[k] = v
    # del temp_file, file_name, file

#%%

fig0, ax = plt.subplots(2, 2)
plt.rcParams.update({'font.size': 16})
plt.tick_params(labelsize=14)
plt.tight_layout()


ax[0, 0].set_title('Rarefaction case - T = 1, hx = {:.4f}'.format(Rarefaction_LF_T1_nx128['hx']))
ax[0, 0].plot(Rarefaction_LF_T1_nx128['xf'], Rarefaction_LF_T1_nx128['u0'], 'k', label='Exact')
ax[0, 0].plot(Rarefaction_LF_T1_nx128['x'], Rarefaction_LF_T1_nx128['u'], 'r--', lw=2.5, label='LF')
ax[0, 0].plot(Rarefaction_Gudonov_T1_nx128['x'], Rarefaction_Gudonov_T1_nx128['u'], 'b', ls='dotted', lw=2.5, label='Gudonov')
ax[0, 0].plot(Rarefaction_RK2_T1_nx128['x'], Rarefaction_RK2_T1_nx128['u'], 'g--', lw=2.5, label='RK 2 (minmod)')
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('u(x, T)')


ax[0, 1].set_title('Rarefaction case - T = 2, hx = {:.4f}'.format(Rarefaction_LF_T1_nx128['hx']))
ax[0, 1].plot(Rarefaction_LF_T2_nx128['xf'], Rarefaction_LF_T2_nx128['u0'], 'k', label='Exact')
ax[0, 1].plot(Rarefaction_LF_T2_nx128['x'], Rarefaction_LF_T2_nx128['u'], 'r--', lw=2.5, label='LF')
ax[0, 1].plot(Rarefaction_Gudonov_T2_nx128['x'], Rarefaction_Gudonov_T2_nx128['u'], 'b', ls='dotted', lw=2.5, label='Gudonov')
ax[0, 1].plot(Rarefaction_RK2_T2_nx128['x'], Rarefaction_RK2_T2_nx128['u'], 'g--', lw=2.5, label='RK 2 (minmod)')
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('u(x, T)')
ax[0, 1].legend(loc='best')


ax[1, 0].set_title('Shock case - T = 1, hx = {:.4f}'.format(Rarefaction_LF_T1_nx128['hx']))
ax[1, 0].plot(Shock_LF_T1_nx128['xf'], Shock_LF_T1_nx128['u0'], 'k', label='Exact')
ax[1, 0].plot(Shock_LF_T1_nx128['x'], Shock_LF_T1_nx128['u'], 'r--', lw=2.5, label='LF')
ax[1, 0].plot(Shock_Gudonov_T1_nx128['x'], Shock_Gudonov_T1_nx128['u'], 'b', ls='dotted', lw=2.5, label='Gudonov')
ax[1, 0].plot(Shock_RK2_T1_nx128['x'], Shock_RK2_T1_nx128['u'], 'g--', lw=2.5, label='RK 2 (minmod)')
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel('u(x, T)')


ax[1, 1].set_title('Shock case - T = 2, hx = {:.4f}'.format(Rarefaction_LF_T1_nx128['hx']))
ax[1, 1].plot(Shock_LF_T2_nx128['xf'], Shock_LF_T2_nx128['u0'], 'k', label='Exact')
ax[1, 1].plot(Shock_LF_T2_nx128['x'], Shock_LF_T2_nx128['u'], 'r--', lw=2.5, label='LF')
ax[1, 1].plot(Shock_Gudonov_T2_nx128['x'], Shock_Gudonov_T2_nx128['u'], 'b', ls='dotted', lw=2.5, label='Gudonov')
ax[1, 1].plot(Shock_RK2_T2_nx128['x'], Shock_RK2_T2_nx128['u'], 'g--', lw=2.5, label='RK 2 (minmod)')
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('u(x, T)')



fig1, ax = plt.subplots(2, 2)
plt.rcParams.update({'font.size': 16})
plt.tick_params(labelsize=14)
plt.tight_layout()


ax[0, 0].set_title('Rarefaction case - T = 1, hx = {:.3f}'.format(Rarefaction_LF_T1_nx1000['hx']))
ax[0, 0].plot(Rarefaction_LF_T1_nx1000['xf'], Rarefaction_LF_T1_nx1000['u0'], 'k', label='Exact')
ax[0, 0].plot(Rarefaction_LF_T1_nx1000['x'], Rarefaction_LF_T1_nx1000['u'], 'r--', lw=2.5, label='LF')
ax[0, 0].plot(Rarefaction_Gudonov_T1_nx1000['x'], Rarefaction_Gudonov_T1_nx1000['u'], 'b', ls='dotted', lw=2.5, label='Gudonov')
ax[0, 0].plot(Rarefaction_RK2_T1_nx1000['x'], Rarefaction_RK2_T1_nx1000['u'], 'g--', lw=2.5, label='RK 2 (minmod)')
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('u(x, T)')


ax[0, 1].set_title('Rarefaction case - T = 2, hx = {:.3f}'.format(Rarefaction_LF_T1_nx1000['hx']))
ax[0, 1].plot(Rarefaction_LF_T2_nx1000['xf'], Rarefaction_LF_T2_nx1000['u0'], 'k', label='Exact')
ax[0, 1].plot(Rarefaction_LF_T2_nx1000['x'], Rarefaction_LF_T2_nx1000['u'], 'r--', lw=2.5, label='LF')
ax[0, 1].plot(Rarefaction_Gudonov_T2_nx1000['x'], Rarefaction_Gudonov_T2_nx1000['u'], 'b', ls='dotted', lw=2.5, label='Gudonov')
ax[0, 1].plot(Rarefaction_RK2_T2_nx1000['x'], Rarefaction_RK2_T2_nx1000['u'], 'g--', lw=2.5, label='RK 2 (minmod)')
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('u(x, T)')
ax[0, 1].legend(loc='best')


ax[1, 0].set_title('Shock case - T = 1, hx = {:.3f}'.format(Rarefaction_LF_T1_nx1000['hx']))
ax[1, 0].plot(Shock_LF_T1_nx1000['xf'], Shock_LF_T1_nx1000['u0'], 'k', label='Exact')
ax[1, 0].plot(Shock_LF_T1_nx1000['x'], Shock_LF_T1_nx1000['u'], 'r--', lw=2.5, label='LF')
ax[1, 0].plot(Shock_Gudonov_T1_nx1000['x'], Shock_Gudonov_T1_nx1000['u'], 'b', ls='dotted', lw=2.5, label='Gudonov')
ax[1, 0].plot(Shock_RK2_T1_nx1000['x'], Shock_RK2_T1_nx1000['u'], 'g--', lw=2.5, label='RK 2 (minmod)')
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel('u(x, T)')


ax[1, 1].set_title('Shock case - T = 2, hx = {:.3f}'.format(Rarefaction_LF_T1_nx1000['hx']))
ax[1, 1].plot(Shock_LF_T2_nx1000['xf'], Shock_LF_T2_nx1000['u0'], 'k', label='Exact')
ax[1, 1].plot(Shock_LF_T2_nx1000['x'], Shock_LF_T2_nx1000['u'], 'r--', lw=2.5, label='LF')
ax[1, 1].plot(Shock_Gudonov_T2_nx1000['x'], Shock_Gudonov_T2_nx1000['u'], 'b', ls='dotted', lw=2.5, label='Gudonov')
ax[1, 1].plot(Shock_RK2_T2_nx1000['x'], Shock_RK2_T2_nx1000['u'], 'g--', lw=2.5, label='RK 2 (minmod)')
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('u(x, T)')


fig2, ax = plt.subplots()
plt.rcParams.update({'font.size': 16})
plt.tick_params(labelsize=14)
plt.tight_layout()
ax.plot([1,3], [1,2], 'k', label='Exact')
ax.plot(Rarefaction_LF_T1_nx128['x_filt'], Rarefaction_LF_T1_nx128['u_filt'], 'r--', label='LF')
ax.plot(Rarefaction_Gudonov_T1_nx128['x_filt'], Rarefaction_Gudonov_T1_nx128['u_filt'], 'b', ls='dotted', label='Gudonov')
ax.plot(Rarefaction_RK2_T1_nx128['x_filt'], Rarefaction_RK2_T1_nx128['u_filt'], 'g--', label='RK2 (minmod)')
