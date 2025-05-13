import matplotlib
#matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.signal import hilbert

## Enable LaTeX for text rendering
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#
## Set preamble to use necessary packages
#plt.rcParams['text.latex.preamble'] = r'''
#\usepackage{amsmath}
#\usepackage{mathrsfs}
#\DeclareSymbolFontAlphabet{\mathrsfs}{rsfs}
#'''

# Normalized constants
epsilon_0 = 1  # Permittivity of free space
e = 1  # Electron charge
m_e = 1  # Electron mass
c = 1 # Speed of light

# Normalized plasma parameters (exepect w_p)
n0 = 1  # Equilibrium electron density 
omega_p = np.sqrt(n0*e**2 / (epsilon_0*m_e))  # Plasma frequency [rad/s]
vth = 0.3
#kx = 15 # Propagation constant in x
#ky = 15 # Propagation constant in y
#k = np.sqrt(kx**2 + ky**2)
kx = 2
k = kx

# Normalized dispersion relation (with w_p)
def omega(x):
    return np.sqrt(1 + (3/2*x**2*vth**2)/omega_p**2)

# Space and time arrays
t_array = np.linspace(0, 4*2*np.pi/omega_p, 200)  # Four full cycles of the plasma wave
x_array = np.linspace(0, 10, 500)

# Density [normalized]
n0 = 1
n1 = 0.1
def n(t, x):
    return n0 + n1*np.cos(k*x - omega(k)*t)

# Electric Field [normalized]
E0 = 0
E1 = -n1/k
def E(t, x):
    return E0 + E1*np.sin(k*x - omega(k)*t)

# Velocity [normalized]
v0 = 0
v1 = (omega(k)/k)*n1
def v(t, x):
    return v0 + v1*np.cos(k*x - omega(k)*t)

# Current Density [normalized]
j0 = 0 #(v0=0)
j1 = -v1

def j(t, x):
    return j0 + j1*np.cos(k*x - omega(k)*t)


''' Create n waves with a Gaussian distribution around a central value k_0 in 1D '''

k_0 = 2*np.pi
sigma_k = 0.2

x = np.linspace(-50, 50, 1000)
t = np.linspace(0, 50, 500)
Delta_x = x[-1]-x[0]  #200
Delta_k = 2*np.pi/Delta_x  #0.0314159

k_f = k_0+4*sigma_k
k_i = k_0-4*sigma_k
n_waves = int((k_f-k_i)/Delta_k)
k_values = np.linspace(k_i, k_f, n_waves)
k_values = k_values[k_values > 0]  # Filter out negative values

def n1_w(k): # n1 = A*exp[-alpha*(k-k0)^2]
    A = 1 / (np.sqrt(2*np.pi*sigma_k**2))
    alpha = 1 / (2*sigma_k**2)
    return A * np.exp(-alpha*(k-k_0)**2)

def n2(k): # Sinc envelope
    delta = 2
    return np.sinc((k-k_0)/delta)

def n3(k): # Lorentzian envelope
    gamma = 1
    return 1/((k-k_0)**2+gamma**2)


# Plot n1(k) vs k
k_array = np.linspace(Delta_k, 100+Delta_k, 1000)
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(k_array, n1_w(k_array), lw=2)
ax2.set_xlim(k_0-5, k_0+5)
ax2.set_xlabel('$k$', fontsize=12)
ax2.set_ylabel('$n_1(k)$', fontsize=12)
plt.show()

# Scatter plot of n1
fig3, ax3 = plt.subplots()
ax3.scatter(k_values, n1_w(k_values), color='b')
ax3.set_xlabel('k')
ax3.set_ylabel('n1(k)')
plt.show()

# Plot the motion of the wave packet (n1)
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(-50, 50)
ax.set_ylim(-1, 1)
ax.set_title('Motion of the resulting n1 wave')
ax.set_xlabel('x')
ax.set_ylabel('Amplitude of the resulting n1 wave')

time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update_n(frame):
    wave_sum = np.zeros_like(x)
    t_i = t[frame]
    for k in k_values:
        wave_sum += n1_w(k)*np.cos(k*x - omega(k)*t_i)
    wave_sum = wave_sum * Delta_k
    
    line.set_data(x, wave_sum)
    time_text.set_text(f'Time: {t_i:.2f}')
    return line, time_text


ani = FuncAnimation(fig, update_n, frames=len(t), init_func=init, blit=True)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("superposed_waves_n.mp4", writer=writer)
plt.show()

# Plot the motion of the wave packets (n1, E1, v1) in separate subplots
fig, axes = plt.subplots(nrows=3, figsize=(10, 8), sharex=True)
line_n, = axes[0].plot([], [], lw=2, color = 'b')
line_E, = axes[1].plot([], [], lw=2, color = 'orange')
line_v, = axes[2].plot([], [], lw=2, color = 'g')

axes[0].set_title('Superposed $n_1$ Wave', fontsize=18)
axes[0].set_xlim(-50, 50)
axes[0].set_ylim(-1, 1)
axes[0].set_ylabel('$n_1 \, [n_0]$', fontsize=18)

axes[1].set_title('Superposed $E_1$ Wave', fontsize=18)
axes[1].set_xlim(-50, 50)
axes[1].set_ylim(-1, 1)
axes[1].set_ylabel('$E_1 \, [m_ec\omega_p/e]$', fontsize=18)

axes[2].set_title('Superposed $v_1$ Wave', fontsize=18)
axes[2].set_xlim(-50, 50)
axes[2].set_ylim(-1, 1)
axes[2].set_ylabel('$v_1 \, [c]$', fontsize=18)

axes[2].set_xlabel('$x \, [c/\omega_p]$', fontsize=18)

# Increasing tick label font sizes
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=16)

#time_text = axes[0].text(0.05, 0.9, '', transform=axes[0].transAxes)
#time_text = fig.text(0.5, 0.95, '', ha='center')
time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=18)

def init():
    line_n.set_data([], [])
    line_E.set_data([], [])
    line_v.set_data([], [])
    time_text.set_text('')
    return line_n, line_E, line_v, time_text

def update(frame):
    wave_sum_n = np.zeros_like(x)
    wave_sum_E = np.zeros_like(x)
    wave_sum_v = np.zeros_like(x)
    t_i = t[frame]
    for k in k_values:
        # n
        wave_sum_n += n1_w(k)*np.cos(k*x - omega(k)*t_i)
        # E
        wave_sum_E += -(n1_w(k))/k*np.sin(k*x - omega(k)*t_i)
        # v
        wave_sum_v += (omega(k)/k)*n1_w(k)*np.cos(k*x - omega(k)*t_i)
    wave_sum_n *= Delta_k
    wave_sum_E *= Delta_k
    wave_sum_v *= Delta_k

    # Update the lines in each subplot
    line_n.set_data(x, wave_sum_n)
    line_E.set_data(x, wave_sum_E)
    line_v.set_data(x, wave_sum_v)

    time_text.set_text(f'$ t = {t_i:.2f} \, [\\omega_p^{{-1}}]$')
    return line_n, line_E, line_v, time_text

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=False)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("superposed_waves_n_E_v_subplots.mp4", writer=writer)

# Save snapshots at omega_p t = 0 and omega_p t = 70
frame_t0 = np.argmin(np.abs(t - 0))
update(frame_t0)  # Update the plot to t = 0
plt.savefig("superposed_waves_at_t0.pdf", format="pdf")

frame_t70 = np.argmin(np.abs(t - 70))
update(frame_t70)  # Update the plot to t = 70
plt.savefig("superposed_waves_at_t70.pdf", format="pdf")

plt.show()

''' Waterfall plot to compare theoretical and numerical group velocities '''

x = np.linspace(-50, 50, 500)
t = np.linspace(0, 100, 300)
waves = np.zeros((len(t), len(x))) # Each row is a wave at a given time

# Theoretical group velocity
def group_velocity(k):
    return (3*k*vth**2) / (2*omega(k)*omega_p**2)

# Calculate the phase front position over time
v_phase = omega(k_0)/k_0
phase_front_positions = v_phase*t  # x = v_phase*t

# Calculate the group velocity front positions over time
vg_theoretical = group_velocity(k_0)
group_velocity_front = vg_theoretical*t  # x = v_g*t

# Compute the wave packet for each time step and store it in 'waves'
for i, t_i in enumerate(t):
    for k in k_values:
        waves[i, :] += n1_w(k) * np.cos(k * x - omega(k) * t_i) # This calculates one entire row
    waves[i, :] *= Delta_k

fig, ax = plt.subplots(figsize=(10, 6))
c = ax.pcolormesh(x, t, waves, shading='auto', cmap='seismic')
ax.plot(phase_front_positions, t, color='yellow', linestyle='-', linewidth=3, label=r'$v_{\phi}$')
ax.plot(group_velocity_front, t, color='cyan', linestyle='--', linewidth=3, label=r'$v_{g}$')
ax.set_title('Waterfall plot of the $n_1$ wave', fontsize=19, pad = 20)
ax.set_xlabel('$x \, [c/\omega_p]$', fontsize=18)
ax.set_ylabel('$t \, [\omega_p^{-1}]$', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(loc='upper left', fontsize=15)

cb = fig.colorbar(c, ax=ax)
cb.set_label(label='$n_1 \, [n_0]$', fontsize=18)
cb.ax.tick_params(labelsize=18)

plt.savefig(f'Waterfall_plot_vth={vth}.pdf', bbox_inches='tight')
plt.show()



x_positions = []
times = t

for i, wave in enumerate(waves):
    # Find the index of the maximum amplitude at each time step
    max_idx = np.argmax(wave) # Finds the index of the maximum amplitude
    x_max = x[max_idx] # Retrieves the position x corresponding to that maximum amplitude
    x_positions.append(x_max)

envelopes = np.abs(hilbert(waves, axis=1))  # Get envelope along the x-axis for each time step

# Find the peak position of the envelope at each time step
x_positions = []
for envelope in envelopes:
    max_idx = np.argmax(envelope)  # Index of the maximum of the envelope
    x_max = x[max_idx]  # Corresponding position x of the maximum
    x_positions.append(x_max)

# Convert lists to arrays
x_positions = np.array(x_positions)
times = np.array(t)

# Linear fit to the peak positions to find the group velocity
def linear_fit(t, vg, x0):
    return vg*t + x0

fit_param, pcov = curve_fit(linear_fit, times, x_positions)
vg_experimental, x0 = fit_param

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(times, x_positions, 'o', label='Envelope Peak Position')
plt.plot(times, linear_fit(times, *fit_param), 'r-', label=f'Linear Fit: $v_g$ = {vg_experimental:.4f}')
plt.xlabel('Time')
plt.ylabel('Position of Envelope Peak')
plt.title('Group Velocity using Hilbert Transform')
plt.legend()
plt.savefig(f'Group_velocity_fit_vth={vth}.png', format='png', bbox_inches='tight')
plt.show()

print(f"Theoretical group velocity: {vg_theoretical}")
print(f"Experimental group velocity: {vg_experimental}")
dif = abs(vg_theoretical - vg_experimental)
print(f"Difference: {dif}")

