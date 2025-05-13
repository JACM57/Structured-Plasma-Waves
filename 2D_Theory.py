import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Normalized constants
epsilon_0 = 1  
e = 1          
m_e = 1   
c = 1    

# Plasma parameters
n0 = 1
omega_p = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
vth = 0.3

# Define dispersion relation
def omega(k):
    return np.sqrt(1 + (3/2 * k**2 * vth**2)/omega_p**2)

# Wave vector components
kx = 2
ky = 2
k = np.sqrt(kx**2 + ky**2)
khat_x = kx / k
khat_y = ky / k

# Wave amplitudes
n1_amp = 0.1
E1_amp = - n1_amp / k
v1_amp = (omega(k)/k) * n1_amp

# Space and time arrays
t_arr = np.linspace(0, 50, 200)
x_array = np.linspace(0, 10, 200)
y_array = np.linspace(0, 10, 200)
X, Y = np.meshgrid(x_array, y_array)

# Define the fields as functions of (t,x,y)
def density(t, x, y):
    phase = kx * x + ky * y - omega(k) * t
    return n0 + n1_amp * np.cos(phase)

def electric_field_x(t, x, y):
    phase = kx * x + ky * y - omega(k) * t
    E_x = E1_amp * np.sin(phase) * khat_x
    return E_x

def velocity_x(t, x, y):
    phase = kx * x + ky * y - omega(k) * t
    v_x = v1_amp * np.cos(phase) * khat_x
    return v_x

# Create figure and axes
fig, axes = plt.subplots(3, 1, figsize=(8, 16))
plt.subplots_adjust(hspace=0.3)

# 1) Density
im_density = axes[0].imshow(density(0, X, Y),
                            extent=[x_array.min(), x_array.max(), y_array.min(), y_array.max()],
                            origin='lower', cmap="RdBu", interpolation='bilinear')
axes[0].set_title("Density $n$")
axes[0].set_xlabel("$x \, [c/\omega_p]$")
axes[0].set_ylabel("$y \, [c/\omega_p]$")
cbar0 = fig.colorbar(im_density, ax=axes[0], label="$n \, [n_0]$")

# 2) Ex
im_Ex = axes[1].imshow(electric_field_x(0, X, Y),
                       extent=[x_array.min(), x_array.max(), y_array.min(), y_array.max()],
                       origin='lower', cmap="RdBu", interpolation='bilinear')
axes[1].set_title("Electric Field $E_x$")
axes[1].set_xlabel("$x \, [c/\omega_p]$")
axes[1].set_ylabel("$y \, [c/\omega_p]$")
cbar1 = fig.colorbar(im_Ex, ax=axes[1], label="$E_x \,[m_ec\\omega_p/e]$")

# 3) vx
im_vx = axes[2].imshow(velocity_x(0, X, Y),
                       extent=[x_array.min(), x_array.max(), y_array.min(), y_array.max()],
                       origin='lower', cmap="RdBu", interpolation='bilinear')
axes[2].set_title("Velocity $v_x$")
axes[2].set_xlabel("$x \, [c/\omega_p]$")
axes[2].set_ylabel("$y \, [c/\omega_p]$")
cbar2 = fig.colorbar(im_vx, ax=axes[2], label="$v_x \, [c]$")

time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=14)

def update_single_wave(frame):
    ti = t_arr[frame]
    # Update density image
    im_density.set_array(density(ti, X, Y))
    # Update Ex
    im_Ex.set_array(electric_field_x(ti, X, Y))
    # Update vx
    im_vx.set_array(velocity_x(ti, X, Y))
    # Update time text
    time_text.set_text(rf"$t = {ti:.2f} \, [\omega_p^{{-1}}]$")
    return im_density, im_Ex, im_vx, time_text

ani_single = FuncAnimation(fig, update_single_wave, frames=len(t_arr), interval=50, blit=False)
writer = FFMpegWriter(fps=20)
ani_single.save("2D_videos/2D_plasma_wave_single_Ex_vx.mp4", writer=writer)
plt.show()

##############################################################

# Envelope parameters
k_0x = 2 * np.pi
k_0y = 2 * np.pi
sigma_kx = 0.2
sigma_ky = 0.2

# Spatial domain
t2 = np.linspace(0, 20, 50)
x = np.linspace(-15, 15, 300)
y = np.linspace(-15, 15, 300)
X2, Y2 = np.meshgrid(x, y)

# 2D Gaussian in k-space
def G_2D(kx_val, ky_val, k0x, k0y):
    A = 1 / (2 * np.pi * sigma_kx * sigma_ky)
    return A * np.exp(-(((kx_val - k0x)**2) / (2 * sigma_kx**2) +
                        ((ky_val - k0y)**2) / (2 * sigma_ky**2)))

# Define k-space sampling
Delta_x = x[-1] - x[0]
Delta_y = y[-1] - y[0]
Delta_kx = 2 * np.pi / Delta_x
Delta_ky = 2 * np.pi / Delta_y

kx_min = k_0x - 4 * sigma_kx
kx_max = k_0x + 4 * sigma_kx
nx_waves = int((kx_max - kx_min) / Delta_kx)
kx_values = np.linspace(kx_min, kx_max, nx_waves)
kx_values = kx_values[kx_values > 0]

ky_min = k_0y - 4 * sigma_ky
ky_max = k_0y + 4 * sigma_ky
ny_waves = int((ky_max - ky_min) / Delta_ky)
ky_values = np.linspace(ky_min, ky_max, ny_waves)
ky_values = ky_values[ky_values > 0]

# Build initial (t=0) fields
n1_init = np.zeros_like(X2)
E1_init_x = np.zeros_like(X2)
v1_init_x = np.zeros_like(X2)
t0 = t2[0]

for kx_val in kx_values:
    for ky_val in ky_values:
        amp = G_2D(kx_val, ky_val, k_0x, k_0y)
        k_mag = np.sqrt(kx_val**2 + ky_val**2)

        khat_x_val = kx_val / k_mag
        khat_y_val = ky_val / k_mag
        phase = kx_val * X2 + ky_val * Y2 - omega(k_mag) * t0
        n1_init += amp * np.cos(phase)
        E1_init_x += -amp / k_mag * np.sin(phase) * khat_x_val
        v1_init_x += amp * (omega(k_mag)/k_mag) * np.cos(phase) * khat_x_val

# Multiply by the k-space increments
n1_init *= Delta_kx * Delta_ky
E1_init_x *= Delta_kx * Delta_ky
v1_init_x *= Delta_kx * Delta_ky

# Create figure and axes for wavepacket animation
fig2, axes2 = plt.subplots(3, 1, figsize=(8, 16))
plt.subplots_adjust(hspace=0.3)

# 1) Density
im_n = axes2[0].imshow(n1_init, extent=[x.min(), x.max(), y.min(), y.max()],
                       origin='lower', cmap='RdBu', interpolation='bilinear')
axes2[0].set_title("Superposed Density $n_1$")
axes2[0].set_xlabel("$x \, [c/\omega_p]$")
axes2[0].set_ylabel("$y \, [c/\omega_p]$")
fig2.colorbar(im_n, ax=axes2[0], label="$n \, [n_0]$")

# 2) E_x
im_Ex2 = axes2[1].imshow(E1_init_x, extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='RdBu', interpolation='bilinear')
axes2[1].set_title("Superposed Electric Field $E_x$")
axes2[1].set_xlabel("$x \, [c/\omega_p]$")
axes2[1].set_ylabel("$y \, [c/\omega_p]$")
fig2.colorbar(im_Ex2, ax=axes2[1], label="$E_x \,[m_ec\\omega_p/e]$")

# 3) v_x
im_vx2 = axes2[2].imshow(v1_init_x, extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='RdBu', interpolation='bilinear')
axes2[2].set_title("Superposed Velocity $v_x$")
axes2[2].set_xlabel("$x \, [c/\omega_p]$")
axes2[2].set_ylabel("$y \, [c/\omega_p]$")
fig2.colorbar(im_vx2, ax=axes2[2], label="$v_x \, [c]$")

time_text2 = fig2.text(0.5, 0.95, '', ha='center', fontsize=14)

def update_superposed(frame):
    # Accumulate new fields for time t2[frame]
    t_i = t2[frame]
    n1_wave_sum = np.zeros_like(X2)
    E1_wave_sum_x = np.zeros_like(X2)
    v1_wave_sum_x = np.zeros_like(X2)

    for kx_val in kx_values:
        for ky_val in ky_values:
            amp = G_2D(kx_val, ky_val, k_0x, k_0y)
            k_mag = np.sqrt(kx_val**2 + ky_val**2)

            khat_x_val = kx_val / k_mag
            phase = kx_val * X2 + ky_val * Y2 - omega(k_mag) * t_i

            n1_wave_sum += amp * np.cos(phase)
            E1_wave_sum_x += -amp / k_mag * np.sin(phase) * khat_x_val
            v1_wave_sum_x += amp * (omega(k_mag)/k_mag) * np.cos(phase) * khat_x_val

    # Multiply by the k-space increments
    n1_wave_sum *= Delta_kx * Delta_ky
    E1_wave_sum_x *= Delta_kx * Delta_ky
    v1_wave_sum_x *= Delta_kx * Delta_ky

    # Update imshow data
    im_n.set_array(n1_wave_sum)
    im_Ex2.set_array(E1_wave_sum_x)
    im_vx2.set_array(v1_wave_sum_x)

    time_text2.set_text(rf"$t = {t_i:.2f} \, [\omega_p^{{-1}}]$")
    return im_n, im_Ex2, im_vx2, time_text2

ani_superposed = FuncAnimation(fig2, update_superposed, frames=len(t2), init_func=lambda: None, interval=50, blit=False)
writer2 = FFMpegWriter(fps=15)
ani_superposed.save(f"2D_videos/2D_plasma_superposed_Ex_vx_vth={vth}.mp4", writer=writer2)
plt.show()

