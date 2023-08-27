import FDM as FDM
import numpy as np
import matplotlib.pyplot as plt
from fdm import FDM
from scipy.interpolate import interp1d

# Define constants
De = 1.0  # Diffusion coefficient of light
Dn = 1.0  # Diffusion coefficient of matter
Ge = 1.0  # Nonlinear refractive index coefficient
Gn = 1.0  # Nonlinear absorption coefficient
alpha = 0.1  # Absorption coefficient
n_th = 1.0  # Thermal equilibrium refractive index
kappa = 0.1  # Time delay coefficient
tau_in = 0.1  # Time delay
J = 1.0  # Injection current density
e = 1.0  # Elementary charge
d = 1.0  # Material layer thickness
tau_s = 0.1  # Material lifetime
n0 = 1.0  # Material refractive index
w0 = 1.0  # Angular frequency

# Define grid and initial conditions
L = 10.0  # Length of spatial domain
T = 10.0  # Length of time domain
nx = 100  # Number of grid points in x direction
nt = 1000  # Number of time steps
dx = L/nx  # Spatial step size
dt = T/nt  # Time step size
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
E = np.zeros((nx, nt))
n = np.zeros((nx, nt))
E[:, 0] = np.sin(np.pi*x/L)
n[:, 0] = n0

# Define finite difference scheme
def laplacian(Z):
    if np.ndim(Z) == 1:
        # 1D input
        Ztop = Z[:-2]
        Zcenter = Z[1:-1]
        Zbottom = Z[2:]
        return (Ztop + Zbottom - 2*Zcenter) / dx**2
    elif np.ndim(Z) == 2:
        # 2D input
        Ztop = Z[0:-2,1:-1]
        Zleft = Z[1:-1,0:-2]
        Zbottom = Z[2:,1:-1]
        Zright = Z[1:-1,2:]
        Zcenter = Z[1:-1,1:-1]
        return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2
    else:
        raise ValueError("Input must be 1D or 2D array")

for i in range(1, nt):
    # Solve for E
    E_xx = laplacian(E[:, i-1])
    interp_func = interp1d(t, np.vstack((E[0, :], E, E[-1, :])), kind='linear', axis=0, bounds_error=False, fill_value=0.0)
    E_tau = interp_func(t[i]-tau_in)[1:-1,:]
    E_tau_exp = np.exp(1j*w0*tau_in)*E_tau
    n_eff = n[:, i-1] + Ge*np.abs(E[:, i-1])**2 - Gn*(n[:, i-1]-n0)
    E_tt = 1j*De*E_xx + (1-1j*alpha)*Ge*(n_eff-n_th)*E[:, i-1] + kappa/tau_in*E_tau_exp
    E[:, i] = 2*E[:, i-1] - E[:, i-2] + dt**2 * E_tt
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import integrate


    # Parameters
    De = 1
    alpha = 0.5
    Gn = 1
    nth = 1
    kappa = 1
    tau_in = 1
    w0 = 1
    Dn = 1
    J = 1
    ed = 1
    tau_s = 1
    n0 = 1

    # Grid
    Lx = 10  # Spatial length
    Nx = 100  # Number of spatial points
    dx = Lx / (Nx - 1)  # Spatial step
    x = np.linspace(0, Lx, Nx)  # Spatial grid

    Lt = 10  # Total time
    Nt = 1000  # Number of time steps
    dt = Lt / Nt  # Time step
    #Define
    #initial
    #conditions:
    # Initial conditions
    E0 = np.zeros(Nx, dtype=complex)
    n0 = np.zeros(Nx)
    #Define
    #functions
    for the FDM:
        def d2_dx2(arr, dx):
            return (np.roll(arr, 1) - 2 * arr + np.roll(arr, -1)) / dx ** 2


    def update_E(E, n, dx, dt):
        d2E_dx2 = d2_dx2(E, dx)
        E_updated = E + dt * (1j * De * d2E_dx2 + 0.5 * (1 - 1j * alpha) * Gn * (n - nth) * E)
        return E_updated


    def update_n(E, n, dx, dt):
        d2n_dx2 = d2_dx2(n, dx)
        n_updated = n + dt * (Dn * d2n_dx2 + J / (ed * dx) - n / tau_s - Gn * (n - n0) * np.abs(E) ** 2)
        return n_updated


    #Time - stepping
   # loop:
    E = E0.copy()
    n = n0.copy()
    E_history = [E0]
    n_history = [n0]

    for t in range(Nt):
        E = update_E(E, n, dx, dt)
        n = update_n(E, n, dx, dt)
        E_history.append(E)
        n_history.append(n)
    #Plot
   # the
    #results:
    # Choose a time step to display the results
    time_step = 500

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(x, np.abs(E_history[time_step]) ** 2)
    plt.xlabel('x')
    plt.ylabel('|E|^2')
    plt.title('Intensity at t = {}'.format(time_step * dt))

    plt.subplot(2, 1, 2)
    plt.plot(x, n_history[time_step])
    plt.xlabel('x')
    plt.ylabel('n')
    plt.title('n at t = {}'.format(time_step * dt))

    plt.tight_layout()
    plt.show()
    # Solve for n
    n_xx = laplacian(n[:, i-1])
    n_tt = Dn*n_xx + J/(e*d) - n[:, i-1]/tau_s - Gn*(n[:, i-1]-n0)*np.abs(E[:, i])**2
    n[:, i] = 2*n[:, i-1] - n[:, i-2] + dt**2 * n_tt

# Plot results
fig, axs = plt.subplots(ncols=2)
axs[0].imshow(E, extent=[0, T, 0, L])
axs[1].imshow(n, extent=[0, T, 0, L])
plt.show()
