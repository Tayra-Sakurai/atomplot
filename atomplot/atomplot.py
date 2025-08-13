from math import pi, sqrt
from typing import Any, NoReturn
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.special import assoc_laguerre, factorial, sph_legendre_p

# Atom Number
Z = 1
# Borh Radius
a0 = 5.291772108e-11  # in meters

def R (r: float|np.floating[Any]|np.typing.NDArray[np.floating[Any]], n: int, l:int) -> float|np.floating[Any]|np.typing.NDArray[np.floating[Any]]:
    """Caluclates the radial wave function for a hydrogen-like atom.

    Arguments:
        r: The radial distance from the nucleus.
        n: The principal quantum number.
        l: The azimuthal quantum number.
    Returns:
        The value of the radial wave function at distance r.
    """
    # Arguments validation
    if not isinstance(r, (float, np.floating, np.ndarray)):
        raise TypeError("r must be a float or np.floating")
    elif not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    elif not isinstance(l, int) or l < 0 or l >= n:
        raise ValueError("l must be a non-negative integer less than n")
    # Constants
    global Z, a0
    # Calculate the radial wave function
    rho = (2 * Z * r) / (n * a0)
    func_base = np.exp(-rho / 2) * (rho**l) * assoc_laguerre(rho, ( n - l - 1 ), (2*l + 1))
    normalization = sqrt(
        (2 * Z / (n * a0))**3 * factorial(n - l - 1) /
        (2 * n**2 * factorial(n + l)))
    return normalization * func_base

def Y (
    theta: float|np.floating[Any]|np.typing.NDArray[np.floating[Any]],
    phi: float|np.floating[Any]|np.typing.NDArray[np.floating[Any]],
    l: int,
    m: int
) -> float|np.floating[Any]|np.typing.NDArray[np.floating[Any]]:
    """Calculates the spherical harmonic function for given angles and quantum numbers.
    Arguments:
        theta: The polar angle in radians.
        phi: The azimuthal angle in radians.
        l: The azimuthal quantum number.
        m: The magnetic quantum number.
    Returns:
        The value of the spherical harmonic function at the given angles.
    """
    # Constants
    global Z, a0
    # Arguments validation
    if not isinstance(theta, (float, np.floating, np.ndarray)):
        raise TypeError("theta must be a float or np.floating")
    elif not isinstance(phi, (float, np.floating, np.ndarray)):
        raise TypeError("phi must be a float or np.floating")
    elif not isinstance(l, int) or l < 0:
        raise ValueError("l must be a non-negative integer")
    elif not isinstance(m, int) or abs(m) > l:
        raise ValueError("m must be an integer such that |m| <= l")
    # Absolute value of m
    abs_m = abs(m)
    # Calculate Theta(theta)
    Theta = (-1)**((m + abs_m)/2) * sqrt(
        (2 * l + 1) * factorial(l - abs_m) / (2 * factorial(l + abs_m))) * sph_legendre_p(l, m, theta)
    # Calculate the azimuthal part
    azimuthal_part = np.exp(1j * m * phi)
    azimuthal_part /= sqrt(2 * pi)
    # Combine the parts to get the spherical harmonic function
    Y_value = Theta * azimuthal_part
    return Y_value

def points_radial_wave_function(
    n: int,
    l: int,
    r_max: float = 5.0
) -> tuple[np.typing.NDArray[np.floating[Any]],
           np.typing.NDArray[np.floating[Any]]]:
    """Plots the radial wave function for a hydrogen-like atom.
    Arguments:
        n: The principal quantum number.
        l: The azimuthal quantum number.
        r_max: The maximum radial distance to plot.
    """
    global Z, a0
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(l, int) or l < 0 or l >= n:
        raise ValueError("l must be a non-negative integer less than n")

    # Generate radial distances
    r = np.linspace(0, r_max, 1000)
    # Calculate the radial wave function
    R_values = R(a0 * r, n, l)
    return r, R_values

def plot_radial_distribution(
    n: int,
    l: int,
    r_max: float = 5.0
) -> NoReturn:
    """Plots the radial distribution function for a hydrogen-like atom.
    Arguments:
        n: The principal quantum number.
        l: The azimuthal quantum number.
        r_max: The maximum radial distance to plot.
    """
    r, R_values = points_radial_wave_function(n, l, r_max)
    # Calculate the radial distribution function
    radial_distribution = (R_values**2) * (r**2)

    plt.figure(figsize=(8, 6))
    plt.plot(r, radial_distribution, label=f"n={n}, l={l}")
    plt.title("Radial Distribution Function")
    plt.xlabel("Radial Distance (m)")
    plt.ylabel("Radial Distribution")
    plt.grid()
    plt.legend()
    plt.show()

def integrate_R_dist (
    r: np.typing.NDArray[np.floating[Any]],
    n: int,
    l: int) -> np.typing.NDArray[np.floating[Any]]:
    """Integrates the radial distribution function over the ranges from r=0 to the given r value.
    Arguments:
        r: The radial distances.
        n: The principal quantum number.
        l: The azimuthal quantum number.
    Returns:
        The integrals of the radial distribution function.
    """
    integ = [
        integrate.quad(
            lambda r_val: (R(r_val, n, l)**2) * (r_val**2),
            0,
            r_val
        )[0] for r_val in r
    ]
    return np.array(integ)

def plot_atomic_orbitals(
    n: int,
    l: int,
    m_abs: int = 0) -> NoReturn:
    """Plots the atomic orbitals for a hydrogen-like atom.
    Arguments:
        n: The principal quantum number.
        l: The azimuthal quantum number.
    """
    global Y
    r, R_values = points_radial_wave_function(n, l)
    dist_int = (integrate_R_dist(a0 * r, n, l) - 0.95)**2
    # Find the suitable radius for plotting
    radius_plot = r[dist_int.argmin()]
    # Create a meshgrid for spherical coordinates
    theta = np.linspace(0, pi, 128)
    phi = np.linspace(0, 2 * pi, 128)
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the spherical coordinates
    Y_Values1 = Y(theta, phi, l, -m_abs)
    Y_Values2 = Y(theta, phi, l, m_abs) if m_abs != 0 else 0
    # Calculate Ys
    Y_Val1 = Y_Values1 + Y_Values2
    Y_Val2 = Y_Values1 - Y_Values2
    # Calculate the Cartesian coordinates
    X = radius_plot * np.sin(theta) * np.cos(phi) * Y_Val1**2
    Y = radius_plot * np.sin(theta) * np.sin(phi) * Y_Val1**2
    Z = radius_plot * np.cos(theta) * Y_Val1**2 * (phi**0)
    # Plot the atomic orbital
    fig = plt.figure(figsize=(10, 6))
    if m_abs != 0:
        ax1, ax2 = fig.subplots(ncols=2, subplot_kw={'projection':'3d'})
        ax1.scatter(X.real, Y.real, Z.real)
        X = radius_plot * np.sin(theta) * np.cos(phi) * Y_Val2**2
        Y = radius_plot * np.sin(theta) * np.sin(phi) * Y_Val2**2
        Z = radius_plot * np.cos(theta) * Y_Val2**2 * (phi**0)
        ax2.scatter(X.real, Y.real, Z.real)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.set_zlabel('Z-axis')
        ax1.set_title(f"Atomic Orbital for n={n}, l={l}, m={-m_abs}")
        ax2.set_aspect('equal')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.set_zlabel('Z-axis')
        ax2.set_title(f"Atomic Orbital for n={n}, l={l}, m={m_abs}")
    else:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X.real, Y.real, Z.real)
        ax.set_aspect('equal')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f"Atomic Orbital for n={n}, l={l}, m={m_abs}")
    plt.show()

# Example usage
plot_atomic_orbitals(int(input("n=")),int(input('l=')),int(input('|m_l|=')))
