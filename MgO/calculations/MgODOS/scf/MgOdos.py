import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def summed_dos(fnames):
    def DOS_one_file(fname):
        ldos = np.genfromtxt(fname)
        E, s, p, d, f = ldos.T
        return E, s + p + d + f
    energies, DOSes = list(zip(*[DOS_one_file(fname) for fname in fnames]))
    return energies[0], np.sum(DOSes, axis = 0)

ldos_Mg = np.genfromtxt('ldos_Mg.dat')
ldos_O = np.genfromtxt('ldos_O.dat')
E, total_dos = summed_dos(['ldos_Mg.dat', 'ldos_O.dat'])

Ef = -10.859

def make_fermi(mu, kT):
    """make a fermi function"""
    def fermi(energy):
        return 1./(1 + np.exp((energy - mu) / kT))
    return fermi
def cb_occupied(E, dos, kT = 0, Ef = Ef):
    fermi = make_fermi(Ef, kT)
    occupied = dos * fermi(E)
    return E, occupied

def cb_integrate(E, dot, kT = 0, Ef = Ef):
    E, occupied = cb_occupied(E, dot, kT = kT, Ef = Ef)
    i = np.where(E > Ef)[0]
    return np.trapz(occupied[i], x = E[i])

def vb_integrate(E, dot, kT = 0, Ef = Ef):
    E, occupied = cb_occupied(E, dot, kT = kT, Ef = Ef)
    i = np.where(E < Ef)[0]
    return np.trapz(occupied[i], x = E[i])

def integrate(E, dot, kT = 0, Ef = Ef):
    E, occupied = cb_occupied(E, dot, kT = kT, Ef = Ef)
    return np.trapz(occupied, x = E)

def cb_energy(E, dot, kT = 0, Ef = Ef):
    ref = np.dot(*cb_occupied(E, dot, 0, Ef = Ef))
    E, occupied = cb_occupied(E, dot, kT = kT, Ef = Ef)
    return np.dot(E, occupied) - ref

import functools
def make_cost_function(kT):
    def cost_function(ef):
        target_charge = 16.27
        return np.abs(target_charge - integrate(E, total_dos, kT = kT, Ef = ef))
    return cost_function

def corrected_Ef(kT):
    cost_function  = make_cost_function(kT)
    res = minimize(cost_function, Ef, tol = 1e-2)
    return res['x'][0]

# define an interpolation function: temperature as a function of internal energy
from scipy.interpolate import interp1d as interp
temps = np.arange(0., 6.1, 0.1)
energies = [cb_energy(E, total_dos, kT, Ef = ef) for kT, ef in zip(temps, map(corrected_Ef, temps))]
T_of_U = interp(energies, temps)
U_of_T = interp(temps, energies)
