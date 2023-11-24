import os
import logging

import math
import numpy as np

import jax
import jax.numpy as jnp


from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

#matplotlib inline
import matplotlib.pyplot as plt

print(f"Using Device: {jax.devices()}")

# Initialize logging and create missing directories
log_file = "output.txt"
fig = "fig/"

if not os.path.exists(fig):
    os.makedirs(fig)

if not os.path.exists(f"{fig}wake/"):
    os.makedirs(f"{fig}wake/")

if not os.path.exists(f"{fig}velocity/"):
    os.makedirs(f"{fig}velocity/")

logging.basicConfig(filename=log_file, filemode="w",
                    force=True, level=logging.INFO, format="%(message)s")

class g:
    tau = 0
    mplot = 1
    vplot = 1
    eps = 0.5e-6
    wplot = 1
    zavoid = 0
    mpath = 0
    delta = 0
    svCont = 0
    wvCont = 0
    ivCont = 0
    vpFreq = 1
    vfplot = 1

def nd_data(l_, phiT, phiB, c_, x_, y_, a_, U_, V_, T_):
    dT_ = l_ * np.sin(phiT)
    dB_ = l_ * np.sin(-phiB)
    d_ = dT_ + dB_
    e_ = dT_ - dB_
    e = e_ / d_
    c = c_ / d_
    a = a_ / d_
    x = x_ / d_
    y = y_ / d_
    t_ = T_ / 2.0
    v_ = d_ / t_
    U = U_ / v_
    V = V_ / v_

    return v_, t_, d_, e, c, x, y, a, U, V


def in_data(l_,
            phiT_,
            phiB_,
            c_,
            x_,
            y_,
            a_,
            beta_,
            f_,
            gMax_,
            U_,
            V_):
    T_ = 1.0 / f_

    fac = np.pi / 180.0
    phiT = fac * phiT_
    phiB = fac * phiB_
    beta = fac * beta_
    gMax = fac * gMax_

    v_, t_, d_, e, c, x, y, a, U, V = \
        nd_data(l_, phiT, phiB, c_, x_, y_, a_, U_, V_, T_)

    return v_, t_, d_, e, c, x, y, a, beta, gMax, U, V


def mesh_r(c, x, y, n, m):
    a = 0.5 * c  # half chord length

    f = CubicSpline(x, y)
    df = f.derivative(nu=1)

    s = [0]

    for i in range(n - 1):
        ds = quad(lambda z: np.sqrt(1 + df(z) ** 2), x[i], x[i+1])
        # Get the first value, cross-checked with matlab code for validation.
        s.append(s[i] + ds[0])

    s = np.array(s)

    gcalc = CubicSpline(s, x)
    dS = s[n - 1] / (m - 1)

    xv = np.zeros((m + 4))
    xv[0] = -a
    xv[1] = gcalc(dS * 0.25)
    xv[2] = gcalc(dS * 0.5)

    for i in range(2, m):
        xv[i + 1] = gcalc(dS * (i - 1))

    xv[m + 1] = gcalc(dS * (m - 1 - 0.5))
    xv[m + 2] = gcalc(dS * (m - 1 - 0.25))
    xv[m + 3] = a

    yv = f(xv)

    xc = np.zeros((m + 3))
    xc[0] = gcalc(dS * 0.125)
    xc[1] = gcalc(dS * 0.375)
    xc[2] = gcalc(dS * 0.75)

    for i in range(2, m - 1):
        xc[i + 1] = gcalc(dS * (i - 0.5))

    xc[m] = gcalc(dS * (m - 1 - 0.75))
    xc[m + 1] = gcalc(dS * (m - 1 - 0.375))
    xc[m + 2] = gcalc(dS * (m - 1 - 0.125))

    yc = df(xc)
    dfc = df(xc)

    mNew = m + 4

    return xv, yv, xc, yc, dfc, mNew


def matrix_coef(xv, yv, xc, yc, dfc, m):
    denom = np.sqrt(1 + dfc ** 2)
    nx = -dfc / denom
    ny = 1.0 / denom
    nc = nx + 1j * ny

    zeta = xc + 1j * yc
    zeta0 = xv + 1j * yv

    MVN = np.imag((((1.0 / (np.expand_dims(zeta, 0).transpose() - zeta0)))
                   * nc.reshape((nc.size, 1))) / (2.0 * np.pi))
    MVN = np.append(MVN, np.ones(MVN.shape[1])).reshape((m, m))

    return MVN


def c_mesh(c_, d_):

    epsX = 0.15 * c_
    epsY = 0.15 * c_
    dX = 0.3 * c_
    dY = 0.3 * c_
    maxX = 1.0 * d_
    maxY = 1.0 * d_

    # define the renge in the quadrant
    rX = np.arange(epsX, maxX, dX)
    rY = np.arange(epsY, maxY, dY)

    # Total range
    Xrange = [-np.flip(rX), rX]
    Yrange = [-np.flip(rY), rY]

    # Mesh points
    xi, eta = np.meshgrid(Xrange, Yrange)
    ZETA = xi + 1j * eta
    ZETA /= d_

    return ZETA


def camber_mesh(c_, d_, camber):

    dX = 0.2 * c_
    dY = 0.2 * c_
    maxX = 1.0 * d_
    maxY = 1.0 * d_

    x1 = np.linspace(-0.5, 0.5, dX)
    x2 = np.linspace(0.7, maxX, dX)
    x3 = -np.fliplr(x2)
    x = np.append(x3, [x1, x2])
    nx = x.shape[0]
    atmp_ = 0.5
    y1 = camber * (atmp_ ** 2 - x1 ** 2)
    y2 = 0.0 * x2
    y = np.append(y2, [y1, y2])
    nyh = np.floor(nx / 2)

    for i in range(nyh):
        xi[i+nyh, :] = x
        eta[i+nyh, :] = y + (i - 0.5) * dY
        xi[i, :] = x
        eta[i, :] = y - (nyh - i + 0.5) * dY

    ZETA = complex(xi, eta)
    return ZETA / d_

class mpath:
    def cos_tail_b2(t):
        return jax.lax.cond(t <= 2, lambda t: jnp.cos(jnp.pi * t), lambda t: 1.0, t)

    def cos_tail_g2(t, e):
        tB = t % 4
        return mpath.cos_tail_b2(tB) + e
    
    def d_cos_tail_b2(t):
        return jax.lax.cond(t <= 2.0, lambda t: -jnp.pi * jnp.sin(jnp.pi * t), lambda t: 0.0, t)

    def d_cos_tail_g2(t):
        tB = t % 4
        return mpath.d_cos_tail_b2(tB)
    
    def d_table_s_tail_b2(t, p, rtOff):
        e0 = jnp.exp(-2.0 * p * (t - (0.0 + rtOff)))
        e1 = jnp.exp(-2.0 * p * (t - (1.0 + rtOff)))
        e2 = jnp.exp(-2.0 * p * (t - (2.0 + rtOff)))
        e4 = jnp.exp(-2.0 * p * (t - (4.0 + rtOff)))
        f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
        f1 = 4.0 * p * e1 / (1.0 + e1) ** 2
        f2 = 2.0 * p * e2 / (1.0 + e2) ** 2
        f4 = 2.0 * p * e4 / (1.0 + e4) ** 2
        return -f0 + f1 - f2 - f4
    
    def d_table_s_tail_g2(t, p, rtOff):
        tB = t % 4
        return mpath.d_table_s_tail_b2(tB + g.tau)
    
    def table_s_tail_b2(t, p, rtOff):
        f0 = 1.0 / (1.0 + jnp.exp(t - (0.0 + rtOff)))
        f1 = 1.0 / (1.0 + jnp.exp(t - (1.0 + rtOff)))
        f2 = 1.0 / (1.0 + jnp.exp(t - (2.0 + rtOff)))
        f4 = 1.0 / (1.0 + jnp.exp(t - (4.0 + rtOff)))
        return -f0 + f1 - f2 - f4
    
    def table_s_tail_g2(t, p, rtOff):
        tB = t % 4
        return mpath.table_s_tail_b2(tB + g.tau, rtOff)
    
    def cos_up_tail_b2(t):
        return jax.lax.cond(t <= 2.0, lambda t: -jnp.cos(jnp.pi * t), lambda t: -1.0, t)
        
    def cos_up_tail_g2(t, e):
        tB = t % 4
        return mpath.cos_up_tail_b2(tB) + e
    
    def d_cos_up_tail_b2(t):
        return jax.lax.cond(t <= 2.0, lambda t: jnp.pi * jnp.sin(jnp.pi * t), lambda t: 0.0, t)
        
    def d_cos_up_tail_g2(t):
        tB = t % 4
        return mpath.d_cos_up_tail_b2(tB)
    
    def d_table_up_s_tail_b2(t, p, rtOff):
        e0 = jnp.exp(-2.0 * p * (t - (0.0 + rtOff)))
        e1 = jnp.exp(-2.0 * p * (t - (1.0 + rtOff)))
        e2 = jnp.exp(-2.0 * p * (t - (2.0 + rtOff)))
        e4 = jnp.exp(-2.0 * p * (t - (4.0 + rtOff)))
        f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
        f1 = 4.0 * p * e1 / (1.0 + e1) ** 2
        f2 = 2.0 * p * e2 / (1.0 + e2) ** 2
        f4 = 2.0 * p * e4 / (1.0 + e4) ** 2
        return -(-f0 + f1 - f2 - f4)
    
    def d_table_up_s_tail_g2(t, p, rtOff):
        tB = t % 4
        return mpath.d_table_up_s_tail_b2(tB + g.tau, p, rtOff)
    
    def table_up_s_tail_b2(t, p, rtOff):
        f0 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (t - (0.0 + rtOff))))
        f1 = 2.0 / (1.0 + jnp.exp(-2.0 * p * (t - (1.0 + rtOff))))
        f2 = 2.0 / (1.0 + jnp.exp(-2.0 * p * (t - (2.0 + rtOff))))
        f4 = 2.0 / (1.0 + jnp.exp(-2.0 * p * (t - (4.0 + rtOff))))
        return -(-f0 + f1 - f2 - f4)
    
    def table_up_s_tail_g2(t, p, rtOff):
        tB = t % 4
        return mpath.table_up_s_tail_b2(tB + g.tau, p, rtOff)
    
    def cos_up_tail_b(t):
        return jax.lax.cond(t <= 4.0, lambda t: -jnp.cos(jnp.pi * t), lambda t: -1.0, t)
        
    def cos_up_tail_g(t, e):
        tB = t % 8
        return mpath.cos_up_tail_b(tB) + e
    
    def d_cos_up_tail_b(t):
        return jax.lax.cond(t <= 4.0, lambda t: jnp.pi * jnp.sin(jnp.pi * t), lambda t: 0.0, t)
        
    def d_cos_up_tail_g(t):
        tB = t % 8
        return mpath.d_cos_up_tail_b(tB)
    
    def d_table_up_s_tail_b(t, p, rtOff):
        e0 = jnp.exp(-2.0 * p * (t - (0.0 + rtOff)))
        e1 = jnp.exp(-2.0 * p * (t - (1.0 + rtOff)))
        e2 = jnp.exp(-2.0 * p * (t - (2.0 + rtOff)))
        e3 = jnp.exp(-2.0 * p * (t - (3.0 + rtOff)))
        e4 = jnp.exp(-2.0 * p * (t - (4.0 + rtOff)))
        e8 = jnp.exp(-2.0 * p * (t - (8.0 + rtOff)))
        f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
        f1 = 2.0 * p * e1 / (1.0 + e0) ** 2
        f2 = 2.0 * p * e2 / (1.0 + e0) ** 2
        f3 = 2.0 * p * e3 / (1.0 + e0) ** 2
        f4 = 2.0 * p * e4 / (1.0 + e0) ** 2
        f8 = 2.0 * p * e8 / (1.0 + e0) ** 2
        return f0 - f1 + f2 - f3 + f4 + f8 # TODO
    
    def d_table_up_s_tail_g(t, p, rtOff):
        tB = t % 8
        return mpath.d_table_up_s_tail_b(tB + g.tau, p, rtOff)
    
    def table_up_s_tail_b(t, p, rtOff):
        f0 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (0.0 + rtOff)))
        f1 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (1.0 + rtOff)))
        f2 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (2.0 + rtOff)))
        f3 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (3.0 + rtOff)))
        f4 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (4.0 + rtOff)))
        f8 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (8.0 + rtOff)))
        return f0 - f1 + f2 - f3 + f4 + f8
    
    def table_up_s_tail_g(t, p, rtOff):
        tB = t % 8
        return mpath.table_up_s_tail_b(tB + g.tau, p, rtOff)
    
    def cos_tail_b(t):
        return jax.lax.cond(t <= 4.0, lambda t: jnp.cos(jnp.pi * t), lambda t: 1.0, t)
        
    def cos_tail_g(t):
        tB = t % 8
        return mpath.cos_tail_b(tB)
    
    def d_cos_tail_b(t):
        return jax.lax.cond(t <= 4.0, lambda t: jnp.pi * jnp.sin(jnp.pi * t), lambda t: 0.0, t)
        
    def d_cos_tail_g(t):
        tB = t % 8
        return mpath.d_cos_tail_b(tB)
    
    def d_table_s_tail_b(t, p, rtOff):
        e0 = jnp.exp(-2.0 * p * (t - (0.0 + rtOff)))
        e1 = jnp.exp(-2.0 * p * (t - (1.0 + rtOff)))
        e2 = jnp.exp(-2.0 * p * (t - (2.0 + rtOff)))
        e3 = jnp.exp(-2.0 * p * (t - (3.0 + rtOff)))
        e4 = jnp.exp(-2.0 * p * (t - (4.0 + rtOff)))
        e8 = jnp.exp(-2.0 * p * (t - (8.0 + rtOff)))
        f0 = 2.0 * p * e0 / (1.0 + e0) ** 2
        f1 = 2.0 * p * e0 / (1.0 + e0) ** 2
        f2 = 2.0 * p * e0 / (1.0 + e0) ** 2
        f3 = 2.0 * p * e0 / (1.0 + e0) ** 2
        f4 = 2.0 * p * e0 / (1.0 + e0) ** 2
        f8 = 2.0 * p * e0 / (1.0 + e0) ** 2
        return -f0 + f1 - f2 + f3 - f4 + f8
        
    def d_table_s_tail_g(t, p, rtOff):
        tB = t % 8
        return mpath.d_table_s_tail_b(tB + g.tau, p, rtOff)
    
    def table_s_tail_b(t, p, rtOff):
        f0 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (t - (0.0 + rtOff))))
        f1 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (t - (1.0 + rtOff))))
        f2 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (t - (2.0 + rtOff))))
        f3 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (t - (3.0 + rtOff))))
        f4 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (t - (4.0 + rtOff))))
        f8 = 1.0 / (1.0 + jnp.exp(-2.0 * p * (t - (8.0 + rtOff))))
        return -f0 + f1 - f2 + f3 - f4 - f8
    
    def table_s_tail_g(t, p, rtOff):
        tB = t % 8
        return mpath.table_s_tail_b(tB + g.tau, p, rtOff)
    
    def dtable_b(t, p, rtOff):
        e0 = jnp.exp(-2.0 * p * (t - (0.0 + rtOff)))
        e1 = jnp.exp(-2.0 * p * (t - (1.0 + rtOff)))
        e2 = jnp.exp(-2.0 * p * (t - (2.0 + rtOff)))
        e3 = jnp.exp(-2.0 * p * (t - (3.0 + rtOff)))
        e4 = jnp.exp(-2.0 * p * (t - (4.0 + rtOff)))
        f0 = 4.0 * p * e0 / (1.0 + e0) ** 2
        f1 = 4.0 * p * e1 / (1.0 + e1) ** 2
        f2 = 4.0 * p * e2 / (1.0 + e2) ** 2
        f3 = 4.0 * p * e3 / (1.0 + e3) ** 2
        f4 = 4.0 * p * e4 / (1.0 + e4) ** 2
        return -f0 + f1 - f2 + f3 - f4
    
    def dtable_g(t, p, rtOff):
        tB = t % 2
        return mpath.dtable_b(tB + g.tau, p, rtOff)
    
    def table_b(t, p, rtOff):
        f0 = 2.0 / (1.0 + jnp.exp(-2.0 * p * (t - (0.0 + rtOff))))
        f1 = 2.0 / (1.0 + jnp.exp(-2.0 * p * (t - (1.0 + rtOff))))
        f2 = 2.0 / (1.0 + jnp.exp(-2.0 * p * (t - (2.0 + rtOff))))
        f3 = 2.0 / (1.0 + jnp.exp(-2.0 * p * (t - (3.0 + rtOff))))
        f4 = 2.0 / (1.0 + jnp.exp(-2.0 * p * (t - (4.0 + rtOff))))
        return 1.0 - f0 + f1 - f2 + f3 - f4
    
    def table_g(t, p, rtOff):
        tB = t % 2
        y = mpath.table_b(tB + g.tau, p, rtOff)
        return y

def airfoil_m(t, e, beta, gMax, p, rtOff, U, V):
    if (g.mpath == 0):
        l = -U * t + 0.5 * (jnp.cos(jnp.pi * (t + g.tau)) + e) * jnp.cos(beta)
        h = -V * t + 0.5 * (jnp.cos(jnp.pi * (t + g.tau)) + e) * jnp.sin(beta)
        dl = -U - 0.5 * jnp.pi * jnp.sin(jnp.pi * (t + g.tau)) * jnp.cos(beta)
        dh = -V - 0.5 * jnp.pi * jnp.sin(jnp.pi * (t + g.tau)) * jnp.sin(beta)
        gam = mpath.table_g(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * jnp.pi - beta + gam
        dgam = mpath.dtable_g(t, p, rtOff)
        dalp = gMax * dgam
    elif (g.mpath == 1):
        dl = -U + 0.5 * mpath.d_cos_tail_g(t + g.tau) * jnp.cos(beta)
        dh = -V + 0.5 * mpath.d_cos_tail_g2(t + g.tau) * jnp.sin(beta)
        l = -U * t + 0.5 * mpath.cos_tail_g(t + g.tau, e) * jnp.cos(beta)
        h = -V * t + 0.5 * mpath.cos_tail_g2(t + g.tau, e) * jnp.sin(beta)
        gam = mpath.table_s_tail_g2(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * jnp.pi - beta + gam
        dgam = mpath.d_table_s_tail_g2(t, p, rtOff)
        dalp = gMax * dgam
    elif (g.mpath == 2):
        # Translational Motion
        dl = -U * 0.5 * mpath.d_cos_up_tail_g2(t + g.tau) * jnp.cos(beta)
        dh = -V + 0.5 * mpath.d_cos_up_tail_g2(t + g.tau) * jnp.sin(beta)
        l = -U * t + 0.5 * mpath.cos_up_tail_g2(t + g.tau, e) * jnp.cos(beta)
        h = -V * t + 0.5 * mpath.cos_up_tail_g2(t + g.tau, e) * jnp.sin(beta)
        # Rotational Motion
        gam = mpath.table_up_s_tail_g2(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * jnp.i - beta + gam
        dgam = mpath.d_table_up_s_tail_g2(t, p, rtOff)
        dalp = gMax * dgam
    elif (g.mpath == 3):
        # Translational Motion
        dl = -U * 0.5 * mpath.d_cos_tail_g(t + g.tau) * jnp.cos(beta)
        dh = -V + 0.5 * mpath.d_cos_tail_g(t + g.tau) * jnp.sin(beta)
        l = -U * t + 0.5 * mpath.cos_tail_g(t + g.tau, e) * jnp.cos(beta)
        h = -V * t + 0.5 * mpath.cos_tail_g(t + g.tau, e) * jnp.sin(beta)
        # Rotational Motion
        gam = mpath.table_s_tail_g(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * jnp.i - beta + gam
        dgam = mpath.d_table_s_tail_g(t, p, rtOff)
        dalp = gMax * dgam
    elif (g.mpath == 4):
        # Translational Motion
        dl = -U * 0.5 * mpath.d_cos_up_tail_g(t + g.tau) * jnp.cos(beta)
        dh = -V + 0.5 * mpath.d_cos_up_tail_g(t + g.tau) * jnp.sin(beta)
        l = -U * t + 0.5 * mpath.cos_up_tail_g(t + g.tau, e) * jnp.cos(beta)
        h = -V * t + 0.5 * mpath.cos_up_tail_g(t + g.tau, e) * jnp.sin(beta)
        # Rotational Motion
        gam = mpath.table_up_s_tail_g(t, p, rtOff)
        gam = gMax * gam
        alp = 0.5 * jnp.i - beta + gam
        dgam = mpath.d_table_up_s_tail_g(t, p, rtOff)
        dalp = gMax * dgam
    return alp, l, h, dalp, dl, dh


def wing_global(istep, t, a, alp, l, h, xv, yv, xc, yc, dfc, ZW, U, V):
    zt = l + 1j * h
    ZWt = ZW

#     if istep != 1:
#         ZWt = ZW - zt
        
    ZWt = jax.lax.cond(istep != 1, lambda istep: ZW - zt, lambda istep: ZW, istep)

    zv = xv + 1j * yv
    zc = xc + 1j * yc
    expmia = jnp.exp(-1j * alp)
    ZVt = (a + zv) * expmia
    ZCt = (a + zc) * expmia
    ZV = ZVt + zt
    ZC = ZCt + zt

    # Unit normal vector of the airfoil in the wing-fixed system
    denom = jnp.sqrt(1 + dfc ** 2)
    nx = -dfc / denom
    ny = 1.0 / denom
    nc = nx + 1j * ny
    # Unit normal vector of the airfoil in the global system
    NC = nc * expmia

    return NC, ZV, ZC, ZVt, ZCt, ZWt


def airfoil_v(ZC, ZCt, NC, t, dl, dh, dalp):
    V = (dl + 1j * dh) - 1j * dalp * ZCt
    VN = jnp.real(jnp.conj(V) * NC)
    return VN

def velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw):
    eps = g.eps
    ZF_c = ZF[0:iGAMAw]
#     ZF_c = jax.lax.dynamic_slice_in_dim(ZF, 0, iGAMAw.astype(jnp.int32))
    GAMAw_c = GAMAw[0:iGAMAw]
#     GAMAw_c = jax.lax.dynamic_slice_in_dim(GAMAw, 0, iGAMAw.astype(jnp.int32))

    r_ = jnp.subtract(jnp.expand_dims(ZC, 0).transpose(), ZF_c)
    r = jnp.abs(r_)
    GF = jnp.where(r < eps, 0.+0.j, (1.0 / r_))
    GF = GF * jnp.where(r < g.delta, (r / g.delta) ** 2, 1.)

    VNW = jnp.sum(GAMAw_c * jnp.imag(jnp.expand_dims(NC, 0).transpose()
                                   * GF) / (2.0 * jnp.pi), 1)

    return VNW

def vel_vortex_improved(GAM, z, z0):
    r = jnp.abs(jnp.subtract(jnp.reshape(z, (z.shape[0], 1)), z0))
    c = jnp.subtract(jnp.reshape(z, (z.shape[0], 1)), z0)
#     v = 1j * jnp.divide(GAM, c, out=jnp.zeros_like(c),
#                        where=c != 0) / (2.0 * jnp.pi)
    
    v_ = jnp.where(c != 0, jnp.divide(GAM, c).reshape(c.shape), 0.0)
    v = 1j * v_ / (2.0 * jnp.pi)
    
    v = v * jnp.where(r < g.delta, (r / g.delta) ** 2, 1.0)
    v = jnp.conjugate(v)
    return v


def velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw):
    v1 = jnp.sum(vel_vortex_improved(GAMA[0:m], ZF[0:iGAMAf], ZV[0:m]), axis=1)
    v2 = jnp.sum(vel_vortex_improved(
        GAMAw[0:iGAMAw], ZF[0:iGAMAf], ZF[0:iGAMAw]), axis=1)
    vs1 = v1.shape[0]
    vs2 = v2.shape[0]

    v1_final = jnp.pad(v1, (0, max(vs2 - vs1, 0)), mode="constant")
    v2_final = jnp.pad(v2, (0, max(vs1 - vs2, 0)), mode="constant")

    return ((v1_final + v2_final) * -1)[0:iGAMAf]

l_ = 0.5 * 5.0 # Change this number.
n = 250
atmp_ = 0.8
x_ = jnp.linspace(-atmp_, atmp_, n, endpoint=True)
camber = 0.0
y_ = camber * (atmp_ ** 2 - x_ ** 2)
c_ = x_[n - 1] - x_[0]
m = 50
phiT_ = 45
phiB_ = -45
a_ = 0
beta_ = -30
f_ = 30
gMax_ = 30
p = 5
rtOff = 0.0
rho_ = 0.001225
U_ = 100.0
V_ = 0.0
itinc = 1
svInc = 0.025
svMax = 2.5
g.svCont = jnp.arange(0.0, svMax + 1e-10, svInc)
wvInc = 0.1
wvMax = 7.0
g.wvCont = jnp.arange(0.0, wvMax + 1e-10, wvInc)
q = 1.0
dt = 0.025
nstep = 81

v_, t_, d_, e, c, x, y, a, beta, gMax, U, V = in_data(l_, phiT_, phiB_, c_, x_, y_, a_, beta_, f_, gMax_, U_, V_)

g.delta = 0.5 * c / (m - 1) * q

if itinc == 1:
    nperiod = 1
    dt = min(c / (m - 1), 0.1 * (4 / p))
    nstep = int(nperiod * np.ceil(2/dt))

air = jnp.sqrt(U_ ** 2 + V_ ** 2)
fk = 2 * f_ * d_ / air
r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)
k = fk * r

if air <= 1e-03:
    r = 0.25 * (c_ / d_) * (p / t_) * (gMax / f_)

xv, yv, xc, yc, dfc, m  = mesh_r(c, x, y, n, m)

def wing_global_plot(ZC, NC, t, ):
    plt.plot(np.real(ZC), np.imag(ZC), 'o')
    sf = 0.025
    xaif = np.real(ZC)
    yaif = np.imag(ZC)
    xtip = xaif + sf * np.real(NC)
    ytip = yaif + sf * np.imag(NC)
    plt.plot([xaif, xtip], [yaif, ytip])
    plt.savefig(f"{fig}w2g_{np.round(t, 4)}.tif")
    plt.clf()
    
    
def air_foil_v_plot(ZC, NC, VN, t):
    sf = 0.025
    xc = np.real(ZC)
    yc = np.imag(ZC)
    nx = np.real(NC)
    ny = np.imag(NC)
    xaif = xc
    yaif = yc
    xtip = xc + sf * VN * nx
    ytip = yc + sf * VN * ny
    plt.plot([xaif, xtip], [yaif, ytip])
    plt.axis('equal')
    plt.plot(xc, yc, 'o')
    plt.savefig(f"{fig}AirfoilVg_{np.round(t, 4)}.tif")
    plt.clf()
    
    
def plot_wake_vortex(iGAMAw, ZV, ZW, istep):
    xpltf = np.real(ZV)
    ypltf = np.imag(ZV)

    if istep == 0:
        plt.plot(xpltf, ypltf, '-k')
        plt.savefig(f"{fig}wake/wake_{istep}.tif")
    else:
        xpltw = np.real(ZW)
        ypltw = np.imag(ZW)

        xpltwo = xpltw[1::2]
        ypltwo = ypltw[1::2]
        xpltwe = xpltw[::2]
        ypltwe = ypltw[::2]

        plt.plot(xpltf, ypltf, '-k',
                 xpltwo, ypltwo, 'ok',
                 xpltwe, ypltwe, 'or')
        plt.savefig(f"{fig}wake/wake_{istep}.tif")
    plt.clf()
    
    
def igVELF(Z, ZV, ZW, GAMA, m, GAMAw, iGAMAw, U, V, alp, dalp, dl, dh):
    sz = np.size(Z)
    VV = complex(0, 0) * np.ones(sz)
    for J in range(1, m + 1):
        for i in range(1, sz[0] + 1):
            for j in range(1, sz[1] + 1):
                VV[i - 1, j - 1] = VV[i - 1, j - 1] + \
                    vel_vortex(GAMA[J - 1], Z[i - 1, j - 1], ZV[J - 1])
    for J in range(1, iGAMAw + 1):
        for i in range(1, sz[0] + 1):
            for j in range(1, sz[1] + 1):
                VV[i - 1, j - 1] = VV[i - 1, j - 1] + \
                    vel_vortex(GAMA[J - 1], Z[i - 1, j - 1], ZV[j - 1])
    return VV

def igVELOCITYF(Z, ZV, ZW, a, GAMA, m, GAMAw, iGAMAw, U, V, alp, dalp, dl, dh):
    sz = np.size(Z)
    VV = np.zeros(sz) + 1j * np.zeros(sz)
    Z_ = np.reshape(Z, (196, 1))
    VV = VV - np.sum((0.5 * 1j / np.pi) *
                     np.divide(GAMA, (np.subtract(Z_, ZV))), 1)
    VV = VV - np.sum((0.5 * 1j / np.pi) *
                     np.divide(GAMAw[0:iGAMAw], (np.subtract(Z_, ZW[0:iGAMAw]))), 1)
    VV = np.conj(VV)
    VVspace = VV
    VVspace = VV + np.exp(1j * alp) * (U + 1j * V) * np.ones(sz)
    return VVspace

def plot_velocity(istep, ZV, ZW, a, GAMA, m,
                  GAMAw, iGAMAw, U, V, alp, l, h, dalp,
                  dl, dh, ZETA, vpFreq, zavoid, ivCont):
    # Airfoil 9/10/2018
    XPLTF = np.real(ZV)
    YPLTF = np.imag(ZV)

    # Plot the velocity field, every vpFreq seps.
    if istep % vpFreq == 0:
        # Calculate the velocity field.
        ROT = np.exp(-1j * alp)
        RZETA = (ZETA + a) * ROT

        X = np.real(RZETA) + l
        Y = np.imag(RZETA) + h
        Z = X + 1j * Y

        if zavoid == 1:
            VVspace = igVELF(Z, ZV, ZW, GAMA, m, GAMAw,
                             iGAMAw, U, V, alp, dalp, dl, dh)
        else:
            VVspace = igVELOCITYF(Z, ZV, ZW, a, GAMA, m, GAMAw,
                                  iGAMAw, U, V, alp, dalp, dl, dh)

        # Plot the velocity field in the space-fixed system.

        U = np.real(VVspace)
        V = np.imag(VVspace)
        S = np.sqrt(U * U + V * V)
        S = np.reshape(
            S, (int(math.sqrt(S.shape[0])), int(math.sqrt(S.shape[0]))))

        plt.quiver(X, Y, U, V)
        # plt.plot(XPLTF, YPLTF, '-b')
        plt.savefig(f"{fig}velocity/spaceVelocity_{istep}.png")
        plt.clf()

        if ivCont == 1:
            plt.contour(X, Y, S, g.svCont)
            plt.contourf(X, Y, S, g.svCont)
        else:
            plt.contour(X, Y, S)
            plt.contourf(X, Y, S)

        plt.colorbar()

        plt.plot(XPLTF, YPLTF, '-b', linewidth='4')
        plt.savefig(f"{fig}velocity/spaceSpeed_{istep}.png")
        plt.clf()
        
        
def force_moment(rho_, v_, d_, nstep, dt, U, V, impulseAb, impulseAw, impulseLb, impulseLw, LDOT, HODT):
    forceb = np.zeros((nstep)) + 1j * np.zeros((nstep))
    forcew = np.zeros((nstep)) + 1j * np.zeros((nstep))
    force = np.zeros((nstep)) + 1j * np.zeros((nstep))
    momentb = np.zeros((nstep))
    momentw = np.zeros((nstep))
    moment = np.zeros((nstep))

    impulseAb = np.real(impulseAb)
    impulseAw = np.real(impulseAw)

    # Reference values of force and moment
    f_ = rho_ * (v_ ** 2) * d_
    m_ = f_ * d_

    for IT in range(nstep):

        U0 = (LDOT[IT] - U) + 1j * (HDOT[IT] - V)
        U0_conj = np.conj(U0)

        if IT == 0:
            forceb[0] = (impulseLb[1] - impulseLb[0]) / dt
            forcew[0] = (impulseLw[1] - impulseLw[0]) / dt
            momentb[0] = (impulseAb[1] - impulseAb[0]) / dt
            momentw[0] = (impulseAw[1] - impulseAw[0]) / dt

        elif IT == (nstep - 1):
            forceb[IT] = 0.5 * (3.0 * impulseLb[IT] - 4.0 * impulseLb[IT - 1] + impulseLb[IT - 2]) / dt
            forcew[IT] = 0.5 * (3.0 * impulseLw[IT] - 4.0 * impulseLw[IT - 1] + impulseLw[IT - 2]) / dt
            momentb[IT] = 0.5 * (3.0 * impulseAb[IT] - 4.0 * impulseAb[IT - 1] + impulseAb[IT - 2]) / dt
            momentw[IT] = 0.5 * (3.0 * impulseAw[IT] - 4.0 * impulseAw[IT - 1] + impulseAw[IT - 2]) / dt

        else:
            forceb[IT] = 0.5 * (impulseLb[IT + 1] - impulseLb[IT - 1]) / dt
            forcew[IT] = 0.5 * (impulseLw[IT + 1] - impulseLw[IT - 1]) / dt
            momentb[IT] = 0.5 * (impulseAb[IT + 1] - impulseAb[IT - 1]) / dt
            momentw[IT] = 0.5 * (impulseAw[IT + 1] - impulseAw[IT - 1]) / dt

        momentb[IT] = momentb[IT] + np.imag(U0_conj * impulseLb[IT])
        momentw[IT] = momentw[IT] + np.imag(U0_conj * impulseLw[IT])

        # Total force and moment ( these are on the fluid )
        # The dimensional force & moment on the wing are obtained by reversing the sign.
        # and multiplying the referse quantities
        force[IT] = -f_ * (forceb[IT] + forcew[IT])
        moment[IT] = -m_ * (momentb[IT] + momentw[IT])

    # print(moment)

    ITa = np.linspace(1, nstep, nstep, endpoint=True)

    plt.plot(ITa, np.real(force), 'x-k')
    plt.savefig(f"{fig}fx.tif")
    plt.clf()
    plt.plot(ITa, np.imag(force), '+-k')
    plt.savefig(f"{fig}fy.tif")
    plt.clf()
    plt.plot(ITa, moment, 'o-r')
    plt.savefig(f"{fig}m.tif")
    plt.clf()

    # Calculate the average forces and moment
    Fx = np.real(force)
    Fy = np.imag(force)
    Mz = moment
    Fx_avr = np.average(Fx)
    Fy_avr = np.average(Fy)
    Mz_avr = np.average(Mz)
    
    
def plot_m_vortex(v_, d_, GAMAw, nstep):

    gama_ = v_ * d_

    # Dimensional alues of the circulation
    GAMAwo = gama_ * GAMAw[0::2]
    GAMAwe = gama_ * GAMAw[1::2]
    it = list(range(1, nstep + 1))

    plt.plot(it, GAMAwo, 'o-k', it, GAMAwe, 'o-r')
    plt.savefig(f"{fig}GAMAw.tif")
    plt.clf()

GAMAw = jnp.zeros(2 * nstep)
GAMAf = jnp.zeros(2 * nstep)
sGAMAw = 0.0
iGAMAw = 0
iGAMAf = 0
ZF = jnp.zeros(2 * nstep, dtype=complex)
ZW = jnp.zeros(2 * nstep, dtype=complex)
impulseLb = jnp.zeros(nstep, dtype=complex)
impulseLw = jnp.zeros(nstep, dtype=complex)
impulseAb = jnp.zeros(nstep)
impulseAw = jnp.zeros(nstep)
LDOT = jnp.zeros(nstep)
HDOT = jnp.zeros(nstep)

MVN = matrix_coef(xv, yv, xc, yc, dfc, m)
MVN_lu = lu_factor(MVN)

ZETA = 0
if g.vfplot == 1:
    if camber == 0.0:
        ZETA = c_mesh(c_, d_)
    else:
        ZETA = camber_mesh(c_, d_, camber)

iterations = {
    'ZC': [],
    'NC': [],
    't': [],
    'VN': [],
    'iGAMAw': [],
    'ZV': [],
    'ZW': [],
    'GAMA': [],
    'GAMAw': [],
    'U': [],
    'V': [],
    'alp': [],
    'l': [],
    'h': [],
    'dalp': [],
    'dl': [],
    'dh': []
}

def timemarch_step(istep, GAMAw, GAMAf, sGAMAw, iGAMAw, iGAMAf, ZF, ZW, impulseLb, impulseLw,
                  impulseAb, impulseAw, LDOT, HDOT):
    t = istep * dt
    alp, l, h, dalp, dl, dh = airfoil_m(t, e, beta, gMax, p, rtOff, U, V)
    
    LDOT = LDOT.at[istep].set(dl)
    HDOT = HDOT.at[istep].set(dh)
    
    NC, ZV, ZC, ZVt, ZCt, ZWt = wing_global(istep, t, a, alp, l, h, xv, yv, xc, yc, dfc, ZW, U, V)
    
    VN = airfoil_v(ZC, ZCt, NC, t, dl, dh, dalp)
    VNW = velocity_w2(m, ZC, NC, ZF, GAMAw, iGAMAw)
    
    GAMA = VN - VNW
    GAMA = jnp.append(GAMA, -sGAMAw)
    GAMA = jax.scipy.linalg.lu_solve(MVN_lu, GAMA)
    
    iGAMAw_ = iGAMAw
    ZW_ = ZW
    
    impulseLb = impulseLb.at[istep].set(-1j * jnp.sum(GAMA * ZVt))
    impulseAb = impulseAb.at[istep].set(0.5 * jnp.sum(GAMA * jnp.abs(ZVt) ** 2))
    impulseLw = impulseLw.at[istep].set(-1j * jnp.sum(GAMAw[0:iGAMAw] * ZWt[0:iGAMAw]))
    impulseAw = impulseAw.at[istep].set(0.5 * jnp.sum(GAMAw[0:iGAMAw] * jnp.abs(ZWt[0:iGAMAw]) ** 2))
    
    GAMAw_ = GAMAw
    
    iGAMAf = 2 * (istep + 1)

    ZF = ZF.at[iGAMAf - 2].set(ZV[0])
    ZF = ZF.at[iGAMAf - 1].set(ZV[m - 1])

    VELF = velocity(ZF, iGAMAf, GAMA, m, ZV, GAMAw, iGAMAw)

    ZW = ZW.at[0:iGAMAf].set(ZF[0:iGAMAf] + VELF * dt)

    iGAMAw = iGAMAw + 2
    GAMAw = GAMAw.at[iGAMAf - 2].set(GAMA[0])
    GAMAw = GAMAw.at[iGAMAf - 1].set(GAMA[m - 1])
    sGAMAw = sGAMAw + GAMA[0] + GAMA[m - 1]

    ZF = ZW
    
    return ZC, NC, t, VN, iGAMAw, iGAMAw_, ZV, ZW, ZW_, GAMA, GAMAw, GAMAw_, U, V, alp, l, h, dalp, dl, dh


timemarch_step_jit = jax.jit(timemarch_step, static_argnums=(0, 4, 5))

#time


for istep in range(nstep):
    
    iGAMAw = int(iGAMAw)
    
    ZC, NC, t, VN, iGAMAw, iGAMAw_, ZV, ZW, ZW_, GAMA, GAMAw, GAMAw_, U, V, alp, l, h, dalp, dl, dh = timemarch_step_jit(istep, 
        GAMAw, GAMAf, sGAMAw, iGAMAw, iGAMAf, ZF, ZW, impulseLb, impulseLw, impulseAb, impulseAw, LDOT, HDOT)
    
    iterations['ZC'].append(np.copy(ZC))  # Ok
    iterations['NC'].append(np.copy(NC))  # Ok
    iterations['t'].append(np.copy(t))  # Ok
    iterations['VN'].append(np.copy(VN))
    iterations['iGAMAw'].append(np.copy(iGAMAw_))  # Ok (modified)
    iterations['ZV'].append(np.copy(ZV))  # Ok
    iterations['ZW'].append(np.copy(ZW_))  # 
    iterations['GAMA'].append(np.copy(GAMA))
    iterations['GAMAw'].append(np.copy(GAMAw_))
    iterations['U'].append(np.copy(U))
    iterations['V'].append(np.copy(V))
    iterations['alp'].append(np.copy(alp))
    iterations['l'].append(np.copy(l))
    iterations['h'].append(np.copy(h))
    iterations['dalp'].append(np.copy(dalp))
    iterations['dl'].append(np.copy(dl))
    iterations['dh'].append(np.copy(dh))

import pickle

pickle.dump(iterations, file=open("output.pickle", "wb"))

import pickle

iterations = pickle.load(open("output.pickle", "rb"))

force_moment(rho_, v_, d_, nstep, dt, U, V, impulseAb, impulseAw, impulseLb, impulseLw, LDOT, HDOT)
plot_m_vortex(v_, d_, GAMAw, nstep)

#time

print(len(iterations['VN']) == nstep)

for istep in range(nstep):
    ZC = iterations['ZC'][istep]
    NC = iterations['NC'][istep]
    t = iterations['t'][istep]
    VN = iterations['VN'][istep]
    iGAMAw = iterations['iGAMAw'][istep]
    ZV = iterations['ZV'][istep]
    ZW = iterations['ZW'][istep]
    GAMA = iterations['GAMA'][istep]
    GAMAw = iterations['GAMAw'][istep]
    U = iterations['U'][istep]
    V = iterations['V'][istep]
    alp = iterations['alp'][istep]
    l = iterations['l'][istep]
    h = iterations['h'][istep]
    dalp = iterations['dalp'][istep]
    dl = iterations['dl'][istep]
    dh = iterations['dh'][istep]
    
    wing_global_plot(ZC, NC, t)
    air_foil_v_plot(ZC, NC, VN, t)
    plot_wake_vortex(iGAMAw, ZV, ZW, istep)
    plot_velocity(istep, ZV, ZW, a, GAMA, m, GAMAw, iGAMAw, U, V, alp, l, h, dalp, dl, dh, ZETA, g.vpFreq, 
                  g.zavoid, g.ivCont)