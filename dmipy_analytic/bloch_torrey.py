"""The one-dimensional Bloch-Torrey equation on an interval of layers, solved to a stated accuracy: the semi-closed
reference for the situations in which restriction, exchange through membranes, relaxation (in the bulk and at the
walls), off-resonance and a background field gradient act together.

**The equation.** The transverse magnetisation density ``m(x, t)`` on ``[x_0, x_N]`` obeys

    dm/dt = D m'' - [R2(x) + i omega(x) + i gamma (G(t) + g0) x] m,

with one diffusivity ``D``, the bulk relaxation rate ``R2(x)`` and the off-resonance ``omega(x)`` constant on each layer,
the applied gradient ``G(t)`` along ``x`` and a background (static) field gradient ``g0`` along ``x``. The interval is
divided into layers by interior edges; at each interior edge the flux ``J = -D m'`` obeys the membrane conditions

    J(x_e-) = kappa (m_- - m_+) + rho_e m_-,      J(x_e+) = kappa (m_- - m_+) - rho_e m_+,

the transmission condition ``D m' = kappa [m]`` with a permeability ``kappa`` (Tanner JE (1978), J. Chem. Phys. 69:1748) and a
surface relaxivity ``rho_e`` on each face of the membrane. An edge of infinite permeability and no relaxivity is no
membrane at all: the layers on either side differ in their ``R2`` and ``omega`` only. The two ends are walls with
the Robin condition ``D m' = +rho m`` at the left wall and ``D m' = -rho m`` at the right one (Brownstein KS, Tarr CE (1979), Phys. Rev. A 19:2446):
``rho = 0`` is a reflecting wall. An ideal 180-degree pulse conjugates the magnetisation, ``m -> conj(m)``; the
gradient is the one played (its sign is not folded in), and so is the background, which does not switch off.

**The method.** Cell-centred finite volumes: each layer is divided into cells of (nearly) equal width, the flux across
a face between two cells is the conductance of the two half-cells in series, and across a membrane or at a wall the
face's nodes are eliminated from the resistor network of the half-cells, the permeability and the relaxivities (a
symmetric Schur complement). The diffusion operator ``-H^-1 C`` (``H`` the cell widths, ``C`` the symmetric
conductance matrix) is symmetrised, ``-H^-1/2 C H^-1/2 = U diag(lambda) U^T`` with ``lambda <= 0``, and each step of
``dt`` is the Strang split: half a step of diffusion in that eigenbasis, the potential ``R2 + i omega + i gamma G x``
applied exactly (it is diagonal), half a step of diffusion.

**Its accuracy.** The scheme is second order in the cell width and in the step: refining either by two leaves a
quarter of the error, so the change is three quarters of it. The cell count and the sub-steps per waveform sample are arguments, and the
tests state the error of the defaults by refining both (Richardson), on every configuration they compare.

The diffusion step needs the symmetric eigendecomposition of ``numpy.linalg.eigh``; nothing else.
"""
import numpy as np

from .constants import GAMMA

__all__ = ["BlochTorrey1D"]


def _per_layer(value, n_layers, name):
    """``value`` as a float array of one entry per layer: a scalar repeated, or a sequence of ``n_layers``."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_layers, float(arr))
    if arr.shape != (n_layers,):
        raise ValueError(f"{name} is a scalar or one value per layer ({n_layers}), not an array of shape {arr.shape}")
    return arr.copy()


def _per_edge(value, n_edges, name):
    """``value`` as a float array of one entry per interior edge (scalar repeated, or ``n_edges`` values)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_edges, float(arr))
    if arr.shape != (n_edges,):
        raise ValueError(f"{name} is a scalar or one value per interior edge ({n_edges}), not an array of shape "
                         f"{arr.shape}")
    return arr.copy()


def _membrane_block(g_left, g_right, kappa, rho):
    """The 2x2 symmetric conductance block of a membrane between two half-cells: the face nodes ``m_-`` and ``m_+``
    eliminated from the network ``node_L -g_left- m_- -kappa- m_+ -g_right- node_R`` with ``rho`` from each face node to
    ground. ``Y = diag(g) - diag(g) M^-1 diag(g)``, ``M = [[g_left + kappa + rho, -kappa], [-kappa, g_right + kappa +
    rho]]``; ``h dm/dt = -Y m`` on the two cells."""
    g = np.array([g_left, g_right])
    M = np.array([[g_left + kappa + rho, -kappa], [-kappa, g_right + kappa + rho]])
    Y = np.diag(g) - (g[:, None] * np.linalg.solve(M, np.diag(g)))
    return 0.5 * (Y + Y.T)


class BlochTorrey1D:
    """The 1-D Bloch-Torrey equation on layers between two walls:
    ``BlochTorrey1D(edges_m, D, permeability=..., R2=..., omega=...).signal(G, dt, refocus=...)``.

    Parameters
    ----------
    edges_m : sequence of float
        The layers' edges in increasing order, m: the first and the last are the walls, every other one an interior
        edge (a membrane, or a change of ``R2`` / ``omega`` alone). The coordinate ``x`` of the gradient terms is this
        one, so the origin is where the caller puts it (it sets the phase of an unbalanced waveform, not its magnitude).
    diffusivity : float
        The diffusivity ``D``, m^2/s, the same in every layer.
    permeability : float or sequence of float
        The permeability ``kappa`` of each interior edge, m/s: ``0`` seals it, ``numpy.inf`` makes it no membrane.
    membrane_relaxivity : float or sequence of float
        The surface relaxivity ``rho_e`` on each face of each interior edge, m/s (an edge with relaxivity needs a finite
        ``kappa``).
    wall_relaxivity : float or pair of float
        The surface relaxivity ``rho`` of the left and the right wall, m/s (``0``: reflecting).
    R2 : float or sequence of float
        The bulk relaxation rate of each layer, 1/s.
    omega : float or sequence of float
        The off-resonance angular frequency of each layer, rad/s.
    background_gradient : float
        A static field gradient ``g0`` along ``x``, T/m: the off-resonance ``gamma g0 x`` added to ``omega``.
    density : float or sequence of float
        The magnetisation density of each layer at the excitation (the spins' density): ``(1, 0)`` starts every
        spin in the first of two layers. The signal is normalised to the total at the excitation.
    n_cells : int
        The number of cells across the whole interval, shared among the layers in proportion to their widths (at
        least two per layer).
    gamma : float
        The gyromagnetic ratio, rad/(s T).
    """

    def __init__(self, edges_m, diffusivity, *, permeability=np.inf, membrane_relaxivity=0.0, wall_relaxivity=0.0,
                 R2=0.0, omega=0.0, background_gradient=0.0, density=1.0, n_cells=800, gamma=GAMMA):
        edges = np.asarray(edges_m, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError(f"edges_m is two or more finite, strictly increasing positions in metres, not {edges_m!r}")
        D = float(diffusivity)
        if not np.isfinite(D) or D <= 0:
            raise ValueError(f"the diffusivity is a positive number in m^2/s, not {diffusivity!r}")
        n_layers = len(edges) - 1
        n_edges = n_layers - 1
        kappa = _per_edge(permeability, n_edges, "permeability")
        rho_e = _per_edge(membrane_relaxivity, n_edges, "membrane_relaxivity")
        rho_w = np.asarray(wall_relaxivity, dtype=np.float64)
        rho_w = np.full(2, float(rho_w)) if rho_w.ndim == 0 else rho_w
        if rho_w.shape != (2,):
            raise ValueError(f"wall_relaxivity is one value or the pair (left, right), not {wall_relaxivity!r}")
        if np.any(kappa < 0) or np.any(np.isnan(kappa)):
            raise ValueError(f"a permeability is non-negative (numpy.inf: no membrane), not {permeability!r}")
        if np.any(rho_e < 0) or not np.all(np.isfinite(rho_e)) or np.any(rho_w < 0) or not np.all(np.isfinite(rho_w)):
            raise ValueError("a surface relaxivity is a finite non-negative number in m/s")
        if np.any(np.isinf(kappa) & (rho_e > 0)):
            raise ValueError("a membrane with a surface relaxivity has a finite permeability; an edge of infinite "
                             "permeability is no membrane and has no faces to relax on")
        n_cells = int(n_cells)
        width = np.diff(edges)
        per_layer = np.maximum(2, np.round(n_cells * width / width.sum()).astype(int))

        self.edges, self.D, self.gamma = edges, D, float(gamma)
        self.permeability, self.membrane_relaxivity, self.wall_relaxivity = kappa, rho_e, rho_w
        self.background_gradient = float(background_gradient)
        self.n_layers = n_layers
        self.cells_per_layer = per_layer
        self.R2_layers = _per_layer(R2, n_layers, "R2")
        self.omega_layers = _per_layer(omega, n_layers, "omega")
        density = _per_layer(density, n_layers, "density")
        if np.any(density < 0) or not np.any(density > 0):
            raise ValueError("the density is non-negative in every layer and positive in one")

        layer = np.repeat(np.arange(n_layers), per_layer)
        h = np.repeat(width / per_layer, per_layer)
        x = np.concatenate([edges[i] + (np.arange(n) + 0.5) * width[i] / n for i, n in enumerate(per_layer)])
        n = len(h)
        self.layer, self.h, self.x = layer, h, x

        C = np.zeros((n, n))
        g_half = 2.0 * D / h                                           # the conductance of a half-cell
        for i in range(n - 1):
            e = layer[i]                                               # across a layer's edge: the edge e
            if layer[i + 1] != e and np.isfinite(kappa[e]):
                Y = _membrane_block(g_half[i], g_half[i + 1], kappa[e], rho_e[e])
            else:                                                      # two half-cells in series
                Y = np.array([[1.0, -1.0], [-1.0, 1.0]]) / (1.0 / g_half[i] + 1.0 / g_half[i + 1])
            C[i:i + 2, i:i + 2] += Y
        for cell, rho in ((0, rho_w[0]), (n - 1, rho_w[1])):          # the Robin walls: half-cell and rho in series
            if rho > 0:
                C[cell, cell] += g_half[cell] * rho / (g_half[cell] + rho)
        s = 1.0 / np.sqrt(h)
        A = -(s[:, None] * C * s[None, :])
        self.lam, self.U = np.linalg.eigh(0.5 * (A + A.T))            # A = U diag(lam) U^T, lam <= 0
        self.m0 = density[layer]
        self.R2_cells = self.R2_layers[layer]
        self.omega_cells = self.omega_layers[layer] + self.gamma * self.background_gradient * x

    def signal(self, G, dt, *, refocus=(), sub_steps=1, by_layer=False):
        """The complex signal after the gradient ``G`` played for ``len(G)`` steps of ``dt``.

        Parameters
        ----------
        G : array_like, shape (n_t,) or (n_meas, n_t)
            The physical gradient along ``x`` on each step (T/m), held constant over the step.
        dt : float
            The step, s.
        refocus : sequence of int
            The step indices before which an ideal 180 conjugates the magnetisation, strictly increasing in
            ``[0, n_t]`` (``n_t``: after the last step).
        sub_steps : int
            The Strang steps each step of ``G`` is divided into (refining the time step by this factor).
        by_layer : bool
            Return the signal of each layer (the magnetisation summed over its cells) on a last axis instead of the
            total.

        Returns
        -------
        complex or ndarray
            The magnetisation at the end (summed over the interval, or per layer), as a fraction of the total at the
            excitation; shape ``()`` or ``(n_meas,)`` (with ``(n_layers,)`` appended when ``by_layer``).
        """
        G = np.asarray(G, dtype=np.float64)
        single = G.ndim == 1
        G2 = G[None] if single else G
        if G2.ndim != 2:
            raise ValueError(f"G is (n_t,) or (n_meas, n_t), not an array of shape {G.shape}")
        n_t = G2.shape[1]
        flips = [int(k) for k in refocus]
        if any(k != r for k, r in zip(flips, refocus)) or any(k < 0 or k > n_t for k in flips) \
                or any(b <= a for a, b in zip(flips, flips[1:])):
            raise ValueError(f"refocus holds strictly increasing step indices in [0, {n_t}], not {tuple(refocus)!r}")
        flips = set(flips)
        sub = int(sub_steps)
        if sub < 1:
            raise ValueError(f"sub_steps is a positive integer, not {sub_steps!r}")
        tau = float(dt) / sub
        half = np.exp(0.5 * tau * self.lam)[:, None]
        U, UT = self.U, self.U.T
        y = (np.sqrt(self.h) * self.m0).astype(complex)[:, None] * np.ones((1, G2.shape[0]))
        c = UT @ y                                                      # the symmetrised state's eigenbasis coefficients
        for k in range(n_t):
            if k in flips:                                              # the 180 conjugates the magnetisation
                c = UT @ np.conj(U @ c)
            V = np.exp(-tau * (self.R2_cells[:, None] + 1j * (self.omega_cells[:, None] + self.gamma * self.x[:, None] * G2[:, k])))
            for _ in range(sub):                                        # Strang: half diffusion, potential, half diffusion
                c = half * (UT @ (V * (U @ (half * c))))
        y = U @ c
        if n_t in flips:
            y = np.conj(y)
        m = y * np.sqrt(self.h)[:, None]                                   # the magnetisation times the cell width
        norm = float(np.sum(self.h * self.m0))
        if by_layer:
            out = np.stack([m[self.layer == i].sum(0) for i in range(self.n_layers)], axis=-1) / norm
        else:
            out = m.sum(0) / norm
        return out[0] if single else out
