r"""Independent narrow-pulse spectral oracle (no package imports).

Derivation, including the sign convention
-----------------------------------------
1. Write s=r/b, a=R/b, b=R_out, d=D_in/D_out, h=kappa*b/D_out,
   x^2=lambda*b^2/D_out. Both radial derivatives below point to INCREASING
   r. Conservation gives D_in f'_-=D_out f'_+=J_derivative. A passive
   membrane has J_derivative=kappa*(f_+-f_-), since the outward particle
   flux is -D f'. The brief and permeable_shell's docstring instead print
   kappa*(f_--f_+). That sign is anti-diffusive; the package's positive
   series conductance implements the passive sign used here. This is an
   explicit correction of the WRITTEN boundary condition, not a fitted
   parameter. With the literal sign replace h by -h in the determinant.

2. Separate f(r)P_l(cos theta) in 3D, f(r)cos(n phi) in 2D (the cylinder
   is per unit axial length, gradient perpendicular to its axis). The
   radial equation is -(s^(p-1) d_i f')' + c*d_i*s^(p-3)*f
   = x^2*s^(p-1)*f, p=3 or 2, c=l(l+1) or n^2. In a constant-D region
   division by d_i*s^(p-1) gives f''+(p-1)*f'/s
   + (x^2/d_i-c/s^2)*f=0. In 2D this is Bessel's equation; in 3D
   f=s^(-1/2)*g reduces it to Bessel order l+1/2. Thus
   its solutions are j_l(x*s/sqrt(d_i)), y_l(...) in 3D, J_n(...),Y_n(...)
   in 2D. Regularity removes y/Y in the inner region. Let u be the inner
   regular solution, v(s)=B(x*s) C'(x)-C(x*s) B'(x), with (B,C)=(j,y)
   or (J,Y). Then v'(1)=0 exactly. The solution is A*u inside, Bcoef*v
   outside. The two remaining conditions are
       d*A*u'(a)-Bcoef*v'(a)=0,
       (d*u'(a)+h*u(a))*A-h*v(a)*Bcoef=0.
   Thus F(x)=d*u'*v'+h*(u*v'-d*u'*v)=0 is a 2x2 determinant (up to sign).
   At h=0 use independent Neumann pools u'=0 and v'=0. At h=infinity
   use F_inf=u*v'-d*u'*v=0, enforcing continuity and flux continuity.
   For order zero add the constant mode: one for h>0, two for h=0.
   Positive rescalings of u and v used below do not change any root.

3. Green's identity, adding the two regions, gives
       <f,A g>=sum_i integral D_i*(grad f)*.grad g dV
                  + kappa*integral_membrane [f]*[g] dS,
   where [f]=f_--f_+. It is symmetric and nonnegative. Distinct modes
   are orthogonal with PLAIN VOLUME weight r^(p-1)dr times the angular
   integral, NOT r^(p-1)/D. The equilibrium density is uniform. The
   literal printed sign changes the membrane term to minus; a piecewise
   constant trial function with a jump has negative energy, so there is
   a negative eigenvalue. A decaying-only expansion cannot solve that
   literal problem. More explicitly, take f_-=1 and
   f_+=-a^p/(1-a^p), orthogonal to the uniform mode. Its literal-sign
   Rayleigh quotient is -p*kappa/[R*(1-a^p)]: at the requested radii
   and kappa=1e-4 it is -79.38/s (sphere), -65.64/s (cylinder).
   The minimum eigenvalue is at most this negative number. This is an
   independent instability diagnosis of the printed sign, without
   relying on any finite-volume eigenvalues.
   Radial norms here are N=integral_0^1 f^2 s^(p-1)ds.

4. The heat kernel is sum phi_k(r) phi_k(r0)* exp(-lambda_k*Delta).
   Uniform initial density 1/V and instantaneous phases exp(+/-i q z)
   give E=V^-1 sum |integral exp(i q z) phi_k* dV|^2 exp(-lambda_k Delta).
   Here q=gamma*G*delta is ANGULAR wave number, rad/m (no 2*pi).
   Rayleigh: exp(i Q s cos theta)=sum_l (2l+1)i^l j_l(Qs)P_l;
   in 2D it is J_0(Qs)+2 sum_n>=1 i^n J_n(Qs)cos(n phi).
   Angular norms are 4*pi/(2l+1), or 2*pi (n=0) and pi (n>0).
   After division by V, each radial mode contributes
       p*w_order * I^2/N * exp(-x^2*D_out*Delta/b^2),
       I=integral f(s) B_order(Qs) s^(p-1) ds,
       w_order=2l+1 (3D), 1 or 2 (2D).
   For a sealed intra start only inner modes remain, and replace the
   volume 1/p by a^p/p. No finite-volume operator or package helper is used.

5. Root completeness is CHECKED, not inferred from an unchanged signal.
   Scan x with step 0.025 and 0.0125, bracket every sign change, brentq
   to 2e-12. Each limiting pool has a simple radial spectrum; merge its
   two Neumann spectra, including the two zeros for order zero. Adding
   membrane energy is a rank-one positive form perturbation for each
   angular order. The min-max principle gives (indices start at zero)
       x_j(0) <= x_j(h) <= x_j(infinity) <= x_(j+1)(0).
   Check every limiting root against this interlacing, choose a cutoff
   with identical counts at zero and infinite permeability, and REQUIRE
   precisely that count for finite h on BOTH grids, as well as the
   individual brackets. Missing pairs therefore fail the count check.
   A degenerate endpoint / unresolved cluster raises; there is no silent
   fallback. This is a numerical completeness check for the requested
   parameters, not a claim that sign scanning alone proves completeness
   for arbitrary coincident spectra.

6. Truncation is explicit: modes_per_order total radial eigenfunctions
   (including constants), orders 0,...,orders-1. The comparisons keep
   10 radial modes per order and 12 orders (0 through 11); closed-form
   checks keep 14 positive roots per order and 16 orders. The radial tail at each
   order is bounded by exp(-x_next(0)^2 D_out Delta/b^2) times that order's
   entire plane-wave norm. The omitted angular tail is bounded by its
   plane-wave norm (Parseval: total=1). These are nonnegative upper
   bounds, not differences of two small echoes. Gauss-Legendre rules
   integrate each region separately, with 64 then 128 nodes per region.
   quadrature_error is twice the echo
   change on doubling nodes, plus 2e-12 for root/roundoff error; also
   compare plane-wave norms. These are numerical error ESTIMATES, not
   interval arithmetic. Report truncation and quadrature separately.

7. Closed-form single-pore check, derived without quadrature. For B=j_l
   or J_n, c=l(l+1) or n^2, alpha a positive root B'(alpha)=0, the radial
   Lagrange identity gives, on the unit pore,
       I=alpha*B'(alpha)*B(Q)/(Q^2-alpha^2)
           - Q*B'(Q)*B(alpha)/(Q^2-alpha^2)
        = -Q*B'(Q)*B(alpha)/(Q^2-alpha^2).
   N=B(alpha)^2*(1-c/alpha^2)/2. Therefore each nonconstant term is
       2*p*w_order*alpha^2/(alpha^2-c)
          *[Q*B'(Q)/(Q^2-alpha^2)]^2 exp(-alpha^2 D Delta/R^2).
   The constant contributes [3*j_1(Q)/Q]^2 (sphere), [2*J_1(Q)/Q]^2
   (disk). This is the restricted-pore Callaghan series; its long-time
   limit is the squared form factor (Balinov et al. 1993). At resonance
   take the continuous limit -(alpha^2-c)*B(alpha)/(2*alpha^2).
   This direct q-space expression supplies the independent closed-form
   checks, rather than importing the package's restricted SGP routine.

Background: Grebenkov, Nguyen & Li (2014), JMR 248:153-163, spectral SGP
construction, https://pmc.polytechnique.fr/pagesperso/dg/publi/2014_09.pdf .
The radial matching, count check and overlap derivation above are explicit
for the present two-region geometry; that paper is background, not a
claim that its 1D formula describes this sphere/cylinder.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import special
from scipy.optimize import brentq


@dataclass(frozen=True)
class Problem:
    geometry: str
    radius: float
    outer_radius: float
    diffusion_in: float
    diffusion_out: float
    permeability: float


@dataclass(frozen=True)
class Echo:
    value: float
    truncation_error: float
    quadrature_error: float

    @property
    def error(self):
        return self.truncation_error + self.quadrature_error


def bessel(geometry, order, x, *, singular, derivative):
    if geometry == "sphere":
        fun = special.spherical_yn if singular else special.spherical_jn
        return fun(order, x, derivative=derivative)
    if geometry == "cylinder":
        if derivative:
            fun = special.yvp if singular else special.jvp
            return fun(order, x, 1)
        fun = special.yv if singular else special.jv
        return fun(order, x)
    raise ValueError(geometry)


def scan_roots(fun, stop, step):
    grid = np.linspace(1e-5, stop, int(np.ceil(stop / step)) + 1)
    values = fun(grid)
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("nonfinite determinant")
    cross = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    roots = [brentq(fun, grid[j], grid[j + 1], xtol=2e-12) for j in cross]
    return np.asarray(roots)


def boundary_values(geometry, a, d, order, x):
    def B(x, *, singular, derivative):
        return bessel(geometry, order, x, singular=singular, derivative=derivative)
    z = x / np.sqrt(d)
    u = B(z * a, singular=False, derivative=False)
    up = z * B(z * a, singular=False, derivative=True)
    jp = B(x, singular=False, derivative=True)
    yp = B(x, singular=True, derivative=True)
    v = (B(x * a, singular=False, derivative=False) * yp
         - B(x * a, singular=True, derivative=False) * jp)
    vp = x * (B(x * a, singular=False, derivative=True) * yp
              - B(x * a, singular=True, derivative=True) * jp)
    us, vs = np.hypot(u, up), np.hypot(v, vp)
    return u / us, up / us, v / vs, vp / vs, us, vs

@lru_cache(maxsize=None)
def limiting_spectra(geometry, a, d, order, count):
    def continuous(x):
        u, up, v, vp, _, _ = boundary_values(geometry, a, d, order, x)
        return u * vp - d * up * v
    # Explicit search extent; insufficient total count is an error.
    stop = (count + order + 6) * np.pi / (a / np.sqrt(d) + 1 - a)
    inside = scan_roots(lambda x: boundary_values(geometry, a, d, order, x)[1], stop, 0.025)
    outside = scan_roots(lambda x: boundary_values(geometry, a, d, order, x)[3], stop, 0.025)
    sealed = [(float(x), "inner") for x in inside] + [(float(x), "outer") for x in outside]
    if order == 0:
        sealed += [(0.0, "inner"), (0.0, "outer")]
    sealed.sort()
    limits0 = np.array([x for x, _ in sealed])
    # Check the disconnected limits independently with the finer grid.
    for fun, roots in ((lambda x: boundary_values(geometry, a, d, order, x)[1], inside),
                       (lambda x: boundary_values(geometry, a, d, order, x)[3], outside)):
        fine = scan_roots(fun, stop, 0.0125)
        if len(fine) != len(roots) or not np.allclose(fine, roots, rtol=0, atol=1e-9):
            raise ArithmeticError("unresolved sealed roots")
    if len(limits0) < count + 3:
        raise ArithmeticError("insufficient limiting roots")
    limit_stop = limits0[count + 2]
    inf_roots = scan_roots(continuous, limit_stop, 0.025)
    inf_fine = scan_roots(continuous, limit_stop, 0.0125)
    if len(inf_roots) != len(inf_fine) or not np.allclose(inf_roots, inf_fine, rtol=0, atol=1e-9):
        raise ArithmeticError("unresolved continuous roots")
    limits_inf = np.r_[0.0, inf_roots] if order == 0 else inf_roots
    for j in range(count + 1):
        if not (limits0[j] - 1e-9 <= limits_inf[j] <= limits0[j + 1] + 1e-9):
            raise ArithmeticError(("limiting interlacing failed", order, j))
    if limits0[count] <= limits_inf[count - 1] + 1e-9:
        raise ArithmeticError("degenerate count cutoff; choose another explicit mode count")
    cutoff = (limits0[count] + limits_inf[count - 1]) / 2
    assert np.count_nonzero(limits0 < cutoff) == count
    assert np.count_nonzero(limits_inf < cutoff) == count
    return sealed, limits0, limits_inf, cutoff

class Spectrum:
    """Dimensionless analytic Bessel modes with explicit completeness checks."""

    def __init__(self, problem, *, modes_per_order, orders):
        self.problem = problem
        self.a = problem.radius / problem.outer_radius
        self.d = problem.diffusion_in / problem.diffusion_out
        self.h = problem.permeability * problem.outer_radius / problem.diffusion_out
        self.dimension = {"sphere": 3, "cylinder": 2}[problem.geometry]
        self.orders = orders
        self.modes_per_order = modes_per_order
        if not (0 < self.a < 1 and self.d > 0 and self.h >= 0):
            raise ValueError(problem)
        self.modes = []
        self.tail_roots = []
        self.counts = []
        for order in range(orders):
            self.build_order(order)

    def b(self, order, x, *, singular, derivative):
        return bessel(self.problem.geometry, order, x, singular=singular, derivative=derivative)

    def boundary(self, order, x):
        """Scaled (u,u',v,v',inner scale,outer scale), primes in s."""
        return boundary_values(self.problem.geometry, self.a, self.d, order, x)

    def determinant(self, order, x, h):
        u, up, v, vp, _, _ = self.boundary(order, x)
        continuous = u * vp - self.d * up * v
        if np.isinf(h):
            return continuous
        return self.d * up * vp + h * continuous

    def build_order(self, order):
        count = self.modes_per_order
        sealed, limits0, limits_inf, cutoff = limiting_spectra(
            self.problem.geometry, self.a, self.d, order, count)
        self.tail_roots.append(limits0[count])
        if self.h == 0:
            modes = sealed[:count]
        elif np.isinf(self.h):
            modes = [(float(x), "coupled") for x in limits_inf[:count]]
        else:
            roots = scan_roots(lambda x: self.determinant(order, x, self.h), cutoff, 0.025)
            fine = scan_roots(lambda x: self.determinant(order, x, self.h), cutoff, 0.0125)
            if len(roots) != len(fine) or not np.allclose(roots, fine, rtol=0, atol=1e-9):
                raise ArithmeticError("unresolved finite-permeability roots")
            full = np.r_[0.0, roots] if order == 0 else roots
            if len(full) != count:
                raise ArithmeticError(("root count failed", order, len(full), count))
            if np.any(full < limits0[:count] - 1e-9) or np.any(full > limits_inf[:count] + 1e-9):
                raise ArithmeticError("finite-permeability interlacing failed")
            modes = [(float(x), "coupled") for x in full]
        self.counts.append((count, count, len(modes)))
        self.modes.append(modes)

    def radial(self, order, root, pool, s_in, s_out):
        if root == 0:
            return (np.ones_like(s_in) if pool != "outer" else np.zeros_like(s_in),
                    np.ones_like(s_out) if pool != "inner" else np.zeros_like(s_out))
        u, up, v, _, us, vs = self.boundary(order, root)
        if pool == "inner":
            A, B = 1.0, 0.0
        elif pool == "outer":
            A, B = 0.0, 1.0
        elif np.isinf(self.h):
            A, B = v, u
        else:
            A, B = self.h * v, self.d * up + self.h * u
        scale = np.hypot(A, B)
        A, B = A / scale, B / scale
        inner = A * self.b(order, root * s_in / np.sqrt(self.d), singular=False, derivative=False) / us
        outer = B * (self.b(order, root * s_out, singular=False, derivative=False)
                     * self.b(order, root, singular=True, derivative=True)
                     - self.b(order, root * s_out, singular=True, derivative=False)
                     * self.b(order, root, singular=False, derivative=True)) / vs
        return inner, outer

    def integrate(self, q, Delta, *, nodes, start):
        if start not in ("uniform", "intra") or (start == "intra" and self.h != 0):
            raise ValueError("intra start is supported only for a sealed membrane")
        z, w = np.polynomial.legendre.leggauss(nodes)
        si = (z + 1) * self.a / 2
        so = self.a + (z + 1) * (1 - self.a) / 2
        wi = w * self.a / 2 * si ** (self.dimension - 1)
        wo = w * (1 - self.a) / 2 * so ** (self.dimension - 1)
        norm = self.dimension / (self.a ** self.dimension if start == "intra" else 1)
        Q = q * self.problem.outer_radius
        time = Delta * self.problem.diffusion_out / self.problem.outer_radius ** 2
        value, radial_tail, plane_norm = 0.0, 0.0, 0.0
        for order, modes in enumerate(self.modes):
            weight = 2 * order + 1 if self.dimension == 3 else (1 if order == 0 else 2)
            bi = self.b(order, Q * si, singular=False, derivative=False)
            bo = self.b(order, Q * so, singular=False, derivative=False)
            power = norm * weight * (np.dot(wi, bi * bi) + (np.dot(wo, bo * bo) if start == "uniform" else 0))
            plane_norm += power
            radial_tail += power * np.exp(-self.tail_roots[order] ** 2 * time)
            for root, pool in modes:
                if start == "intra" and pool != "inner":
                    continue
                fi, fo = self.radial(order, root, pool, si, so)
                N = np.dot(wi, fi * fi) + np.dot(wo, fo * fo)
                I = np.dot(wi, fi * bi) + np.dot(wo, fo * bo)
                value += norm * weight * I * I / N * np.exp(-root * root * time)
        return value, radial_tail + max(0.0, 1 - plane_norm), plane_norm

    def echo(self, q, Delta, *, nodes, start):
        coarse = self.integrate(q, Delta, nodes=nodes, start=start)
        fine = self.integrate(q, Delta, nodes=2 * nodes, start=start)
        error = 2 * (abs(coarse[0] - fine[0]) + abs(coarse[2] - fine[2])) + 2e-12
        return Echo(fine[0], fine[1], error)


def single_pore(geometry, q, Delta, radius, diffusion, *, modes_per_order, orders):
    """Closed-form overlap series from step 7; q is angular rad/m."""
    p = {"sphere": 3, "cylinder": 2}[geometry]
    Q = q * radius
    if Q == 0:
        return 1.0
    def B(order, x, derivative):
        return bessel(geometry, order, x, singular=False, derivative=derivative)
    form = p * B(1, Q, False) / Q
    value = form * form
    for order in range(orders):
        roots = scan_roots(lambda x: B(order, x, True), (modes_per_order + order + 3) * np.pi, 0.025)[:modes_per_order]
        if len(roots) != modes_per_order:
            raise ArithmeticError("single-pore root count")
        c = order * (order + 1) if p == 3 else order ** 2
        weight = 2 * order + 1 if p == 3 else (1 if order == 0 else 2)
        for root in roots:
            if abs(Q * Q - root * root) < 1e-9:
                overlap = -(root * root - c) * B(order, root, False) / (2 * root * root)
            else:
                overlap = Q * B(order, Q, True) / (Q * Q - root * root)
            value += 2 * p * weight * root ** 2 / (root ** 2 - c) * overlap ** 2 * np.exp(-root ** 2 * diffusion * Delta / radius ** 2)
    return float(value)
