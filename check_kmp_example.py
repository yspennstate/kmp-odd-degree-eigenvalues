"""Checks for the paper on odd-degree eigenvalues of discrete KMP on hypergraphs.

1. Exact characteristic polynomials of the matrices M_k (k = 1..5) for the eleven-vertex example.
2. The least eigenvalue of L(Gamma, KMP_k) on PURE(n, k), computed directly in floating point
   (L = sum_B w_B (I - N_B) on the multiset basis, PURE = orthogonal complement of the image of
   iota(I) = sum_x (#x(I) + 1) (I + x)), compared with W - lambda_max(M_k).
3. The same comparison for other families of disjoint retained groups, k <= 9 (k <= 6 on six vertices).
4. Positive semidefiniteness of H_1 for all size patterns with at most four groups, n <= 24.
5. The size inequality of Theorem 1.5 against the eigenvalues of H_k for odd k <= 11.
Needs numpy and sympy.
"""
import itertools
import numpy as np
import sympy as sp


def states(n, k):
    out = []
    for c in itertools.combinations(range(n + k - 1), n - 1):
        prev, comp = -1, []
        for x in c:
            comp.append(x - prev - 1)
            prev = x
        comp.append(n + k - 1 - prev - 1)
        out.append(tuple(comp))
    return out


def pure_least(n, edges, k, prev):
    S = states(n, k)
    idx = {s: i for i, s in enumerate(S)}
    L = np.zeros((len(S), len(S)))
    for B, w in edges:
        kept = [v for v in range(n) if v not in B]
        classes = {}
        for i, s in enumerate(S):
            classes.setdefault(tuple(s[v] for v in kept), []).append(i)
        for mem in classes.values():
            for i in mem:
                L[i, i] += w
                for j in mem:
                    L[i, j] -= w / len(mem)
    J = np.zeros((len(S), len(prev)))
    for c, s in enumerate(prev):
        for x in range(n):
            t = list(s)
            t[x] += 1
            J[idx[tuple(t)], c] = s[x] + 1
    U, sv, _ = np.linalg.svd(J, full_matrices=True)
    Q = U[:, int((sv > 1e-9).sum()):]
    return np.linalg.eigvalsh(Q.T @ L @ Q)[0], S


def M_exact(n, sizes, weights, k):
    m = len(sizes)
    return sp.Matrix(m, m, lambda i, j: weights[i] if i == j else
                     weights[i] * (-1) ** k * sp.rf(sizes[j], k) / sp.rf(n - sizes[i], k))


def omega_formula(n, sizes, weights, k):
    M = np.array(M_exact(n, sizes, weights, k).evalf(), dtype=float)
    return sum(weights) - max(np.linalg.eigvals(M).real)


def edges_from(n, groups, weights):
    return [(tuple(v for v in range(n) if v not in g), w) for g, w in zip(groups, weights)]


x = sp.symbols("x")
n, sizes, weights = 11, [9, 1, 1], [17, 715, 715]
groups = [list(range(9)), [9], [10]]
W = sum(weights)
print("1. Eleven-vertex example: retained groups of sizes", sizes, "weights", weights, "W =", W)
for k in range(1, 6):
    M = M_exact(n, sizes, weights, k)
    print(f"   k={k}: det(xI - M_k) =", sp.factor(M.charpoly(x).as_expr()))
lam3 = sp.Rational(2873, 4)
lam5 = (10243 + sp.sqrt(10243 ** 2 - 56 * 133620)) / 28
assert sp.Rational(10243 ** 2 - 56 * 133620) > 9868 ** 2
print("   omega_3 =", W - lam3, " omega_5 =", sp.nsimplify(W - lam5), "=", sp.N(W - lam5, 15))
print("   omega_3 - omega_5 =", sp.N(lam5 - lam3, 15), "> 0")

print("2. Direct computation of the pure spectrum, same example:")
prev = [tuple([0] * n)]
for k in range(1, 6):
    b, prev = pure_least(n, edges_from(n, groups, weights), k, prev)
    f = omega_formula(n, sizes, weights, k)
    print(f"   k={k}: direct {b:.10f}  formula {f:.10f}  diff {abs(b - f):.1e}")

print("3. Other families (direct vs formula), k <= 9 on at most five vertices, k <= 6 on six:")
cases = [
    (4, [[0, 1], [2, 3]], [1, 5]),
    (4, [[0], [1], [2], [3]], [1, 2, 3, 7]),
    (5, [[0, 1, 2], [3, 4]], [17, 2]),
    (5, [[0, 1, 2], [3], [4]], [17, 40, 40]),
    (5, [[0], [1], [2]], [3, 1, 2]),
    (5, [[0, 1], [2], [3], [4]], [1, 9, 2, 5]),
    (6, [[0, 1, 2], [3], [5]], [2, 7, 3]),
]
worst = 0.0
for nn, gg, ww in cases:
    prev = [tuple([0] * nn)]
    kmax = 9 if nn <= 5 else 6
    for k in range(1, kmax + 1):
        b, prev = pure_least(nn, edges_from(nn, gg, ww), k, prev)
        f = omega_formula(nn, [len(g) for g in gg], ww, k)
        worst = max(worst, abs(b - f))
    print(f"   n={nn} groups={gg} weights={ww} k<={kmax}: max diff so far {worst:.1e}")
assert worst < 1e-8

print("4. H_1 positive semidefinite for every size pattern with 2..4 groups, n <= 24:")
bad = 0
count = 0
for nn in range(3, 25):
    for m in range(2, 5):
        for a in itertools.combinations_with_replacement(range(1, nn), m):
            if sum(a) > nn:
                continue
            u = np.array([np.sqrt((s + 1) * (s + 2) / ((nn - s + 1) * (nn - s + 2))) for s in a])
            H = np.outer(u, u)
            np.fill_diagonal(H, 1.0)
            count += 1
            if np.linalg.eigvalsh(H)[0] < -1e-12:
                bad += 1
print(f"   patterns checked {count}, failures {bad}")
assert bad == 0

print("5. The size inequality of Theorem 1.5 against the eigenvalues of H_k, odd k <= 11,")
print("   2..4 retained groups, n <= 24:")
n_s, a_s, k_s = sp.symbols("n a k", positive=True)
u2 = (a_s + k_s) * (a_s + k_s + 1) / ((n_s - a_s + k_s) * (n_s - a_s + k_s + 1))
ident = sp.simplify(1 - u2 - (n_s - 2 * a_s) * (n_s + 2 * k_s + 1) / ((n_s - a_s + k_s) * (n_s - a_s + k_s + 1)))
print(f"   identity (4.1) for 1 - u^2: {ident == 0}")
assert ident == 0


def theta(nn, kk, x):
    return sp.Rational((x + kk) * (x + kk + 1), nn - 2 * x)


total = 0
mismatch = 0
not_psd = {}
for nn in range(3, 25):
    for m in range(2, 5):
        for a in itertools.combinations_with_replacement(range(1, nn), m):
            if sum(a) > nn:
                continue
            for kk in range(1, 12, 2):
                u = np.array([np.sqrt((s + kk) * (s + kk + 1) / ((nn - s + kk) * (nn - s + kk + 1))) for s in a])
                H = np.outer(u, u)
                np.fill_diagonal(H, 1.0)
                psd = np.linalg.eigvalsh(H)[0] >= -1e-10
                a1 = max(a)
                if 2 * a1 <= nn:
                    crit = True
                else:
                    rest = list(a)
                    rest.remove(a1)
                    crit = sum(theta(nn, kk, x) for x in rest) <= theta(nn, kk, nn - a1)
                total += 1
                if crit != psd:
                    mismatch += 1
                if not psd:
                    not_psd[kk] = not_psd.get(kk, 0) + 1
print(f"   cases {total}, disagreements {mismatch}, not positive semidefinite by k: {dict(sorted(not_psd.items()))}")
assert mismatch == 0
