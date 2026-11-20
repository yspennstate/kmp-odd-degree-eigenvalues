# Odd-Degree Eigenvalues of the KMP Process on Hypergraphs

Yitzchak Shmalo

For a weighted hypergraph on $n$ vertices let $\omega_k$ be the smallest eigenvalue of the discrete
Kipnis-Marchioro-Presutti Laplacian with $k$ particles on the orthogonal complement of the image of the space of
$k-1$ particles. Alon and Puder conjectured that $\omega_1\le\omega_3\le\omega_5\le\cdots$ and
$\omega_2\le\omega_4\le\cdots$; Caputo, Quattropani and Sau proved the even chain. The odd chain
fails: on eleven vertices, three hyperedges with weights 17, 715, 715 give $\omega_3=2915/4>\omega_5$.

When the complements of the hyperedges are pairwise disjoint, $\omega_k$ is the total weight minus the largest
eigenvalue of an explicit matrix with one row per hyperedge. For odd $k$ the step $\omega_k\le\omega_{k+2}$ then holds
for all weights exactly when an explicit inequality between the sizes of the complements holds, and for $k=1$ this
inequality always holds, so $\omega_1\le\omega_3$ for all these hypergraphs.

`main.tex` and `main.pdf` are the paper.

## The check program

`check_kmp_example.py` needs NumPy and SymPy:

    python check_kmp_example.py

It computes the characteristic polynomials of $M_1,\dots,M_5$ for the example exactly, computes
$\omega_1,\dots,\omega_5$ directly from the definition of the Laplacian and of the pure subspace and compares them with
the formula of Theorem 1.3, makes the same comparison for seven other hypergraphs with disjoint retained groups on four
to six vertices, and compares the size inequality of Theorem 1.5 with the eigenvalues of $H_k$ for two to four retained
groups on at most 24 vertices and every odd $k\le 11$ (47,178 cases). Its output is `check_output.txt`.
