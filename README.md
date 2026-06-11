# Entanglement-Certification-and-Entropy-Bounds-in-the-Asymmetric-CHSH-Scenario
## Numerical setting

The SDP calculations use NCPOL2SDPA with MOSEK. The Bell constraint is imposed
in normalized form \(v=S_\alpha/[2(1+\alpha)]\). The figures in the paper report
the corresponding unnormalized Bell value \(S_\alpha\).


The implementation uses `level=2` in NCPOL2SDPA together with additional ABZ-type
monomials in order to represent the Gauss--Radau auxiliary-operator terms
\(Z^\dagger Z\), \(ZZ^\dagger\), and \(A Z^\dagger Z\).

## Main scripts

- `compute_npa_alpha_0_9.py`: standalone computation for the \(H(A|E)\) boundary at \(\alpha=0.9\).
- `Main_Asym_CHSH.py`: computes asymmetric-CHSH entropy and noise-model data.
- `get_data_achsh_from_dat.py`: post-processes precomputed entropy data into visibility/efficiency key-rate data.
