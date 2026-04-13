"""
la_pipeline.py
──────────────
THE CORE LINEAR ALGEBRA PIPELINE
Following the PES University UE24MA241B workflow diagram exactly.

Steps implemented
─────────────────
1  Matrix Representation     — load images → data matrix  A ∈ ℝⁿˣᵈ
2  Matrix Simplification     — mean-centre  A  (analogous to RREF / centering)
3  Structure of the Space    — covariance matrix, rank, nullity
4  Remove Redundancy         — check linear independence of samples
5  Orthogonalization         — Gram–Schmidt on a small basis; eigenvectors are
                               guaranteed orthogonal (spectral theorem)
6  Projection                — project new face onto eigenface subspace
7  Least Squares Prediction  — x̂ = (AᵀA)⁻¹Aᵀb  for reconstruction
8  Pattern Discovery         — eigenvalues / eigenvectors of covariance matrix
9  System Simplification     — diagonalization, keep top-k eigenfaces
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from utils import print_section, print_step


IMG_SIZE   = (64, 64)
IMG_DIM    = IMG_SIZE[0] * IMG_SIZE[1]   # d = 4 096


# ─────────────────────────────────────────────────────────────────────────────
class LinearAlgebraPipeline:

    def __init__(self, n_components: int = 20):
        """
        n_components : number of eigenfaces (top-k) to retain.
        """
        self.k           = n_components
        self.is_trained  = False

        # Set by train()
        self.mean_face   : Optional[np.ndarray] = None   # shape (d,)
        self.eigenfaces  : Optional[np.ndarray] = None   # shape (k, d)  — rows are eigenfaces
        self.eigenvalues : Optional[np.ndarray] = None   # shape (k,)
        self.projections : Optional[np.ndarray] = None   # shape (n, k)  — training coords
        self.labels      : List[str]             = []
        self.rank        : int                   = 0
        self.nullity     : int                   = 0


    # ═════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════════════

    def train(self, data_dir: str = "faces_db") -> None:
        """Run the complete pipeline on all saved face images."""

        # ── STEP 1 ── Matrix Representation ──────────────────────────────────
        print_step(1, "Matrix Representation")
        A, labels = self._load_data(data_dir)
        if A is None:
            print("  [ERROR] No training data found. Collect faces first (option 1).")
            return
        n, d = A.shape
        print(f"  Data matrix  A  :  {n} samples × {d} features  (n={n}, d={d})")
        print(f"  Labels collected : {set(labels)}")

        # ── STEP 2 ── Matrix Simplification (mean-centering) ──────────────────
        print_step(2, "Matrix Simplification — Mean-Centering (analogous to RREF)")
        mean_face = np.mean(A, axis=0)           # shape (d,)
        A_centred = A - mean_face                # subtract mean from every row
        print(f"  Mean face vector computed. Shape: {mean_face.shape}")
        print(f"  Centred matrix  Ã  shape : {A_centred.shape}")
        print(f"  ‖mean_face‖ = {np.linalg.norm(mean_face):.3f}")

        # ── STEP 3 ── Structure of the Space ─────────────────────────────────
        print_step(3, "Structure of the Space — Rank, Nullity, Covariance")
        rank    = int(np.linalg.matrix_rank(A_centred))
        nullity = d - rank
        print(f"  rank(Ã)    = {rank}")
        print(f"  nullity(Ã) = {nullity}  (by Rank–Nullity theorem: rank + nullity = d)")
        # Covariance matrix  C = (1/n) · ÃᵀÃ  — shape (d, d) is huge (4096²)
        # Trick: compute  L = (1/n) · Ã·Ãᵀ  shape (n, n) — same non-zero eigenvalues
        L = (1.0 / n) * (A_centred @ A_centred.T)   # shape (n, n)
        print(f"  Surrogate covariance matrix  L = ÃÃᵀ/n  :  shape {L.shape}")

        # ── STEP 4 ── Remove Redundancy ───────────────────────────────────────
        print_step(4, "Remove Redundancy — Linear Independence")
        # The number of linearly independent directions = rank
        # Dependent rows = n - rank
        n_dependent = n - rank
        print(f"  Independent sample directions : {rank}")
        print(f"  Dependent (redundant) samples : {n_dependent}")

        # ── STEP 5 ── Orthogonalization (Gram–Schmidt / Spectral Theorem) ─────
        print_step(5, "Orthogonalization — Eigenvectors are Orthogonal (Spectral Theorem)")
        # Eigendecompose the (n×n) surrogate matrix L
        eigenvalues_L, eigenvectors_L = np.linalg.eigh(L)
        # eigh returns ascending order — reverse for descending
        idx              = np.argsort(eigenvalues_L)[::-1]
        eigenvalues_L    = eigenvalues_L[idx]
        eigenvectors_L   = eigenvectors_L[:, idx]

        # Recover true eigenvectors of the (d×d) covariance from eigenvectors of L
        # v_i = Ãᵀ u_i  (normalized)
        k_use = min(self.k, rank, n)
        eigenfaces = np.zeros((k_use, d), dtype=np.float32)
        eigenvals  = np.zeros(k_use,       dtype=np.float32)
        for i in range(k_use):
            u_i   = eigenvectors_L[:, i]                 # left eigenvector, shape (n,)
            v_i   = A_centred.T @ u_i                    # map back to feature space
            norm  = np.linalg.norm(v_i)
            if norm > 1e-10:
                v_i /= norm
            eigenfaces[i] = v_i.astype(np.float32)
            eigenvals[i]  = float(eigenvalues_L[i])

        # Quick Gram–Schmidt verification (printed, not re-applied — eigh already orthogonal)
        self._verify_orthogonality(eigenfaces[:5], label="first 5 eigenfaces")

        print(f"  Top-{k_use} eigenfaces computed. Each eigenface shape: {eigenfaces[0].shape}")

        # ── STEP 6 ── Projection ──────────────────────────────────────────────
        print_step(6, "Projection — Project Training Faces onto Eigenface Subspace")
        # Ω_i = Eᵀ (a_i − mean)   where E = eigenfaces matrix (k×d)
        projections = (A_centred @ eigenfaces.T)     # shape (n, k)
        print(f"  Projection matrix shape : {projections.shape}  (n × k)")
        print(f"  Each face is now a point in ℝ^{k_use}  instead of  ℝ^{d}")

        # ── STEP 7 ── Least Squares ───────────────────────────────────────────
        print_step(7, "Prediction / Approximation — Least Squares Reconstruction")
        # Demonstrate LS on the first training image
        b       = A_centred[0]                        # original centred sample, shape (d,)
        E       = eigenfaces.T                        # shape (d, k)
        # x̂ = (EᵀE)⁻¹Eᵀ b  — since E columns are orthonormal, EᵀE ≈ I  → x̂ = Eᵀ b
        ETE     = E.T @ E                             # (k, k)
        ETb     = E.T @ b                             # (k,)
        x_hat   = np.linalg.lstsq(ETE, ETb, rcond=None)[0]
        b_hat   = E @ x_hat                           # reconstruction in face space
        ls_err  = np.linalg.norm(b - b_hat)
        print(f"  Least squares solution  x̂  shape : {x_hat.shape}")
        print(f"  Reconstruction error  ‖b − Eẋ‖ = {ls_err:.4f}")

        # ── STEP 8 ── Pattern Discovery — Eigenvalues ─────────────────────────
        print_step(8, "Pattern Discovery — Eigenvalues & Dominant Patterns")
        total_var    = float(np.sum(eigenvals))
        explained    = (eigenvals / total_var * 100) if total_var > 0 else eigenvals * 0
        cumulative   = np.cumsum(explained)
        print(f"  Top-{k_use} eigenvalues:")
        for i in range(min(5, k_use)):
            print(f"    λ_{i+1} = {eigenvals[i]:10.3f}  |  "
                  f"explains {explained[i]:5.2f}%  |  cumulative {cumulative[i]:5.2f}%")
        print(f"  ...")
        print(f"  Total variance explained by {k_use} eigenfaces: {cumulative[-1]:.2f}%")

        # ── STEP 9 ── System Simplification — Diagonalization ─────────────────
        print_step(9, "System Simplification — Diagonalization")
        # The covariance C = E · Λ · Eᵀ  (spectral decomposition)
        Lambda = np.diag(eigenvals)                  # diagonal matrix of eigenvalues
        print(f"  Diagonal matrix Λ  (top-{k_use}) shape : {Lambda.shape}")
        print(f"  C ≈ E · Λ · Eᵀ  — system reduced from d={d} to k={k_use} dimensions")
        print(f"  Compression ratio : {d}/{k_use} = {d/k_use:.1f}×")

        # ── Store results ─────────────────────────────────────────────────────
        self.mean_face   = mean_face
        self.eigenfaces  = eigenfaces
        self.eigenvalues = eigenvals
        self.projections = projections
        self.labels      = labels
        self.rank        = rank
        self.nullity     = nullity
        self.k           = k_use
        self.is_trained  = True
        self._A_centred  = A_centred       # kept for LS demo

        print("\n  ✔  Pipeline training complete.\n")


    def project(self, face_vec: np.ndarray) -> np.ndarray:
        """
        STEP 6 (inference) — project a new (flattened, float) face onto the
        eigenface subspace.

        face_vec : shape (d,) raw pixel vector
        returns  : shape (k,) coordinate vector in eigenspace
        """
        centred = face_vec.astype(np.float32) - self.mean_face
        return centred @ self.eigenfaces.T          # (k,)


    def reconstruct(self, coords: np.ndarray) -> np.ndarray:
        """
        STEP 7 — reconstruct a face from its eigenspace coordinates (LS inverse).
        coords : shape (k,)
        returns: shape (d,) reconstructed face pixel vector
        """
        return (coords @ self.eigenfaces) + self.mean_face


    def visualize(self) -> None:
        """Show eigenfaces and explained-variance plot."""
        if not self.is_trained:
            print("  Not trained yet.")
            return

        n_show = min(self.k, 16)
        cols   = 4
        rows   = (n_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
        fig.suptitle(f"Top-{n_show} Eigenfaces  (Step 8 & 9 — Pattern Discovery)",
                     fontsize=13, fontweight="bold")

        for i, ax in enumerate(axes.flat):
            if i < n_show:
                ef   = self.eigenfaces[i].reshape(IMG_SIZE)
                ef_n = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
                ax.imshow(ef_n, cmap="bone")
                ax.set_title(f"EF-{i+1}\nλ={self.eigenvalues[i]:.1f}", fontsize=7)
            ax.axis("off")
        plt.tight_layout()

        # Explained variance curve
        total_var  = float(np.sum(self.eigenvalues))
        explained  = self.eigenvalues / total_var * 100
        cumulative = np.cumsum(explained)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(range(1, self.k + 1), explained,  color="steelblue", alpha=0.7, label="Individual %")
        ax2.plot(range(1, self.k + 1), cumulative, color="crimson",   marker="o", label="Cumulative %")
        ax2.axhline(95, linestyle="--", color="grey", alpha=0.7, label="95% threshold")
        ax2.set_xlabel("Eigenface index")
        ax2.set_ylabel("Variance explained (%)")
        ax2.set_title("Step 8 — Eigenvalue Analysis: Explained Variance per Eigenface")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _load_data(data_dir: str) -> Tuple[Optional[np.ndarray], List[str]]:
        """Load all .npy face files, flatten and stack into matrix A."""
        if not os.path.isdir(data_dir):
            return None, []

        rows, labels = [], []
        for person in sorted(os.listdir(data_dir)):
            person_dir = os.path.join(data_dir, person)
            if not os.path.isdir(person_dir):
                continue
            for fname in sorted(os.listdir(person_dir)):
                if not fname.endswith(".npy"):
                    continue
                img = np.load(os.path.join(person_dir, fname))
                rows.append(img.flatten().astype(np.float32))
                labels.append(person)

        if not rows:
            return None, []
        return np.array(rows, dtype=np.float32), labels


    @staticmethod
    def _verify_orthogonality(vectors: np.ndarray, label: str = "") -> None:
        """Print dot products between pairs — should be ≈ 0 for orthogonal sets."""
        n = len(vectors)
        max_off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dp = float(np.dot(vectors[i], vectors[j]))
                max_off = max(max_off, abs(dp))
        print(f"  Orthogonality check ({label}): max|vᵢ·vⱼ| = {max_off:.2e}  "
              f"({'✔ orthogonal' if max_off < 1e-4 else '⚠ not perfectly orthogonal'})")
