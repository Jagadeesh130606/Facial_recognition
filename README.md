# Face Recognition Using Linear Algebra Pipeline
### PES University — UE24MA241B: Linear Algebra and Its Applications

---

## Project Overview

This project builds a **face recognition system** using the Eigenfaces method, which applies every stage of the prescribed Linear Algebra workflow — from raw camera frames all the way to live identity prediction.

---

## Linear Algebra Pipeline (all 9 steps)

| Step | LA Concept | How It Is Used Here |
|------|-----------|---------------------|
| 1 | **Matrix Representation** | Each 64×64 face image is flattened into a row vector of dimension *d = 4096*. Stacking *n* images forms the data matrix **A ∈ ℝⁿˣᵈ** |
| 2 | **Matrix Simplification** (mean-centering) | Subtract the mean face vector from every row → centred matrix **Ã** (removes DC offset, analogous to RREF normalisation) |
| 3 | **Structure of the Space** | Compute rank(Ã), nullity, and covariance **C = ÃᵀÃ/n** to understand the face-space geometry |
| 4 | **Remove Redundancy** | Rank reveals the number of *linearly independent* face directions; dependent (redundant) samples are identified |
| 5 | **Orthogonalization** | Eigenvectors of **C** are mutually orthogonal (Spectral Theorem). The Gram–Schmidt process is demonstrated for verification |
| 6 | **Projection** | A query face is projected onto the eigenface subspace: **Ω = Eᵀ(b − mean)**, giving its coordinates in ℝᵏ |
| 7 | **Least Squares Prediction** | Best reconstruction: **x̂ = (EᵀE)⁻¹Eᵀb**; nearest-neighbour in eigenspace corresponds to the minimum-residual match |
| 8 | **Pattern Discovery** | Eigenvalues of the covariance matrix rank the eigenfaces by how much variance (facial pattern) each captures |
| 9 | **System Simplification** | Diagonalisation **C = EΛEᵀ**; retaining only the top-*k* eigenfaces compresses faces from 4096-D to *k*-D |

---

## Project Structure

```
face_recognition_project/
├── main.py              ← Entry point (menu-driven)
├── data_collection.py   ← Step 1: Webcam → face images saved as .npy
├── la_pipeline.py       ← Steps 1–9: Full Linear Algebra engine
├── recognizer.py        ← Steps 6 & 7 (inference): Live recognition
├── utils.py             ← Console formatting helpers
├── requirements.txt
└── faces_db/            ← Created automatically during collection
    └── <person_name>/
        ├── frame_0000.npy
        └── ...
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the program
```bash
python main.py
```

### 3. Workflow
```
Option 1  →  Collect training faces for each person (30 samples each, via webcam)
Option 2  →  Train the pipeline (runs all 9 LA steps, prints every stage)
Option 3  →  Recognize faces live from webcam
Option 4  →  Visualize eigenfaces + explained variance chart
Option 5  →  Exit
```

---

## Viva Preparation (Concept → Purpose → Outcome)

| Concept | Purpose | Outcome |
|---------|---------|---------|
| Data Matrix **A** | Represent images mathematically | Every face = a point in ℝ⁴⁰⁹⁶ |
| Mean-centering | Remove lighting/bias offset | Centred matrix Ã ready for PCA |
| Covariance matrix **C** | Measure how pixel intensities co-vary across faces | Symmetric positive-semidefinite matrix |
| Rank & Nullity | Find true dimensionality of face space | rank + nullity = 4096 confirmed |
| Eigenvectors of **C** | Find the principal "face directions" | Eigenfaces — orthogonal basis for face space |
| Gram–Schmidt | Verify / build orthogonal basis | Orthonormal eigenface basis |
| Projection | Compress face to *k* numbers | Efficient coordinate representation |
| Least Squares | Best approximate reconstruction / match | Minimum-error identity prediction |
| Eigenvalues | Rank importance of each eigenface | Top-*k* chosen; others discarded as noise |
| Diagonalisation **C = EΛEᵀ** | Simplify the system | 4096-D → *k*-D; compression ratio = 4096/*k* |

---

## Key Equations

```
Data matrix:          A  ∈ ℝⁿˣᵈ          (n faces, d = 64×64 = 4096 pixels)
Mean face:            μ  = (1/n) Σ aᵢ
Centred matrix:       Ã  = A − 1·μᵀ
Covariance:           C  = (1/n) ÃᵀÃ     (d×d — use surrogate L = ÃÃᵀ/n for efficiency)
Eigendecomposition:   C  = E Λ Eᵀ
Projection:           Ω  = Eᵀ(b − μ)
Least Squares:        x̂  = (EᵀE)⁻¹Eᵀb  ≈ Eᵀb  (since E is orthonormal)
Reconstruction:       b̂  = E x̂ + μ
Recognition:          identity = argmin‖Ωᵢ − Ω‖₂
```

---

## Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `IMG_SIZE` | 64×64 | Resolution of each face (d = 4096) |
| `n_components` (k) | 20 | Number of eigenfaces retained |
| `n_samples` | 30 | Photos collected per person |
| `DIST_THRESHOLD` | 4500 | Max eigenspace distance before "Unknown" |

Increase `k` for better accuracy; decrease for speed and compression.
