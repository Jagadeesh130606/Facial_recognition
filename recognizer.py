"""
recognizer.py
─────────────
STEPS 6 & 7 (Inference)

Nearest-neighbour classification in the eigenface subspace:
  1. Capture frame from webcam
  2. Detect and crop face → flat vector  b  of dimension d
  3. Project:  Ω = Eᵀ(b − mean)         → coordinates in ℝᵏ
  4. Find training sample with minimum Euclidean distance in eigenspace
     (equivalent to the Least Squares closest match)
  5. Overlay prediction on live video
"""

import cv2
import numpy as np
from la_pipeline import LinearAlgebraPipeline, IMG_SIZE

CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DIST_THRESHOLD = 4500.0   # tune this: faces farther than threshold → "Unknown"


class FaceRecognizer:

    def __init__(self, pipeline: LinearAlgebraPipeline):
        self.pipeline = pipeline
        self.detector = cv2.CascadeClassifier(CASCADE)

    # ── Public ───────────────────────────────────────────────────────────────

    def recognize_from_camera(self) -> None:
        """Open webcam, recognize faces live. Press Q to quit."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [ERROR] Cannot open camera.")
            return

        print("  Live recognition active — press Q to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                crop    = grey[y:y+h, x:x+w]
                resized = cv2.resize(crop, IMG_SIZE).astype(np.float32)
                vec     = resized.flatten()

                label, dist, coords = self._predict(vec)

                # ── draw bounding box + label ────────────────────────────────
                color = (0, 200, 0) if label != "Unknown" else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label}  ({dist:.0f})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                # ── mini eigenspace coords display ───────────────────────────
                coord_str = "  ".join([f"ω{i+1}:{v:.1f}" for i, v in enumerate(coords[:3])])
                cv2.putText(frame, coord_str,
                            (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 50), 1)

            cv2.putText(frame, "Eigenface Recognition — Press Q to quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.imshow("Face Recognition (Linear Algebra Pipeline)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ── Private ──────────────────────────────────────────────────────────────

    def _predict(self, vec: np.ndarray):
        """
        Project vec into eigenspace, find nearest neighbour.
        Returns (label, distance, coords).
        """
        p      = self.pipeline
        coords = p.project(vec)               # shape (k,)  — Step 6

        # Euclidean distance to every training projection — Step 7 (LS nearest)
        diffs  = p.projections - coords       # (n, k)
        dists  = np.linalg.norm(diffs, axis=1)
        idx    = int(np.argmin(dists))
        dist   = float(dists[idx])

        label  = p.labels[idx] if dist < DIST_THRESHOLD else "Unknown"
        return label, dist, coords
