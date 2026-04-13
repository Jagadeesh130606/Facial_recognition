"""
data_collection.py
──────────────────
STEP 1 — Real-World Data  →  Matrix Representation

Uses OpenCV to open the webcam, detects faces with a Haar cascade, crops
them to a standard size (64×64 grey), and saves them as .npy files so the
Linear Algebra Pipeline can load them as rows of a data matrix.

Each saved image is one *observation vector* of dimension d = 64*64 = 4096.
Stacking n such vectors gives the data matrix  A ∈ ℝⁿˣᵈ.
"""

import os
import cv2
import numpy as np

IMG_SIZE  = (64, 64)   # every face cropped to 64×64 pixels → d = 4 096
CASCADE   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def collect_faces(label: str, n_samples: int = 30, data_dir: str = "faces_db") -> None:
    """
    Open webcam, detect faces, collect `n_samples` crops for the given label.
    Saved path: faces_db/<label>/frame_XXXX.npy
    """
    detector = cv2.CascadeClassifier(CASCADE)
    save_dir = os.path.join(data_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera. Check permissions / device index.")
        return

    count     = 0
    existing  = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
    print(f"\n  [INFO] Collecting {n_samples} samples for '{label}'.")
    print("  Press  Q  to quit early.\n")

    while count < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        grey   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = detector.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # ── crop & resize ───────────────────────────────────────────────
            face_crop = grey[y:y+h, x:x+w]
            face_resz = cv2.resize(face_crop, IMG_SIZE)

            # ── save as flat float vector (= one row in data matrix A) ──────
            idx  = existing + count
            path = os.path.join(save_dir, f"frame_{idx:04d}.npy")
            np.save(path, face_resz.astype(np.float32))
            count += 1

            # ── visual feedback ─────────────────────────────────────────────
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}  [{count}/{n_samples}]",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            break  # one face per frame is enough

        cv2.putText(frame, "Collecting faces — press Q to stop",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Face Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  [DONE] Saved {count} samples → '{save_dir}'\n")
