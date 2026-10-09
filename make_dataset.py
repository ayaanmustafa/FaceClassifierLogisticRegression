"""Generate synthetic two-class face dataset for classical CV demo."""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)

PARAMS = {
    "person_a": dict(face_axes=(92, 122), eye_dx=38, eye_w=26, eye_h=15,
                     brow_thick=5, brow_angle=4, mouth_w=64, mouth_h=12,
                     iris_r=9, iris_bgr=(60, 90, 170), skin_bgr=(200, 214, 235)),
    "person_b": dict(face_axes=(112, 114), eye_dx=28, eye_w=22, eye_h=13,
                     brow_thick=10, brow_angle=18, mouth_w=42, mouth_h=10,
                     iris_r=8, iris_bgr=(90, 140, 110), skin_bgr=(192, 208, 232)),
}

def draw_face(p, rng, out_path):
    W, H = 320, 380
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    cx, cy = W // 2 + int(rng.integers(-8, 9)), H // 2 + int(rng.integers(-8, 9))
    ax = p["face_axes"][0] + int(rng.integers(-6, 7))
    ay = p["face_axes"][1] + int(rng.integers(-6, 7))
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, p["skin_bgr"], -1)
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (90, 90, 90), 2)
    # eyes
    ey = cy - 30 + int(rng.integers(-4, 5))
    for side in (-1, 1):
        ex = cx + side * (p["eye_dx"] + int(rng.integers(-3, 4)))
        cv2.ellipse(img, (ex, ey), (p["eye_w"], p["eye_h"]), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (ex, ey), (p["eye_w"], p["eye_h"]), 0, 0, 360, (40, 40, 40), 2)
        cv2.circle(img, (ex, ey), p["iris_r"], p["iris_bgr"], -1)
        cv2.circle(img, (ex, ey), 3, (10, 10, 10), -1)
        # eyebrow above eye
        ang = p["brow_angle"] * side * -1
        bx, by = ex, ey - 32 + int(rng.integers(-3, 4))
        M = cv2.getRotationMatrix2D((bx, by), ang, 1.0)
        brow = np.zeros_like(img)
        cv2.ellipse(brow, (bx, by), (p["eye_w"] + 4, p["brow_thick"]), 0, 0, 360, (35, 35, 35), -1)
        brow = cv2.warpAffine(brow, M, (W, H))
        mask = (brow.sum(axis=2) > 0)
        img[mask] = (35, 35, 35)
    # nose (simple line for realism)
    cv2.line(img, (cx, ey + 18), (cx - 4, cy + 40), (120, 120, 120), 2)
    # mouth
    mw = p["mouth_w"] + int(rng.integers(-5, 6))
    mh = p["mouth_h"] + int(rng.integers(-2, 3))
    my = cy + 62 + int(rng.integers(-4, 5))
    cv2.ellipse(img, (cx, my), (mw // 2, mh), 0, 0, 360, (120, 60, 90), -1)
    cv2.ellipse(img, (cx, my), (mw // 2, mh), 0, 0, 360, (40, 40, 40), 2)
    # light noise burst (capture variation)
    noise = rng.integers(-6, 7, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), img)

def main(n_per_class=40):
    for cls, p in PARAMS.items():
        d = ROOT / "dataset" / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            draw_face(p, RNG, d / f"img{i+1:03d}.jpg")
            print(f"wrote {cls}/img{i+1:03d}.jpg")
    # holdout "new" photos for inference demo (not in dataset folders)
    new_dir = ROOT / "data" / "new_photos"
    new_dir.mkdir(parents=True, exist_ok=True)
    draw_face(PARAMS["person_a"], np.random.default_rng(101), new_dir / "new_person_a.jpg")
    draw_face(PARAMS["person_b"], np.random.default_rng(202), new_dir / "new_person_b.jpg")
    print("done")

if __name__ == "__main__":
    main()
