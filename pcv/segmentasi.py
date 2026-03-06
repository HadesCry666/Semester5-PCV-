import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# FUNGSI: SEGMENTASI & CROP TOMAT (TANPA KAYU)

def segment_and_crop_bgr(img_bgr, resize_dim=(300, 300)):
    """
    return: img_asli, mask_full, crop_resized
    - mask_full : siluet tomat (putih = tomat, hitam = background)
    - crop_resized : gambar 300x300, background sudah hitam polos, pantulan tetap
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ==== RANGE WARNA TOMAT (tune sesuai dataset-mu) ====
    # Merah
    low1 = np.array([0,   80, 40])
    up1  = np.array([10, 255, 255])
    # Merah ujung
    low2 = np.array([170, 80, 40])
    up2  = np.array([180, 255, 255])
    # Oranye kemerahan (sangat jenuh)
    low3 = np.array([10, 140, 40])
    up3  = np.array([25, 255, 255])

    mask1 = cv2.inRange(hsv, low1, up1)
    mask2 = cv2.inRange(hsv, low2, up2)
    mask3 = cv2.inRange(hsv, low3, up3)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    # Bersihkan noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Cari kontur
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape

    if len(cnts) == 0:
        print("⚠ Tidak ada kontur, pakai gambar asli.")
        crop = img_bgr.copy()
        mask_full = np.zeros((H, W), dtype=np.uint8)
    else:
        img_area = H * W
        min_area = 0.01 * img_area  # min 1% area gambar

        big_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]

        if len(big_cnts) == 0:
            print("⚠ Semua kontur terlalu kecil, pakai gambar asli.")
            crop = img_bgr.copy()
            mask_full = np.zeros((H, W), dtype=np.uint8)
        else:
            # === 1) GABUNG SEMUA KONTUR BESAR JADI MASK FULL ===
            mask_full = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(mask_full, big_cnts, -1, 255, thickness=-1)

            # === 1b) RAPIKAN DENGAN CONVEX HULL (agar pantulan di tepi tidak 'tergigit') ===
            cnts_full, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            hull_mask = np.zeros_like(mask_full)
            for c in cnts_full:
                hull = cv2.convexHull(c)
                cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
            mask_full = hull_mask  # pakai hull sebagai siluet final

            # === 2) BOUNDING BOX DARI MASK FULL ===
            ys, xs = np.where(mask_full > 0)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

            w = x1 - x0
            h = y1 - y0
            pad = int(0.05 * max(w, h))  # 5% padding

            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(W, x1 + pad)
            y1 = min(H, y1 + pad)

            # === 3) HAPUS BACKGROUND DI LUAR TOMAT, PAKAI MASK FULL ===
            img_clean = img_bgr.copy()
            img_clean[mask_full == 0] = (0, 0, 0)  # background jadi hitam

            # Crop di sekitar tomat
            crop = img_clean[y0:y1, x0:x1]

    crop_resized = cv2.resize(crop, resize_dim)
    return img_bgr, mask_full, crop_resized

# PROSES SEMUA FOTO & SIMPAN HASIL CROP SAJA
BASE_IN  = r"C:\laragon\www\pcv\new_dataset\dataset_training"          # <-- ganti lokasi kamu
BASE_OUT = r"C:\laragon\www\pcv\new_dataset\dataset_segmented fix"  # <-- output


CATEGORIES = ["Grade A", "Grade B", "Grade C"]

# buat folder output utama
os.makedirs(BASE_OUT, exist_ok=True)

for cat in CATEGORIES:
    in_dir  = os.path.join(BASE_IN, cat)
    out_dir = os.path.join(BASE_OUT, cat)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Proses kelas: {cat} ===")

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        in_path = os.path.join(in_dir, fname)
        img = cv2.imread(in_path)

        if img is None:
            print(f"⚠ Gagal baca gambar: {in_path}")
            continue

        # segmentasi + crop tomat
        _, _, crop300 = segment_and_crop_bgr(img, resize_dim=(300, 300))

        # Simpan hanya crop
        name_no_ext, _ = os.path.splitext(fname)
        out_crop_path = os.path.join(out_dir, f"{name_no_ext}_crop.png")

        cv2.imwrite(out_crop_path, crop300)
        print(f"✔ Simpan: {out_crop_path}")

print("\nSelesai memproses semua gambar ✅")
