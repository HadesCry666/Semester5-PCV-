import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import seaborn as sns

# Folder dataset segmented 300x300
BASE_DIR = r"C:\laragon\www\pcv\new_datasett\dataset_segmented_fix"
CATEGORIES = ["Grade A", "Grade B", "Grade C"]

# ================================
# MASK hitam → objek
# ================================
def get_mask(crop_bgr):
    # piksel bukan hitam = objek
    mask = np.any(crop_bgr > 0, axis=2).astype(np.uint8) * 255
    return mask

# ================================
# FITUR WARNA
# ================================
def extract_color(crop_bgr):
    mask = get_mask(crop_bgr)
    idx = mask > 0

    B, G, R = cv2.split(crop_bgr)
    mean_R = np.mean(R[idx])
    mean_G = np.mean(G[idx])
    mean_B = np.mean(B[idx])

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mean_H = np.mean(H[idx])
    mean_S = np.mean(S[idx])
    mean_V = np.mean(V[idx])

    return {
        "Mean_R": mean_R,
        "Mean_G": mean_G,
        "Mean_B": mean_B,
        "Mean_H": mean_H,
        "Mean_S": mean_S,
        "Mean_V": mean_V,
    }

# ================================
# FITUR GLCM (langsung dari crop 300x300)
# ================================
def extract_glcm(crop_bgr):
    mask = get_mask(crop_bgr)
    if mask.sum() == 0:
        mask[:] = 255

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Hapus background hitam
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    gray_roi = gray[y0:y1+1, x0:x1+1]

    # Normalisasi dan tajamkan
    gray_roi = cv2.normalize(gray_roi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_roi = clahe.apply(gray_roi)

    # Laplacian untuk tekstur
    edges = cv2.Laplacian(gray_roi, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))

    # Quantize 8 level
    gray_q = (edges / 32).astype(np.uint8)

    glcm = graycomatrix(
        gray_q,
        distances=[1, 2, 4],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=8,
        symmetric=True,
        normed=True,
    )

    return {
        "GLCM_Contrast": graycoprops(glcm, "contrast").mean(),
        "GLCM_Correlation": graycoprops(glcm, "correlation").mean(),
        "GLCM_Energy": graycoprops(glcm, "energy").mean(),
        "GLCM_Homogeneity": graycoprops(glcm, "homogeneity").mean(),
    }

# ================================
# Ekstrak fitur 1 gambar
# ================================
def extract_features(img_path):
    crop = cv2.imread(img_path)
    if crop is None:
        raise FileNotFoundError(img_path)

    color_feat = extract_color(crop)
    glcm_feat  = extract_glcm(crop)

    return {**color_feat, **glcm_feat}

# ================================
# Loop semua file → CSV
# ================================
def extract_last_number(fname):
    nums = re.findall(r"(\d+)", fname)
    return int(nums[-1]) if nums else float("inf")

rows = []

for cat in CATEGORIES:
    folder = os.path.join(BASE_DIR, cat)
    if not os.path.isdir(folder):
        print("Folder tidak ditemukan:", folder)
        continue

    files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png",".jpeg"))],
        key=extract_last_number
    )

    print(f"Kategori {cat} | Jumlah gambar: {len(files)}")

    for fname in tqdm(files, desc=f"Ekstrak {cat}", unit="file"):
        fpath = os.path.join(folder, fname)

        try:
            feats = extract_features(fpath)
        except Exception as e:
            print("Error:", fpath, e)
            continue

        feats["Category"] = cat
        feats["Filename"] = fname
        rows.append(feats)

df = pd.DataFrame(rows)

# Simpan CSV
csv_out = r"C:\laragon\www\pcv\fiturtomattoday.csv"
df.to_csv(csv_out, index=False)

print("Selesai! CSV disimpan:", csv_out)
print(df.head())

# Visualisasi sampel
sns.scatterplot(data=df, x="GLCM_Contrast", y="GLCM_Homogeneity", hue="Category")
plt.show()
