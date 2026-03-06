import pandas as pd
import matplotlib.pyplot as plt

# === 1. Load CSV hasil ekstraksi fitur ===
csv_path = r"C:\laragon\www\pcv\fiturtomattoday.csv"   # ganti sesuai path
df = pd.read_csv(csv_path)

mean_rgb = df.groupby("Category")[["Mean_R", "Mean_G", "Mean_B"]].mean()
mean_glcm = df.groupby("Category")[[
    "GLCM_Contrast",
    "GLCM_Correlation",
    "GLCM_Energy",
    "GLCM_Homogeneity"
]].mean()

# === 3. Plot RGB + GLCM ===
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# --- RGB ---
mean_rgb.plot(kind='bar', ax=axes[0])
axes[0].set_title('Mean RGB per Grade (Masked / Crop256)')
axes[0].set_ylabel('Mean Intensity')
axes[0].set_xlabel('Grade')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(title='Channel')

# --- GLCM ---
mean_glcm.plot(kind='bar', ax=axes[1])
axes[1].set_title('Mean GLCM per Grade (Crop256)')
axes[1].set_ylabel('Mean GLCM Value')
axes[1].set_xlabel('Grade')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title='GLCM Feature')

plt.tight_layout()
plt.show()
