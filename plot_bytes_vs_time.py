
# plot_bytes_vs_time.py
import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/summary_by_modality_size_crypto_batch.csv")

def plot_bytes_vs_time(df, modality, savepath=None):
    sub = df[df["modality"] == modality].copy()
    x = "t_encrypt_ms_mean"
    y = "bytes_ciphertext_or_shares"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    markers = {"HE-BFV":"o", "HE-CKKS":"s", "SMPC-additive":"^"}
    colors  = {"HE-BFV":"tab:blue", "HE-CKKS":"tab:orange", "SMPC-additive":"tab:green"}

    for cr in sub["crypto"].unique():
        d = sub[sub["crypto"] == cr]
        ax.scatter(d[x], d[y], label=cr, marker=markers.get(cr,"o"),
                   color=colors.get(cr,"C0"), alpha=0.85)
        # Annotate points by size_label for clarity
        for _, row in d.iterrows():
            ax.annotate(str(row["size_label"]),
                        (row[x], row[y]),
                        textcoords="offset points", xytext=(5,5), fontsize=8, alpha=0.7)

    ax.set_title(f"Dimensione payload cifrato (bytes) (ms) — {modality}")
    ax.set_xlabel(" generazione quote (ms)")
    ax.set_ylabel("Dimensione del payload cifrato / quote (bytes)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=220)
    plt.show()

for mod in ["vector", "image", "gradient"]:
    out = f"results/figures/bytes_vs_time_{mod}.png"
    plot_bytes_vs_time(df, mod, savepath=out)
print("✅ Saved figures to results/figures/")
