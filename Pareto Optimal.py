import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# VERİLER
# ============================================================================
data = {
    'Algoritma': ['GA', 'WOA', 'GWO', 'PSO', 'Bayesian'],
    'RMSE': [0.03046, 0.03051, 0.03052, 0.03050, 0.03180],
    'Egitim_Suresi': [22.32603, 20.55164, 22.87358, 22.09232, 2.88425],
    'RAM': [0.53385, 0.43750, 0.08464, 25.53906, 67.73047],
    'Pareto_Optimal': ['Evet', 'Evet', 'Evet', 'Hayir', 'Hayir'],
}

df = pd.DataFrame(data)

# ============================================================================
# RADAR/ÖRÜMCEK AĞI GRAFİĞİ (ZIT RENKLER)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Normalize edilmiş veriler
categories = ['RMSE', 'Egitim_Suresi', 'RAM']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# Pareto-optimal algoritmalar için çizim
pareto_algos = df[df['Pareto_Optimal'] == 'Evet']

# ZIT RENK PALETİ - Her algoritma için farklı renk
colors_radar = ['#e74c3c', '#3498db', '#f39c12']  # Kırmızı, Mavi, Turuncu
# Alternatif: ['#ff6b6b', '#4ecdc4', '#ffd93d']

for idx, (_, row) in enumerate(pareto_algos.iterrows()):
    values = [row['RMSE'], row['Egitim_Suresi'], row['RAM']]
    norm_values = []
    for i, cat in enumerate(categories):
        col = df[cat]
        min_val, max_val = col.min(), col.max()
        if max_val - min_val > 0:
            norm_values.append(1 - (values[i] - min_val) / (max_val - min_val))
        else:
            norm_values.append(0.5)
    norm_values += norm_values[:1]

    # Çizgi kalınlığı ve stil
    ax.plot(angles, norm_values, 'o-', linewidth=3,
            color=colors_radar[idx],
            markerfacecolor=colors_radar[idx],
            markeredgecolor='white', markeredgewidth=1.5,
            markersize=10,
            label=row['Algoritma'])
    ax.fill(angles, norm_values, alpha=0.2, color=colors_radar[idx])

# Kategori etiketleri
category_labels = ['RMSE\n(Düşük İyi)', 'Eğitim Süresi\n(Düşük İyi)', 'RAM\n(Düşük İyi)']
ax.set_xticks(angles[:-1])
ax.set_xticklabels(category_labels, fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.set_title('Pareto-Optimal Algoritmaların Performans Karşılaştırması\n(Normalize Edilmiş, 1 = En İyi)',
             fontsize=14, fontweight='bold', pad=20)

# Legend
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=11,
          framealpha=0.95, edgecolor='#bdc3c7')

# Grid iyileştirme
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('pareto_radar_chart.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

print("\nRadar grafiği kaydedildi: pareto_radar_chart.png")