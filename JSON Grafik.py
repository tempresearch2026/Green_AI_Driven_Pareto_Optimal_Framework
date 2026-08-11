import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from datetime import datetime
from scipy.stats import wilcoxon, ttest_rel

# =========================
# MATPLOTLIB AYARLARI
# =========================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =========================
# 1. JSON DOSYASINI BUL
# =========================

def find_json_file():
    """JSON dosyasını otomatik olarak bul"""

    # Aranacak klasörler
    search_paths = [
        r"C:\Users\LENOVO\Desktop\--W2---Paretooptimal SOH\Ugur Hocadan Gelen Tezde Kullan",
        r"C:\Users\LENOVO\Desktop\--W2---Paretooptimal SOH",
        r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\25 degree",
        r"C:\Users\LENOVO\PycharmProjects\XG Boost",
    ]

    print("🔍 JSON dosyası aranıyor...")
    print("=" * 60)

    found_files = []

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        print(f"\n📁 Taranıyor: {search_path}")

        # Önce tam dosya isimlerini ara
        for file_name in ['all_runs_summary.json', 'stability_analysis.json', 'experiment_results.json']:
            full_path = os.path.join(search_path, file_name)
            if os.path.exists(full_path):
                found_files.append(full_path)
                print(f"   ✅ Bulundu: {full_path}")
                print(f"      Boyut: {os.path.getsize(full_path) / 1024:.2f} KB")

        # Wildcard ile ara
        for pattern in ['*summary*.json', '*stability*.json']:
            for file_path in glob.glob(os.path.join(search_path, pattern)):
                if file_path not in found_files:
                    found_files.append(file_path)
                    print(f"   ✅ Bulundu: {file_path}")
                    print(f"      Boyut: {os.path.getsize(file_path) / 1024:.2f} KB")

        # Alt klasörlerde ara
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.endswith('.json') and (
                        'summary' in file.lower() or 'stability' in file.lower() or 'experiment' in file.lower()):
                    if len(file) > 10:
                        full_path = os.path.join(root, file)
                        if full_path not in found_files:
                            found_files.append(full_path)
                            print(f"   ✅ Bulundu: {full_path}")
                            print(f"      Boyut: {os.path.getsize(full_path) / 1024:.2f} KB")

    if not found_files:
        print("\n❌ Hiç JSON dosyası bulunamadı!")
        return None

    # En büyük dosyayı seç (genellikle en kapsamlı olan)
    if len(found_files) > 1:
        print("\n" + "=" * 60)
        print("📋 BULUNAN JSON DOSYALARI:")
        for i, file in enumerate(found_files, 1):
            size = os.path.getsize(file) / 1024
            print(f"{i}. {os.path.basename(file)} - {size:.2f} KB")
            print(f"   {file}")

        # Kullanıcıya sor veya en büyüğünü seç
        sizes = [os.path.getsize(f) for f in found_files]
        largest_idx = sizes.index(max(sizes))
        selected_file = found_files[largest_idx]
        print(f"\n✅ En büyük dosya seçildi: {os.path.basename(selected_file)}")
        return selected_file

    return found_files[0]


# =========================
# 2. JSON DOSYASINI OKU
# =========================

json_path = find_json_file()

if json_path is None:
    print("\n❌ JSON dosyası bulunamadı!")
    print("\nManuel olarak belirtmek için:")
    print('json_path = r"C:\\your\\path\\to\\file.json"')
    exit()

print("\n" + "=" * 60)
print(f"📄 Kullanılacak JSON dosyası: {json_path}")
print("=" * 60)

# JSON dosyasını oku
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("✅ JSON dosyası başarıyla okundu!")

# =========================
# 3. VERİ YAPISINI ANALİZ ET
# =========================

print("\n📊 JSON yapısı analiz ediliyor...")
print(f"   Anahtar sayısı: {len(data.keys())}")
print(f"   Anahtarlar: {list(data.keys())}")

# Veriyi anlamlandır
runs = []
stability_stats = {}

if 'runs' in data:
    runs = data['runs']
    print(f"   ✅ 'runs' bulundu: {len(runs)} çalıştırma")
elif 'run_id' in data:
    runs = [data]
    print(f"   ✅ Tek çalıştırma bulundu")
else:
    # Stabilite analizi dosyası olabilir
    if 'best_rmse_per_run' in data:
        print("   ✅ Stabilite analizi dosyası")
        stability_stats = data
        # Sanal runs oluştur
        runs = []
        for i, rmse in enumerate(data['best_rmse_per_run']):
            runs.append({'run_id': i + 1, 'results': {'B0005': {'GWO': {'RMSE': rmse}}}})
    else:
        print("   ⚠️ Bilinmeyen JSON yapısı")
        runs = []

# Eğer stability_stats boşsa, data'dan al
if not stability_stats and 'stability_stats' in data:
    stability_stats = data['stability_stats']
elif not stability_stats and 'num_runs' in data:
    stability_stats = data

# =========================
# 4. VERİLERİ TOPLA
# =========================

print("\n📊 Veriler toplanıyor...")

optimizer_rmse = {}
optimizer_times = {}
best_rmse_per_run = []

# Optimizer'ları tespit et
if runs:
    first_run = runs[0]
    for battery, results in first_run['results'].items():
        for opt in results.keys():
            if opt not in optimizer_rmse:
                optimizer_rmse[opt] = []
                optimizer_times[opt] = []

# Tüm verileri topla
for run in runs:
    run_best_rmse = float('inf')

    for battery, results in run['results'].items():
        for opt, metrics in results.items():
            if opt in optimizer_rmse:
                if 'RMSE' in metrics and metrics['RMSE'] != float('inf'):
                    optimizer_rmse[opt].append(metrics['RMSE'])
                    if metrics['RMSE'] < run_best_rmse:
                        run_best_rmse = metrics['RMSE']

                if 'training_time' in metrics:
                    optimizer_times[opt].append(metrics['training_time'])
                elif 'Training Time' in metrics:
                    optimizer_times[opt].append(metrics['Training Time'])

    if run_best_rmse != float('inf'):
        best_rmse_per_run.append(run_best_rmse)

# Eğer best_rmse_per_run boşsa, stability_stats'tan al
if not best_rmse_per_run and 'best_rmse_per_run' in stability_stats:
    best_rmse_per_run = stability_stats['best_rmse_per_run']

print(f"   ✅ {len(best_rmse_per_run)} çalıştırma bulundu")
print(f"   ✅ Optimizer'lar: {list(optimizer_rmse.keys())}")

# Optimizer'ları sıralı hale getir
optimizers = ['PSO', 'WOA', 'GA', 'GWO', 'Bayesian']
optimizers = [opt for opt in optimizers if opt in optimizer_rmse]

# =========================
# 5. RENKLER VE ÇIKTI KLASÖRÜ
# =========================

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
colors_dark = ['#C0392B', '#16A085', '#2980B9', '#27AE60', '#F39C12']

output_dir = os.path.join(os.path.dirname(json_path), "stability_visualizations")
os.makedirs(output_dir, exist_ok=True)
print(f"\n📁 Görseller ve tablolar kaydedilecek: {output_dir}")

# =========================
# 6. GRAFİKLERİ OLUŞTUR
# =========================

print("\n🎨 Grafikler oluşturuluyor...")

# FIGURE 1: Boxplot - RMSE Dağılımı
fig, ax = plt.subplots(figsize=(12, 7))

data_for_box = [optimizer_rmse[opt] for opt in optimizers if optimizer_rmse[opt]]
if data_for_box:
    bp = ax.boxplot(data_for_box,
                    labels=optimizers[:len(data_for_box)],
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    meanprops={'color': 'red', 'linestyle': '--', 'linewidth': 2})

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Optimization Algorithm', fontsize=12, fontweight='bold')
    ax.set_title('RMSE Distribution Across Independent Runs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_rmse_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ 1_rmse_boxplot.png")

# FIGURE 2: Best RMSE per Run
if best_rmse_per_run:
    fig, ax = plt.subplots(figsize=(14, 7))

    run_numbers = list(range(1, len(best_rmse_per_run) + 1))
    ax.plot(run_numbers, best_rmse_per_run, 'o-', color='#2C3E50', linewidth=2,
            markersize=8, label='Best RMSE per Run', alpha=0.8)

    if len(best_rmse_per_run) > 1:
        mean_rmse = np.mean(best_rmse_per_run)
        std_rmse = np.std(best_rmse_per_run)
        ax.axhline(y=mean_rmse, color='red', linestyle='--', linewidth=2,
                   label=f'Mean RMSE = {mean_rmse:.5f}')
        ax.axhline(y=mean_rmse + std_rmse, color='orange', linestyle=':', linewidth=1.5,
                   label=f'±1 Std = {std_rmse:.5f}')
        ax.axhline(y=mean_rmse - std_rmse, color='orange', linestyle=':', linewidth=1.5)

        # Güven aralığı
        ci_lower = np.percentile(best_rmse_per_run, 2.5)
        ci_upper = np.percentile(best_rmse_per_run, 97.5)
        ax.fill_between(run_numbers, ci_lower, ci_upper, alpha=0.2, color='blue',
                        label=f'95% CI [{ci_lower:.5f}, {ci_upper:.5f}]')

    ax.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Best RMSE Values Across Independent Runs', fontsize=14, fontweight='bold')
    ax.set_xticks(run_numbers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_best_rmse_per_run.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ 2_best_rmse_per_run.png")

# FIGURE 3: Violin Plot
fig, ax = plt.subplots(figsize=(12, 7))

data_for_violin = [optimizer_rmse[opt] for opt in optimizers if optimizer_rmse[opt]]
if data_for_violin:
    parts = ax.violinplot(data_for_violin, positions=range(1, len(data_for_violin) + 1),
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')

    ax.set_xticks(range(1, len(data_for_violin) + 1))
    ax.set_xticklabels([opt for opt in optimizers if optimizer_rmse[opt]])
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Optimization Algorithm', fontsize=12, fontweight='bold')
    ax.set_title('RMSE Distribution by Optimizer', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_optimizer_violin_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ 3_optimizer_violin_plot.png")

# FIGURE 4: Training Times
if any(optimizer_times[opt] for opt in optimizers):
    fig, ax = plt.subplots(figsize=(12, 7))

    time_data = [optimizer_times[opt] for opt in optimizers if optimizer_times[opt]]
    if time_data:
        time_labels = [opt for opt in optimizers if optimizer_times[opt]]
        bp_time = ax.boxplot(time_data, labels=time_labels,
                             patch_artist=True, showmeans=True, meanline=True)

        for i, patch in enumerate(bp_time['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)

        ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Optimization Algorithm', fontsize=12, fontweight='bold')
        ax.set_title('Training Time Distribution Across Runs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # İstatistikleri ekle
        for i, opt in enumerate(time_labels, 1):
            values = optimizer_times[opt]
            if values:
                mean_val = np.mean(values)
                ax.text(i, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.3f}s',
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_training_time_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ 4_training_time_boxplot.png")

# FIGURE 5: RMSE vs Training Time
fig, ax = plt.subplots(figsize=(12, 8))

for idx, opt in enumerate(optimizers):
    if optimizer_rmse[opt] and optimizer_times[opt]:
        rmse_mean = np.mean(optimizer_rmse[opt])
        time_mean = np.mean(optimizer_times[opt])
        rmse_std = np.std(optimizer_rmse[opt])
        time_std = np.std(optimizer_times[opt])

        ax.errorbar(time_mean, rmse_mean,
                    xerr=time_std, yerr=rmse_std,
                    fmt='o', markersize=12, capsize=5,
                    label=f'{opt}', color=colors[idx % len(colors)])

ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax.set_title('RMSE vs Training Time (Mean ± Std)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_rmse_vs_time.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 5_rmse_vs_time.png")

# FIGURE 6: Heatmap
if len(runs) > 1:
    fig, ax = plt.subplots(figsize=(14, max(8, len(runs) * 0.4)))

    heatmap_data = []
    run_labels = []

    for run in runs[:min(20, len(runs))]:
        run_id = run.get('run_id', len(run_labels) + 1)
        run_labels.append(f'Run {run_id}')

        row = []
        for opt in optimizers:
            opt_rmse = []
            for battery, results in run['results'].items():
                if opt in results and 'RMSE' in results[opt]:
                    val = results[opt]['RMSE']
                    if val != float('inf'):
                        opt_rmse.append(val)
            if opt_rmse:
                row.append(np.mean(opt_rmse))
            else:
                row.append(np.nan)
        heatmap_data.append(row)

    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data, columns=optimizers, index=run_labels)

        sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn_r',
                    cbar_kws={'label': 'RMSE'}, ax=ax, linewidths=0.5,
                    annot_kws={'size': 8})

        ax.set_title('RMSE Heatmap Across Runs and Optimizers', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ 6_heatmap.png")

print("\n🎨 Grafikler tamamlandı!")

# =========================
# 7. EXCEL TABLOLARI OLUŞTUR
# =========================

print("\n📊 Excel tabloları oluşturuluyor...")


def create_excel_tables(data, runs, stability_stats, optimizer_rmse, optimizer_times, optimizers, output_dir):
    """
    SCI makalesi için tüm tabloları içeren Excel dosyası oluşturur
    """

    excel_path = os.path.join(output_dir, 'SCI_Makale_Tablolari.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # ============================================================
        # TABLO 1: 20 Çalıştırmanın En İyi RMSE Değerleri
        # ============================================================
        print("   Tablo 1 oluşturuluyor...")

        best_rmse = stability_stats.get('best_rmse_per_run', [])
        if not best_rmse and best_rmse_per_run:
            best_rmse = best_rmse_per_run

        if best_rmse:
            table1_data = {
                'Çalıştırma No': list(range(1, len(best_rmse) + 1)),
                'En İyi RMSE': best_rmse
            }
            df1 = pd.DataFrame(table1_data)

            # İstatistik satırı
            stats_row = pd.DataFrame({
                'Çalıştırma No': ['Ortalama', 'Std. Sapma', 'Min', 'Max', '95% CI (Alt)', '95% CI (Üst)'],
                'En İyi RMSE': [
                    np.mean(best_rmse) if len(best_rmse) > 0 else 0,
                    np.std(best_rmse) if len(best_rmse) > 1 else 0,
                    np.min(best_rmse) if len(best_rmse) > 0 else 0,
                    np.max(best_rmse) if len(best_rmse) > 0 else 0,
                    np.percentile(best_rmse, 2.5) if len(best_rmse) > 1 else 0,
                    np.percentile(best_rmse, 97.5) if len(best_rmse) > 1 else 0
                ]
            })
            df1 = pd.concat([df1, stats_row], ignore_index=True)
            df1.to_excel(writer, sheet_name='Tablo1_Best_RMSE', index=False)

        # ============================================================
        # TABLO 2: Optimizasyon Algoritmalarının Performans İstatistikleri
        # ============================================================
        print("   Tablo 2 oluşturuluyor...")

        optimizer_stats = stability_stats.get('optimizer_stats', {})
        if not optimizer_stats:
            # Kendi hesaplamalarımızı yap
            for opt in optimizers:
                if opt in optimizer_rmse and optimizer_rmse[opt]:
                    optimizer_stats[opt] = {
                        'mean': np.mean(optimizer_rmse[opt]),
                        'std': np.std(optimizer_rmse[opt]),
                        'min': np.min(optimizer_rmse[opt]),
                        'max': np.max(optimizer_rmse[opt])
                    }

        if optimizer_stats:
            table2_data = []
            for opt, stats in optimizer_stats.items():
                table2_data.append({
                    'Algoritma': opt,
                    'Ortalama RMSE': stats.get('mean', 0),
                    'Standart Sapma': stats.get('std', 0),
                    'Minimum RMSE': stats.get('min', 0),
                    'Maksimum RMSE': stats.get('max', 0),
                    'Çalıştırma Sayısı': len(optimizer_rmse.get(opt, []))
                })
            df2 = pd.DataFrame(table2_data)
            df2.to_excel(writer, sheet_name='Tablo2_Optimizer_Stats', index=False)

        # ============================================================
        # TABLO 3: Her Çalıştırmanın Tüm Optimizer Sonuçları
        # ============================================================
        print("   Tablo 3 oluşturuluyor...")

        if runs:
            table3_data = []
            for run in runs:
                run_id = run.get('run_id', 'N/A')
                row = {'Çalıştırma': run_id}

                for opt in optimizers:
                    opt_rmse = []
                    for battery, results in run['results'].items():
                        if opt in results and 'RMSE' in results[opt]:
                            val = results[opt]['RMSE']
                            if val != float('inf'):
                                opt_rmse.append(val)
                    if opt_rmse:
                        row[f'{opt}_RMSE'] = np.mean(opt_rmse)
                    else:
                        row[f'{opt}_RMSE'] = None

                    # Süreleri de ekle
                    opt_time = []
                    for battery, results in run['results'].items():
                        if opt in results and 'training_time' in results[opt]:
                            val = results[opt]['training_time']
                            if val > 0:
                                opt_time.append(val)
                    if opt_time:
                        row[f'{opt}_Süre(sn)'] = np.mean(opt_time)
                    else:
                        row[f'{opt}_Süre(sn)'] = None

                table3_data.append(row)

            if table3_data:
                df3 = pd.DataFrame(table3_data)
                df3.to_excel(writer, sheet_name='Tablo3_Tum_Runlar', index=False)

        # ============================================================
        # TABLO 4: Batarya Bazında Performans
        # ============================================================
        print("   Tablo 4 oluşturuluyor...")

        if runs:
            batteries = set()
            for run in runs:
                for battery in run['results'].keys():
                    batteries.add(battery)

            table4_data = []
            for battery in sorted(batteries):
                row = {'Batarya': battery}

                for opt in optimizers:
                    opt_rmse = []
                    for run in runs:
                        if battery in run['results'] and opt in run['results'][battery]:
                            val = run['results'][battery][opt].get('RMSE', float('inf'))
                            if val != float('inf'):
                                opt_rmse.append(val)
                    if opt_rmse:
                        row[f'{opt}_RMSE'] = np.mean(opt_rmse)
                        row[f'{opt}_Std'] = np.std(opt_rmse) if len(opt_rmse) > 1 else 0
                    else:
                        row[f'{opt}_RMSE'] = None
                        row[f'{opt}_Std'] = None

                table4_data.append(row)

            if table4_data:
                df4 = pd.DataFrame(table4_data)
                df4.to_excel(writer, sheet_name='Tablo4_Batarya_Bazinda', index=False)

        # ============================================================
        # TABLO 5: Stabilite Analizi Özeti
        # ============================================================
        print("   Tablo 5 oluşturuluyor...")

        # Stabilite metriklerini hesapla
        if best_rmse_per_run and len(best_rmse_per_run) > 1:
            mean_val = np.mean(best_rmse_per_run)
            std_val = np.std(best_rmse_per_run)
            ci_lower = np.percentile(best_rmse_per_run, 2.5)
            ci_upper = np.percentile(best_rmse_per_run, 97.5)
            rel_std = (std_val / mean_val) * 100

            stability_level = "Mükemmel (<%5)" if rel_std < 5 else "İyi (<%10)" if rel_std < 10 else "Orta (<%20)" if rel_std < 20 else "Zayıf (>%20)"
        else:
            mean_val = stability_stats.get('mean_best_rmse', 0)
            std_val = stability_stats.get('std_best_rmse', 0)
            ci_lower = stability_stats.get('ci_95_lower', 0)
            ci_upper = stability_stats.get('ci_95_upper', 0)
            rel_std = stability_stats.get('relative_std', 0)
            stability_level = stability_stats.get('stability', 'N/A')

        table5_data = {
            'Metrik': [
                'Çalıştırma Sayısı',
                'Ortalama En İyi RMSE',
                'Standart Sapma',
                'Minimum RMSE',
                'Maksimum RMSE',
                '95% Güven Aralığı (Alt)',
                '95% Güven Aralığı (Üst)',
                'Göreceli Standart Sapma (%)',
                'Stabilite Seviyesi'
            ],
            'Değer': [
                len(best_rmse_per_run) if best_rmse_per_run else stability_stats.get('num_runs', 0),
                mean_val,
                std_val,
                np.min(best_rmse_per_run) if best_rmse_per_run else stability_stats.get('min_best_rmse', 0),
                np.max(best_rmse_per_run) if best_rmse_per_run else stability_stats.get('max_best_rmse', 0),
                ci_lower,
                ci_upper,
                rel_std,
                stability_level
            ]
        }
        df5 = pd.DataFrame(table5_data)
        df5.to_excel(writer, sheet_name='Tablo5_Stabilite_Ozet', index=False)

        # ============================================================
        # TABLO 6: Eğitim Süreleri İstatistikleri
        # ============================================================
        print("   Tablo 6 oluşturuluyor...")

        time_stats = {}
        for opt in optimizers:
            times = optimizer_times.get(opt, [])
            if times:
                time_stats[opt] = {
                    'Ortalama (sn)': np.mean(times),
                    'Std. Sapma (sn)': np.std(times),
                    'Minimum (sn)': np.min(times),
                    'Maksimum (sn)': np.max(times)
                }

        if time_stats:
            table6_data = []
            for opt, stats in time_stats.items():
                table6_data.append({
                    'Algoritma': opt,
                    'Ortalama Eğitim Süresi (sn)': stats['Ortalama (sn)'],
                    'Standart Sapma (sn)': stats['Std. Sapma (sn)'],
                    'Minimum Süre (sn)': stats['Minimum (sn)'],
                    'Maksimum Süre (sn)': stats['Maksimum (sn)']
                })
            df6 = pd.DataFrame(table6_data)
            df6.to_excel(writer, sheet_name='Tablo6_Egitim_Sureleri', index=False)

        # ============================================================
        # TABLO 7: Wilcoxon Testi Sonuçları
        # ============================================================
        print("   Tablo 7 oluşturuluyor...")

        if runs and len(runs) > 1:
            base_rmse = []
            opt_rmse = []

            for run in runs:
                for battery, results in run['results'].items():
                    for opt, metrics in results.items():
                        if 'RMSE' in metrics and metrics['RMSE'] != float('inf'):
                            if opt == 'Bayesian':
                                base_rmse.append(metrics['RMSE'])
                            else:
                                opt_rmse.append(metrics['RMSE'])

            if base_rmse and opt_rmse and len(base_rmse) == len(opt_rmse) and len(base_rmse) > 1:
                try:
                    stat, p_value = wilcoxon(base_rmse, opt_rmse)
                    table7_data = {
                        'Test': ['Wilcoxon İşaretli Sıra Testi'],
                        'Test İstatistiği': [stat],
                        'p-değeri': [p_value],
                        'Anlamlı mı?': ['Evet' if p_value < 0.05 else 'Hayır'],
                        'Örnek Sayısı': [len(base_rmse)]
                    }
                    df7 = pd.DataFrame(table7_data)
                    df7.to_excel(writer, sheet_name='Tablo7_Wilcoxon_Test', index=False)
                except:
                    pass

        # ============================================================
        # TABLO 8: Paired T-Test Sonuçları
        # ============================================================
        print("   Tablo 8 oluşturuluyor...")

        if runs:
            batteries = set()
            for run in runs:
                for battery in run['results'].keys():
                    batteries.add(battery)

            table8_data = []
            for battery in sorted(batteries):
                base_errors = []
                opt_errors = []

                for run in runs:
                    if battery in run['results']:
                        for opt, metrics in run['results'][battery].items():
                            if 'RMSE' in metrics and metrics['RMSE'] != float('inf'):
                                if opt == 'Bayesian':
                                    base_errors.append(metrics['RMSE'])
                                else:
                                    opt_errors.append(metrics['RMSE'])

                if base_errors and opt_errors and len(base_errors) == len(opt_errors) and len(base_errors) > 1:
                    try:
                        t_stat, p_val = ttest_rel(base_errors, opt_errors)
                        diff = np.array(base_errors) - np.array(opt_errors)
                        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

                        effect_size = 'Büyük' if abs(cohens_d) > 0.8 else 'Orta' if abs(
                            cohens_d) > 0.5 else 'Küçük' if abs(cohens_d) > 0.2 else 'Çok Küçük'

                        table8_data.append({
                            'Batarya': battery,
                            't-istatistiği': t_stat,
                            'p-değeri': p_val,
                            'Anlamlı mı?': 'Evet' if p_val < 0.05 else 'Hayır',
                            "Cohen's d": cohens_d,
                            'Etki Büyüklüğü': effect_size,
                            'Örnek Sayısı': len(base_errors)
                        })
                    except:
                        pass

            if table8_data:
                df8 = pd.DataFrame(table8_data)
                df8.to_excel(writer, sheet_name='Tablo8_Paired_TTest', index=False)

        # ============================================================
        # TABLO 9: Green AI Metrikleri
        # ============================================================
        print("   Tablo 9 oluşturuluyor...")

        green_ai_data = []
        all_rmse = []
        all_times = []

        for opt in optimizers:
            all_rmse.extend(optimizer_rmse.get(opt, []))
            all_times.extend(optimizer_times.get(opt, []))

        if all_rmse and all_times:
            rmse_min, rmse_max = min(all_rmse), max(all_rmse)
            time_min, time_max = min(all_times), max(all_times)
            rmse_range = rmse_max - rmse_min if rmse_max > rmse_min else 1
            time_range = time_max - time_min if time_max > time_min else 1

            for opt in optimizers:
                rmse_vals = optimizer_rmse.get(opt, [])
                time_vals = optimizer_times.get(opt, [])

                if rmse_vals and time_vals:
                    rmse_norm = (np.mean(rmse_vals) - rmse_min) / rmse_range
                    time_norm = (np.mean(time_vals) - time_min) / time_range
                    green_score = (rmse_norm + time_norm) / 2

                    green_ai_data.append({
                        'Algoritma': opt,
                        'Ortalama RMSE': np.mean(rmse_vals),
                        'Ortalama Süre (sn)': np.mean(time_vals),
                        'RMSE Normalize': rmse_norm,
                        'Süre Normalize': time_norm,
                        'Green AI Skoru': green_score
                    })

        if green_ai_data:
            df9 = pd.DataFrame(green_ai_data)
            df9 = df9.sort_values('Green AI Skoru')
            # Sıralama ekle
            df9['Sıralama'] = range(1, len(df9) + 1)
            df9.to_excel(writer, sheet_name='Tablo9_Green_AI', index=False)

        # ============================================================
        # TABLO 10: Makale İçin Formatlanmış Tablo (LaTeX)
        # ============================================================
        print("   Tablo 10 oluşturuluyor...")

        if optimizer_stats:
            latex_lines = []
            latex_lines.append("\\begin{table}[htbp]")
            latex_lines.append("\\centering")
            latex_lines.append("\\caption{Optimizasyon Algoritmalarının Karşılaştırmalı Performansı}")
            latex_lines.append("\\label{tab:optimizer_comparison}")
            latex_lines.append("\\begin{tabular}{lcccc}")
            latex_lines.append("\\hline")
            latex_lines.append("Algoritma & Ortalama RMSE & Std. Sapma & Min RMSE & Max RMSE \\\\")
            latex_lines.append("\\hline")

            for opt in optimizers:
                if opt in optimizer_stats:
                    stats = optimizer_stats[opt]
                    latex_lines.append(
                        f"{opt} & {stats['mean']:.5f} & {stats['std']:.5f} & {stats['min']:.5f} & {stats['max']:.5f} \\\\")

            latex_lines.append("\\hline")
            latex_lines.append("\\end{tabular}")
            latex_lines.append("\\end{table}")

            latex_path = os.path.join(output_dir, 'latex_table.txt')
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(latex_lines))
            print(f"   ✅ LaTeX tablosu kaydedildi: {latex_path}")

    print(f"\n✅ Excel dosyası oluşturuldu: {excel_path}")
    return excel_path


# Excel tablolarını oluştur
excel_file = create_excel_tables(
    data, runs, stability_stats, optimizer_rmse, optimizer_times, optimizers, output_dir
)

# =========================
# 8. ÖZET RAPOR
# =========================

print("\n" + "=" * 60)
print("🎉 TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
print("=" * 60)
print(f"\n📁 Çıktı Klasörü: {output_dir}")

print("\n📊 Oluşturulan Görseller:")
print("   1. 1_rmse_boxplot.png - RMSE Boxplot")
print("   2. 2_best_rmse_per_run.png - Best RMSE per Run")
print("   3. 3_optimizer_violin_plot.png - Violin Plot")
print("   4. 4_training_time_boxplot.png - Training Times")
print("   5. 5_rmse_vs_time.png - RMSE vs Time")
print("   6. 6_heatmap.png - Heatmap")

print("\n📄 Oluşturulan Excel Tabloları (10 Tablo):")
print("   Tablo 1: 20 Çalıştırmanın En İyi RMSE Değerleri")
print("   Tablo 2: Optimizasyon Algoritmalarının Performans İstatistikleri")
print("   Tablo 3: Her Çalıştırmanın Tüm Optimizer Sonuçları")
print("   Tablo 4: Batarya Bazında Performans")
print("   Tablo 5: Stabilite Analizi Özeti")
print("   Tablo 6: Eğitim Süreleri İstatistikleri")
print("   Tablo 7: Wilcoxon Testi Sonuçları")
print("   Tablo 8: Paired T-Test Sonuçları")
print("   Tablo 9: Green AI Metrikleri")
print("   Tablo 10: LaTeX Formatında Tablo (latex_table.txt)")

print(f"\n📊 Excel dosyası: {excel_file}")
print("\n" + "=" * 60)