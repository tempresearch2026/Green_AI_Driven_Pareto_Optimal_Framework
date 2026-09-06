"""
friedman_nemenyi_analysis.py

Bağımsız analiz kodu: Mevcut run sonuçlarını (results.json) okuyarak
Friedman testi ve Nemenyi post-hoc analizi yapar.
"""

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scipy.stats import rankdata
import scikit_posthocs as sp

# =========================
# 1. KONFIGÜRASYON
# =========================
BASE_PATH = r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\25 degree"
OUTPUT_DIR = os.path.join(BASE_PATH, "green_ai_final_runs")
NUM_RUNS = 20
EXPECTED_N_EVALS = 100
EXPECTED_EXPERIMENT_ID = "FINAL_V1_20260811"

OPTIMIZERS = ['RandomSearch', 'Bayesian', 'PSO', 'WOA', 'GA', 'GWO']
BATTERIES = ['B0005', 'B0006', 'B0018']
MODELS = ['XGBoost', 'LightGBM', 'RandomForest']


# =========================
# 2. JSON DOSYALARINI OKU
# =========================
def load_all_results(output_dir, num_runs, expected_n_evals, expected_experiment_id):
    """Tüm run'ların results.json dosyalarını oku ve birleştir."""
    all_data = []

    for run_id in range(1, num_runs + 1):
        run_dir = os.path.join(output_dir, f"run_{run_id:02d}")
        result_file = os.path.join(run_dir, 'results.json')

        if not os.path.exists(result_file):
            print(f"⚠️ Run {run_id} dosyası bulunamadı: {result_file}")
            continue

        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Kontroller
            if data.get('experiment_id') != expected_experiment_id:
                print(
                    f"⚠️ Run {run_id}: experiment_id uyuşmuyor (beklenen: {expected_experiment_id}, mevcut: {data.get('experiment_id')})")
                continue

            if data.get('n_evals') != expected_n_evals:
                print(
                    f"⚠️ Run {run_id}: n_evals uyuşmuyor (beklenen: {expected_n_evals}, mevcut: {data.get('n_evals')})")
                continue

            all_data.append(data)

        except Exception as e:
            print(f"❌ Run {run_id} okunamadı: {e}")

    print(f"✅ {len(all_data)}/{num_runs} run başarıyla yüklendi.")
    return all_data


# =========================
# 3. VERİYİ DÜZENLE
# =========================
def extract_rmse_data(all_data):
    """
    Tüm run'lardan RMSE verilerini çıkar.
    Returns: DataFrame with columns: run_id, battery, model, optimizer, test_RMSE
    """
    rows = []

    for run in all_data:
        run_id = run['run_id']

        for battery, bat_res in run['battery_results'].items():
            # Optimize edilmiş modeller
            for model_name, opt_res in bat_res['optimized'].items():
                for opt_name, metrics in opt_res.items():
                    rmse = metrics.get('test_RMSE')
                    if rmse is not None and np.isfinite(float(rmse)):
                        rows.append({
                            'run_id': run_id,
                            'battery': battery,
                            'model': model_name,
                            'optimizer': opt_name,
                            'test_RMSE': float(rmse)
                        })

    df = pd.DataFrame(rows)
    print(f"✅ {len(df)} satır veri çıkarıldı.")
    return df


# =========================
# 4. FRIEDMAN TEST + NEMENYI
# =========================
def friedman_nemenyi_analysis(df):
    """
    Friedman testi ve Nemenyi post-hoc analizi yap.
    """
    friedman_results = []
    posthoc_results = []

    # Her battery-model kombinasyonu için
    for battery in BATTERIES:
        for model in MODELS:
            print(f"\n📊 Analiz: {battery} | {model}")

            # Bu kombinasyon için veriyi filtrele
            mask = (df['battery'] == battery) & (df['model'] == model)
            sub_df = df[mask].copy()

            if len(sub_df) == 0:
                print(f"   ⚠️ Veri yok")
                continue

            # Her optimizer için veriyi hazırla
            opt_data = {}
            valid_runs = None

            for opt in OPTIMIZERS:
                opt_mask = sub_df['optimizer'] == opt
                opt_sub = sub_df[opt_mask][['run_id', 'test_RMSE']].dropna()

                if len(opt_sub) == 0:
                    continue

                # Run_id'ye göre sırala
                opt_sub = opt_sub.sort_values('run_id')

                if valid_runs is None:
                    valid_runs = set(opt_sub['run_id'].values)
                else:
                    valid_runs = valid_runs.intersection(set(opt_sub['run_id'].values))

                opt_data[opt] = opt_sub.set_index('run_id')['test_RMSE']

            # Tüm optimizer'lar için veri yoksa atla
            if len(opt_data) < 3:
                print(f"   ⚠️ Yeterli optimizer verisi yok (sadece {len(opt_data)} optimizer)")
                continue

            valid_runs = sorted(list(valid_runs))
            if len(valid_runs) < 3:
                print(f"   ⚠️ Yeterli ortak run yok (sadece {len(valid_runs)} run)")
                continue

            # Veri matrisi oluştur
            data_matrix = []
            opt_names = []
            for opt in OPTIMIZERS:
                if opt in opt_data:
                    opt_values = opt_data[opt].reindex(valid_runs).values
                    if not np.any(np.isnan(opt_values)):
                        data_matrix.append(opt_values)
                        opt_names.append(opt)

            if len(data_matrix) < 3:
                continue

            data_matrix = np.array(data_matrix)

            # Friedman test
            try:
                stat, p_value = friedmanchisquare(*data_matrix)
                print(f"   Friedman: χ²={stat:.2f}, p={p_value:.4f}")
            except Exception as e:
                print(f"   ❌ Friedman test hatası: {e}")
                continue

            # Ortalama rank'ler
            ranks_per_run = np.apply_along_axis(lambda x: rankdata(x, method='average'), 0, data_matrix)
            mean_ranks = np.mean(ranks_per_run, axis=1)

            friedman_results.append({
                'battery': battery,
                'model': model,
                'n_optimizers': len(opt_names),
                'n_runs': len(valid_runs),
                'friedman_stat': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                **{f'mean_rank_{opt}': float(rank) for opt, rank in zip(opt_names, mean_ranks)}
            })

            # Nemenyi post-hoc (sadece anlamlı ise)
            if p_value < 0.05 and len(opt_names) >= 3:
                try:
                    posthoc = sp.posthoc_nemenyi_friedman(data_matrix.T)

                    for i, opt1 in enumerate(opt_names):
                        for j, opt2 in enumerate(opt_names):
                            if i < j:
                                p_val = posthoc.iloc[i, j]
                                if p_val < 0.05:
                                    better = opt1 if mean_ranks[i] < mean_ranks[j] else opt2
                                    worse = opt2 if mean_ranks[i] < mean_ranks[j] else opt1

                                    posthoc_results.append({
                                        'battery': battery,
                                        'model': model,
                                        'optimizer_1': opt1,
                                        'optimizer_2': opt2,
                                        'p_value': float(p_val),
                                        'significant': True,
                                        'better_optimizer': better,
                                        'worse_optimizer': worse,
                                        'rank_1': float(mean_ranks[i]),
                                        'rank_2': float(mean_ranks[j])
                                    })
                                    print(f"   ✅ {better} > {worse} (p={p_val:.4f})")
                except Exception as e:
                    print(f"   ❌ Nemenyi post-hoc hatası: {e}")

    friedman_df = pd.DataFrame(friedman_results) if friedman_results else pd.DataFrame()
    posthoc_df = pd.DataFrame(posthoc_results) if posthoc_results else pd.DataFrame()

    return friedman_df, posthoc_df


# =========================
# 5. SONUÇLARI KAYDET
# =========================
def save_results(friedman_df, posthoc_df, output_dir):
    """Sonuçları CSV dosyalarına kaydet."""

    if len(friedman_df) > 0:
        friedman_df.to_csv(os.path.join(output_dir, 'friedman_test_results.csv'), index=False)
        print(f"\n✅ Friedman test sonuçları kaydedildi: {len(friedman_df)} satır")

    if len(posthoc_df) > 0:
        posthoc_df.to_csv(os.path.join(output_dir, 'nemenyi_posthoc_results.csv'), index=False)
        print(f"✅ Nemenyi post-hoc sonuçları kaydedildi: {len(posthoc_df)} satır")

    # Özet istatistikler
    if len(friedman_df) > 0:
        print("\n📊 FRIEDMAN TEST ÖZETİ:")
        for _, row in friedman_df.iterrows():
            status = "✅ SIGNIFICANT" if row['significant'] else "❌ NOT SIGNIFICANT"
            print(
                f"   {row['battery']} | {row['model']}: χ²={row['friedman_stat']:.2f}, p={row['p_value']:.4f} {status}")

    if len(posthoc_df) > 0:
        print("\n📊 NEMENYI POST-HOC (ANLAMLI KARŞILAŞTIRMALAR):")
        for _, row in posthoc_df.iterrows():
            print(
                f"   {row['battery']} | {row['model']}: {row['better_optimizer']} > {row['worse_optimizer']} (p={row['p_value']:.4f})")


# =========================
# 6. MAIN
# =========================
def main():
    print("=" * 80)
    print("FRIEDMAN TEST + NEMENYI POST-HOC ANALİZİ")
    print("=" * 80)

    # 1. JSON dosyalarını yükle
    print("\n📂 JSON dosyaları yükleniyor...")
    all_data = load_all_results(OUTPUT_DIR, NUM_RUNS, EXPECTED_N_EVALS, EXPECTED_EXPERIMENT_ID)

    if len(all_data) == 0:
        print("❌ Hiçbir run yüklenemedi. Çıkılıyor.")
        return

    # 2. RMSE verilerini çıkar
    print("\n📊 RMSE verileri çıkarılıyor...")
    df = extract_rmse_data(all_data)

    if len(df) == 0:
        print("❌ Veri çıkarılamadı. Çıkılıyor.")
        return

    # 3. Friedman + Nemenyi analizi
    print("\n📊 Friedman testi ve Nemenyi post-hoc analizi yapılıyor...")
    friedman_df, posthoc_df = friedman_nemenyi_analysis(df)

    # 4. Sonuçları kaydet
    save_results(friedman_df, posthoc_df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("✅ ANALİZ TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()