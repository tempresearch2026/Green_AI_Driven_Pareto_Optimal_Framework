import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
import time
from datetime import datetime
import optuna  # Hızlı Bayesian optimizasyon için
from optuna.samplers import TPESampler

# Modeller
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ============================================================================
# 1. AYARLAR VE KONFİGÜRASYON
# ============================================================================
class AnalysisConfig:
    """Analiz konfigürasyon ayarları"""
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    COLOR_PALETTE = "husl"
    FIGURE_SIZE = (16, 10)
    FONT_SIZE = 12
    DPI = 300

    # Model renkleri
    MODEL_COLORS = {
        'XGBoost': '#1f77b4',
        'LightGBM': '#ff7f0e',
        'HistGradientBoosting': '#2ca02c',
        'GradientBoosting': '#d62728',
        'ElasticNet': '#9467bd',
        'Lasso': '#8c564b',
        'RandomForest': '#e377c2'
    }

    # Çıktı dizinleri
    OUTPUT_DIR = "battery_analysis_results"
    OPTIMIZATION_DIR = os.path.join(OUTPUT_DIR, "fast_hyperparameter_optimization")

    # Optimizasyon ayarları (hız için optimize edilmiş)
    OPTUNA_TRIALS = 20  # Düşürüldü: 50'den 20'ye
    CV_FOLDS = 3  # Düşürüldü: 5'ten 3'e
    RANDOM_SEARCH_ITER = 20  # RandomizedSearch için iterasyon sayısı
    EARLY_STOPPING_ROUNDS = 10


# Görselleştirme ayarlarını uygula
plt.style.use(AnalysisConfig.PLOT_STYLE)
sns.set_palette(AnalysisConfig.COLOR_PALETTE)
plt.rcParams['figure.figsize'] = AnalysisConfig.FIGURE_SIZE
plt.rcParams['font.size'] = AnalysisConfig.FONT_SIZE
plt.rcParams['savefig.dpi'] = AnalysisConfig.DPI


# ============================================================================
# 2. VERİ YÜKLEME VE HAZIRLAMA
# ============================================================================
def load_and_prepare_data(file_path, sample_frac=None):
    """Veriyi yükle ve hazırla"""
    print("=" * 80)
    print("BATARYA SOH TAHMİNİ - HIZLI BAYESYEN OPTİMİZASYONLU ML ANALİZİ")
    print("=" * 80)

    # Dosya kontrolü
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Dosya bulunamadı: {file_path}")

    # Veriyi yükle
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Veri yüklendi. Boyut: {df.shape}")
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        return None, None, None

    # Örneklem büyüklüğünü azalt (isteğe bağlı)
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"✓ Veri örneklenerek azaltıldı: {df.shape}")

    # SOH sütunu kontrolü
    if 'SOH' not in df.columns:
        soh_cols = [c for c in df.columns if 'soh' in c.lower()]
        if not soh_cols:
            raise ValueError("❌ 'SOH' sütunu bulunamadı")
        df.rename(columns={soh_cols[0]: 'SOH'}, inplace=True)
        print(f"✓ SOH sütunu bulundu ve yeniden adlandırıldı: {soh_cols[0]} -> SOH")

    # Veri kalitesi raporu
    print("\n📊 VERİ KALİTESİ RAPORU:")
    print("-" * 40)
    print(f"Toplam örnek: {len(df):,}")
    print(f"Özellik sayısı: {len(df.columns) - 1}")
    print(f"Başlangıç SOH aralığı: {df['SOH'].min():.2f}% - {df['SOH'].max():.2f}%")
    print(f"SOH ortalaması: {df['SOH'].mean():.2f}%")

    # SOH normalizasyonu (0-100%)
    if df['SOH'].max() <= 1:
        df['SOH'] = df['SOH'] * 100
        print("✓ SOH değerleri 0-100% aralığına çevrildi")

    # Outlier temizliği (IQR metodu) - hız için daha hızlı yöntem
    Q1, Q3 = df['SOH'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    if (df['SOH'] < lower_bound).any() or (df['SOH'] > upper_bound).any():
        print("✓ Outlier temizliği uygulanıyor...")
        df['SOH'] = df['SOH'].clip(lower_bound, upper_bound)

    # Eksik değer ve tip dönüşümü
    df = df.dropna(subset=['SOH'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    # Hızlı eksik değer doldurma
    df = df.fillna(df.median())

    print(f"✓ Temizlenmiş veri boyutu: {df.shape}")
    print(f"✓ Son SOH aralığı: {df['SOH'].min():.2f}% - {df['SOH'].max():.2f}%")

    # Özellik sayısını azalt
    if df.shape[1] > 50:
        # Korelasyon matrisi ile en önemli özellikleri seç
        corr_matrix = df.corr().abs()
        soh_corr = corr_matrix['SOH'].sort_values(ascending=False)

        # En yüksek korelasyonlu 30 özelliği seç
        top_features = soh_corr.index[1:31]  # SOH hariç top 30
        df = df[['SOH'] + list(top_features)]
        print(f"✓ Özellik sayısı azaltıldı: {len(top_features)} özellik seçildi")

    # Feature-target ayrımı
    X = df.drop('SOH', axis=1)
    y = df['SOH']

    print(f"✓ Özellik sayısı: {X.shape[1]}")
    print("✓ Veri hazırlığı tamamlandı")

    return X, y, df


# ============================================================================
# 3. VERİ BÖLME VE ÖN İŞLEME
# ============================================================================
def prepare_train_test_data(X, y, test_size=0.2, random_state=42):
    """Data leakage'ı önleyerek veriyi hazırla"""
    print("\n" + "=" * 80)
    print("VERİ HAZIRLAMA (DATA LEAKAGE KORUMALI)")
    print("=" * 80)

    # Basit train-test split (daha hızlı)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"✓ Train/Test bölme tamamlandı:")
    print(f"  Eğitim seti: {X_train_raw.shape[0]:,} örnek")
    print(f"  Test seti: {X_test_raw.shape[0]:,} örnek")

    # StandardScaler kullan (RobustScaler'dan daha hızlı)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # DataFrame yapısını koru
    X_train = pd.DataFrame(
        X_train_scaled,
        columns=X.columns,
        index=X_train_raw.index
    )

    X_test = pd.DataFrame(
        X_test_scaled,
        columns=X.columns,
        index=X_test_raw.index
    )

    print("✓ Data leakage engellendi (Scaler sadece train ile fit edildi)")
    print("✓ Ölçeklendirme tamamlandı")

    return X_train, X_test, y_train, y_test, scaler


# ============================================================================
# 4. METRİK HESAPLAMA FONKSİYONLARI
# ============================================================================
def calculate_comprehensive_metrics(y_true, y_pred, model_name=None):
    """Kapsamlı performans metrikleri hesapla"""

    # Temel metrikler
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Yüzde hatalar
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'n_samples': len(y_true)
    }

    return metrics


def create_performance_summary(performance_results):
    """Performans sonuçlarını özet tabloya dönüştür"""
    summary_data = []

    for model_name, metrics in performance_results.items():
        row = {
            'Model': model_name,
            'R²': f"{metrics['R2']:.6f}",
            'RMSE': f"{metrics['RMSE']:.4f}%",
            'MAE': f"{metrics['MAE']:.4f}%",
            'MAPE': f"{metrics['MAPE']:.4f}%",
            'Samples': metrics['n_samples']
        }

        if 'time' in metrics:
            row['Training_Time (s)'] = f"{metrics['time']:.2f}"
        if 'optimization_time' in metrics:
            row['Optimization_Time (s)'] = f"{metrics['optimization_time']:.2f}"

        summary_data.append(row)

    return pd.DataFrame(summary_data)


# ============================================================================
# 5. OPTUNA İLE  BAYESYEN OPTİMİZASYON
# ============================================================================
def optimize_xgboost_optuna(trial, X_train, y_train):
    """XGBoost için Optuna optimizasyonu"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'n_jobs': -1
    }

    model = XGBRegressor(**params)


    scores = cross_val_score(model, X_train, y_train,
                             cv=AnalysisConfig.CV_FOLDS,
                             scoring='r2',
                             n_jobs=1)  # Paralel işlemleri kapat

    return np.mean(scores)


def optimize_lightgbm_optuna(trial, X_train, y_train):
    """LightGBM için Optuna optimizasyonu"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'verbose': -1
    }

    model = LGBMRegressor(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=AnalysisConfig.CV_FOLDS,
                             scoring='r2')
    return np.mean(scores)


def optimize_randomforest_optuna(trial, X_train, y_train):
    """Random Forest için Optuna optimizasyonu"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=AnalysisConfig.CV_FOLDS,
                             scoring='r2',
                             n_jobs=1)
    return np.mean(scores)


# ============================================================================
# 6.  OPTİMİZASYON STRATEJİLERİ
# ============================================================================
def optimize_model_fast(model_name, X_train, y_train, method='optuna'):
    """Modeli hızlı bir şekilde optimize et"""

    print(f"  ⚡ {model_name} için hızlı optimizasyon başlatılıyor...")
    start_time = time.time()

    if method == 'optuna':
        # Optuna ile Bayesian optimizasyon
        if model_name == "XGBoost":
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(lambda trial: optimize_xgboost_optuna(trial, X_train, y_train),
                           n_trials=AnalysisConfig.OPTUNA_TRIALS,
                           show_progress_bar=False)

            best_params = study.best_params
            model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)

        elif model_name == "LightGBM":
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(lambda trial: optimize_lightgbm_optuna(trial, X_train, y_train),
                           n_trials=AnalysisConfig.OPTUNA_TRIALS,
                           show_progress_bar=False)

            best_params = study.best_params
            model = LGBMRegressor(**best_params, random_state=42, verbose=-1)

        elif model_name == "RandomForest":
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(lambda trial: optimize_randomforest_optuna(trial, X_train, y_train),
                           n_trials=AnalysisConfig.OPTUNA_TRIALS,
                           show_progress_bar=False)

            best_params = study.best_params
            model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

        else:
            # Diğer modeller için basit parametreler
            if model_name == "GradientBoosting":
                model = GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=4,
                    random_state=42
                )
            elif model_name == "HistGradientBoosting":
                model = HistGradientBoostingRegressor(
                    max_iter=100, learning_rate=0.1, max_depth=6,
                    random_state=42
                )
            elif model_name == "ElasticNet":
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            elif model_name == "Lasso":
                model = Lasso(alpha=0.1, random_state=42)
            else:
                return None, {}, 0

            best_params = {}

    elif method == 'random':
        # RandomizedSearchCV ile hızlı optimizasyon
        if model_name == "XGBoost":
            param_dist = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            base_model = XGBRegressor(random_state=42, n_jobs=-1)

        elif model_name == "LightGBM":
            param_dist = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [20, 31, 50],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = LGBMRegressor(random_state=42, verbose=-1)

        else:
            return None, {}, 0

        random_search = RandomizedSearchCV(
            base_model, param_dist,
            n_iter=AnalysisConfig.RANDOM_SEARCH_ITER,
            cv=AnalysisConfig.CV_FOLDS,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        best_params = random_search.best_params_

    optimization_time = time.time() - start_time

    return model, best_params, optimization_time


# ============================================================================
# 7. GÖRSELLEŞTİRME FONKSİYONLARI
# ============================================================================
def plot_model_comparison_fast(performance_results, best_model_name):
    """Model performans karşılaştırması görselleştirmesi"""

    models = list(performance_results.keys())
    r2_values = [performance_results[m]['R2'] for m in models]
    rmse_values = [performance_results[m]['RMSE'] for m in models]

    colors = [AnalysisConfig.MODEL_COLORS.get(m, '#666666') for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('HIZLI OPTİMİZASYONLU MODEL PERFORMANS KARŞILAŞTIRMASI\nBatarya SOH Tahmini',
                 fontsize=16, fontweight='bold', y=1.05)

    # R² Karşılaştırması
    ax1 = axes[0]
    bars1 = ax1.bar(models, r2_values, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title('R² Karşılaştırması (Yüksek = İyi)', fontweight='bold', pad=15)
    ax1.set_ylabel('R² Skoru', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([min(r2_values) - 0.1, 1.05])

    # R² değerlerini ekle
    for bar, value in zip(bars1, r2_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # RMSE Karşılaştırması
    ax2 = axes[1]
    bars2 = ax2.bar(models, rmse_values, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title('RMSE Karşılaştırması (Düşük = İyi)', fontweight='bold', pad=15)
    ax2.set_ylabel('RMSE (% SOH)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # RMSE değerlerini ekle
    for bar, value in zip(bars2, rmse_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmse_values) * 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Toplam Süre Karşılaştırması
    ax3 = axes[2]
    if all('optimization_time' in performance_results[m] for m in models):
        opt_times = [performance_results[m]['optimization_time'] for m in models]
        bars3 = ax3.bar(models, opt_times, color=colors, edgecolor='black', alpha=0.8)
        ax3.set_title('Optimizasyon Süresi', fontweight='bold', pad=15)
        ax3.set_ylabel('Süre (saniye)', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars3, opt_times):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(opt_times) * 0.01,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Süre bilgisi yok',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Optimizasyon Süresi', fontweight='bold', pad=15)

    plt.tight_layout()

    # Kaydet
    os.makedirs(AnalysisConfig.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(AnalysisConfig.OUTPUT_DIR, 'fast_optimization_comparison.png'),
                bbox_inches='tight', dpi=AnalysisConfig.DPI)
    plt.show()

    print(f"✓ Hızlı optimizasyon karşılaştırma grafiği kaydedildi")


def plot_predictions_vs_actual_fast(y_true, y_pred, model_name):
    """Tahminler vs Gerçek Değerler görselleştirmesi (hızlı)"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Tahminler vs Gerçek Değerler
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20, color='steelblue', edgecolor='black', linewidth=0.5)

    # Ideal line (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Mükemmel Tahmin')

    ax1.set_xlabel('Gerçek SOH Değerleri (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tahmin Edilen SOH (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{model_name} - Tahminler vs Gerçek Değerler',
                  fontsize=13, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # R² değerini ekle
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2. Hataların dağılımı
    ax2 = axes[1]
    errors = y_true - y_pred

    ax2.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Sıfır Hata')
    ax2.axvline(x=errors.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Ortalama: {errors.mean():.3f}')

    ax2.set_xlabel('Tahmin Hatası (Gerçek - Tahmin)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frekans', fontsize=11, fontweight='bold')
    ax2.set_title(f'{model_name} - Hata Dağılımı',
                  fontsize=13, fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Kaydet
    os.makedirs(AnalysisConfig.OPTIMIZATION_DIR, exist_ok=True)
    plt.savefig(os.path.join(AnalysisConfig.OPTIMIZATION_DIR, f'{model_name}_predictions_fast.png'),
                bbox_inches='tight', dpi=AnalysisConfig.DPI)
    plt.show()

    print(f"✓ {model_name} hızlı tahmin görselleştirmesi kaydedildi")


# ============================================================================
# 8. ANA ANALİZ FONKSİYONU
# ============================================================================
def main_fast(sample_frac=0.5):
    """Hızlı optimizasyon analiz fonksiyonu"""

    # 1. VERİ YÜKLEME (örnekleme ile)
    file_path = r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\25 degree\kaynak_birlesik_veri.csv"

    X, y, df = load_and_prepare_data(file_path, sample_frac=sample_frac)
    if X is None:
        return

    # 2. VERİ HAZIRLAMA
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(X, y)

    print(f"\n⚡ OPTİMİZASYON AYARLARI:")
    print(f"  • Örnek oranı: {sample_frac}")
    print(f"  • Optuna deneme sayısı: {AnalysisConfig.OPTUNA_TRIALS}")
    print(f"  • CV katları: {AnalysisConfig.CV_FOLDS}")
    print(f"  • Eğitim örnekleri: {len(X_train):,}")
    print(f"  • Test örnekleri: {len(X_test):,}")

    # 3. MODELLERİ OPTİMİZE ET
    print("\n" + "=" * 80)
    print("HIZLI OPTİMİZASYON BAŞLATILIYOR")
    print("=" * 80)

    performance_results = {}
    trained_models = {}

    # Hangi modelleri optimize edeceğiz
    models_to_optimize = ["XGBoost", "LightGBM", "RandomForest",
                          "GradientBoosting", "HistGradientBoosting"]

    for model_name in models_to_optimize:
        print(f"\n{'=' * 50}")
        print(f"OPTİMİZASYON: {model_name}")
        print(f"{'=' * 50}")

        # Hızlı optimizasyon uygula
        optimized_model, best_params, opt_time = optimize_model_fast(
            model_name, X_train, y_train, method='optuna'
        )

        if optimized_model is None:
            print(f"  ⚠️  {model_name} için optimizasyon atlandı")
            continue

        # Modeli eğit
        train_start = time.time()
        optimized_model.fit(X_train, y_train)
        train_time = time.time() - train_start

        # Test et
        y_pred = optimized_model.predict(X_test)

        # Metrikleri hesapla
        metrics = calculate_comprehensive_metrics(y_test, y_pred, model_name)
        metrics['time'] = train_time
        metrics['optimization_time'] = opt_time
        if best_params:
            metrics['best_params'] = best_params

        performance_results[model_name] = metrics
        trained_models[model_name] = optimized_model

        print(f"  ✓ R²: {metrics['R2']:.6f}")
        print(f"  ✓ RMSE: {metrics['RMSE']:.4f}% SOH")
        print(f"  ✓ MAE: {metrics['MAE']:.4f}% SOH")
        print(f"  ✓ Optimizasyon süresi: {opt_time:.1f} saniye")
        print(f"  ✓ Eğitim süresi: {train_time:.1f} saniye")

        # Feature importance
        if hasattr(optimized_model, 'feature_importances_'):
            importances = optimized_model.feature_importances_
            top_idx = np.argsort(importances)[-3:][::-1]
            print(f"  ✓ En önemli 3 özellik:")
            for i, idx in enumerate(top_idx):
                print(f"     {i + 1}. {X.columns[idx]}: {importances[idx]:.4f}")

    # Basit modeller (optimizasyon olmadan)
    print(f"\n{'=' * 50}")
    print(f"BASİT MODELLER")
    print(f"{'=' * 50}")

    simple_models = {
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "Lasso": Lasso(alpha=0.1, random_state=42)
    }

    for model_name, model in simple_models.items():
        print(f"\n  🔧 {model_name} eğitiliyor...")

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)

        metrics = calculate_comprehensive_metrics(y_test, y_pred, model_name)
        metrics['time'] = train_time
        metrics['optimization_time'] = 0

        performance_results[model_name] = metrics
        trained_models[model_name] = model

        print(f"    ✓ R²: {metrics['R2']:.6f}")
        print(f"    ✓ RMSE: {metrics['RMSE']:.4f}% SOH")

    # 4. EN İYİ MODELİ BELİRLE
    if performance_results:
        best_model_name = max(performance_results,
                              key=lambda m: performance_results[m]['R2'])
        best_model = trained_models[best_model_name]
        best_metrics = performance_results[best_model_name]

        print("\n" + "=" * 80)
        print("🏆 EN İYİ MODEL BELİRLENDİ")
        print("=" * 80)
        print(f"Model: {best_model_name}")
        print(f"R²: {best_metrics['R2']:.6f}")
        print(f"RMSE: {best_metrics['RMSE']:.4f}% SOH")
        print(f"MAE: {best_metrics['MAE']:.4f}% SOH")

        if 'optimization_time' in best_metrics:
            print(f"Toplam süre: {best_metrics['optimization_time'] + best_metrics['time']:.1f} saniye")

    # 5. GÖRSELLEŞTİRMELER
    print("\n" + "=" * 80)
    print("HIZLI GÖRSELLEŞTİRMELER OLUŞTURULUYOR")
    print("=" * 80)

    # Optimizasyon dizinini oluştur
    os.makedirs(AnalysisConfig.OPTIMIZATION_DIR, exist_ok=True)

    # Model karşılaştırma grafiği
    if performance_results:
        plot_model_comparison_fast(performance_results, best_model_name)

        # En iyi model için tahmin görselleştirmesi
        y_pred_best = best_model.predict(X_test)
        plot_predictions_vs_actual_fast(y_test, y_pred_best, best_model_name)

    # 6. PERFORMANS ÖZETİ OLUŞTUR
    if performance_results:
        performance_df = create_performance_summary(performance_results)

        # CSV olarak kaydet
        csv_path = os.path.join(AnalysisConfig.OUTPUT_DIR, 'fast_optimization_performance.csv')
        performance_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n✓ Performans özeti kaydedildi: {csv_path}")

        # JSON raporu
        report = {
            'analysis_date': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': list(X.columns),
                'sample_fraction': sample_frac
            },
            'best_model': {
                'name': best_model_name,
                'metrics': best_metrics
            },
            'optimization_settings': {
                'optuna_trials': AnalysisConfig.OPTUNA_TRIALS,
                'cv_folds': AnalysisConfig.CV_FOLDS,
                'method': 'Optuna TPE Sampler'
            }
        }

        json_path = os.path.join(AnalysisConfig.OUTPUT_DIR, 'fast_optimization_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ Optimizasyon raporu kaydedildi: {json_path}")

        # En iyi modeli kaydet
        best_model_path = os.path.join(AnalysisConfig.OUTPUT_DIR, 'best_fast_model.pkl')
        joblib.dump(best_model, best_model_path)
        print(f"✓ En iyi model kaydedildi: {best_model_path}")

    # 7. FİNAL RAPOR
    print("\n" + "=" * 80)
    print("HIZLI OPTİMİZASYON ANALİZİ TAMAMLANDI")
    print("=" * 80)

    print(f"\n📊 TOPLAM SÜRELER:")
    total_opt_time = sum([p.get('optimization_time', 0) for p in performance_results.values()])
    total_train_time = sum([p.get('time', 0) for p in performance_results.values()])
    print(f"  • Toplam optimizasyon süresi: {total_opt_time:.1f} saniye")
    print(f"  • Toplam eğitim süresi: {total_train_time:.1f} saniye")
    print(f"  • Toplam süre: {total_opt_time + total_train_time:.1f} saniye")

    print(f"\n✅ ANALİZ BAŞARIYLA TAMAMLANDI")


# ============================================================================
# PROGRAMIN BAŞLATILMASI
# ============================================================================
if __name__ == "__main__":
    # Çıktı dizinini oluştur
    os.makedirs(AnalysisConfig.OUTPUT_DIR, exist_ok=True)
    os.makedirs(AnalysisConfig.OPTIMIZATION_DIR, exist_ok=True)

    # Gerekli paket kontrolü
    try:
        import optuna

        print("✓ Optuna kütüphanesi yüklü")
    except ImportError:
        print("❌ Optuna kütüphanesi yüklü değil. Yüklemek için:")
        print("   pip install optuna")
        exit(1)

    # Hızlı modda çalıştır
    print("\n⚡ HIZLI OPTİMİZASYON MODU")
    print("   • Veri örneklenerek azaltılacak")
    print("   • Daha az iterasyon")
    print("   • Daha hızlı CV")
    print("\n" + "-" * 80)

    # Kullanıcıdan onay al
    response = input("Devam etmek istiyor musunuz? (E/H): ")
    if response.lower() != 'e':
        print("İşlem iptal edildi.")
        exit(0)

    # Ana fonksiyonu çalıştır
    try:
        # Daha hızlı olması için veriyi örnekle (%50)
        main_fast(sample_frac=0.5)
    except Exception as e:
        print(f"\n❌ Analiz sırasında hata oluştu: {e}")
        import traceback

        traceback.print_exc()