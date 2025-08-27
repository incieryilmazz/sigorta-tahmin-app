# app2.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# -------------------------------------------------------------
# Sayfa ayarları
# -------------------------------------------------------------
st.set_page_config(page_title="Sigorta Teklif Onay Sistemi", layout="wide")
st.title("Sigorta Teklif Onay Tahmin Sistemi")

DATA_PATH = Path("Case_Study_Data.xlsx - DATA.csv")
MODEL_BIREYSEL_PATH = Path("voting_bireysel.pkl")
MODEL_KURUMSAL_PATH = Path("voting_kurumsal.pkl")

# -------------------------------------------------------------
# Yükleyiciler (cache'li)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.exists():
        st.error(f"Veri seti bulunamadı: {DATA_PATH}")
        st.stop()
    # Türkçe karakterler için farklı encoding dene
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1254", "latin1"):
        try:
            df = pd.read_csv(DATA_PATH, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        st.error(f"Veri seti okunamadı: {last_err}")
        st.stop()
    # Boş stringleri NaN yap ve kolon adlarındaki boşlukları temizle
    df = df.replace("", np.nan)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_resource(show_spinner=False)
def load_models():
    if not MODEL_BIREYSEL_PATH.exists() or not MODEL_KURUMSAL_PATH.exists():
        st.error(
            "Model dosyaları bulunamadı. Lütfen notebook içinde eğitimli pipeline'ları kaydedin: \n"
            "joblib.dump(voting_bireysel, 'voting_bireysel.pkl') ve joblib.dump(voting_kurumsal, 'voting_kurumsal.pkl')"
        )
        st.stop()
    try:
        model_bireysel = joblib.load(MODEL_BIREYSEL_PATH)
        model_kurumsal = joblib.load(MODEL_KURUMSAL_PATH)
        return model_bireysel, model_kurumsal
    except Exception as e:
        st.error(f"Model dosyaları yüklenemedi: {e}")
        st.info("Model pipeline'ları notebook'ta joblib ile kaydedilmiş olmalı ve gerekli paketler bu ortamda kurulu olmalı (örn. xgboost).")
        st.stop()

# -------------------------------------------------------------
# Yardımcılar
# -------------------------------------------------------------
def parse_percent_to_ratio(pct_str: str) -> float:
    # "30%" -> 0.3
    s = str(pct_str).strip().replace("%", "")
    if s == "" or pd.isna(s):
        return 0.0
    return float(s) / 100.0

@st.cache_data(show_spinner=False)
def get_ui_options(df: pd.DataFrame):
    # Kategorik alanlar için dataset'teki TAM küme (fazla/eksik yok)
    marka_list = sorted(df["MARKA"].dropna().astype(str).str.strip().unique().tolist())
    yakit_list = sorted(df["YAKIT TİPİ"].dropna().astype(str).str.strip().unique().tolist())
    il_list = sorted(df["İL"].dropna().astype(str).str.strip().unique().tolist())
    portfoy_list = sorted(df["PORTFÖY AYRIMI"].dropna().astype(str).str.strip().unique().tolist())
    hasarsizlik_list = (
        df["HASARSIZLIK İNDİRİMİ KADEMESİ"].dropna().astype(str).str.strip().unique().tolist()
    )
    # Yüzdeleri sayısal sırada göstermek için sırala
    try:
        hasarsizlik_list = sorted(
            hasarsizlik_list, key=lambda x: float(str(x).replace("%", "").strip())
        )
    except Exception:
        hasarsizlik_list = sorted(hasarsizlik_list)

    # Trafik basamak kodu seçenekleri
    trafik_list = (
        df["TRAFİK BASAMAK KODU"].dropna().astype(int).unique().tolist()
    )
    trafik_list = sorted(trafik_list)

    # İl -> İlçe map'i (boş ilçe hariç)
    ilce_map = (
        df[["İL", "İLÇE"]]
        .dropna()
        .astype(str)
        .apply(lambda s: s.str.strip())
        .drop_duplicates()
        .groupby("İL")["İLÇE"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    # Sayısal alanlar için mantıklı sınırlar (dataset'ten min/max)
    yas_min, yas_max = int(df["YAŞ"].min()), int(df["YAŞ"].max()) if "YAŞ" in df.columns else (18, 100)
    arac_yasi_min, arac_yasi_max = int(df["ARAÇ YAŞI"].min()), int(df["ARAÇ YAŞI"].max())
    model_yili_min, model_yili_max = int(df["MODEL YILI"].min()), int(df["MODEL YILI"].max())
    # Teklif primi input'u kullanıcıdan TL olarak alınacak; sınırlar dataset'ten
    # Dataset'teki değerler 18.743 gibi (binlik noktalı) olabilir, bu yüzden sadece aralık için kaba tahmin kullanalım
    prim_values = (
        df["TEKLİF PRİMİ"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".")
    )
    prim_values = pd.to_numeric(prim_values, errors="coerce")
    prim_min = int(np.nanmin(prim_values)) if np.isfinite(np.nanmin(prim_values)) else 0
    prim_max = int(np.nanmax(prim_values)) if np.isfinite(np.nanmax(prim_values)) else 100000

    return {
        "MARKA": marka_list,
        "YAKIT TİPİ": yakit_list,
        "İL": il_list,
        "PORTFÖY AYRIMI": portfoy_list,
        "HASARSIZLIK İNDİRİMİ KADEMESİ": hasarsizlik_list,
        "TRAFİK BASAMAK KODU": trafik_list,
        "ILCE_MAP": ilce_map,
        "YAŞ_RANGE": (yas_min, yas_max),
        "ARAÇ_YAŞI_RANGE": (arac_yasi_min, arac_yasi_max),
        "MODEL_YILI_RANGE": (model_yili_min, model_yili_max),
        "PRIM_RANGE": (prim_min, prim_max),
    }


# -------------------------------------------------------------
# UI: Sigortalı Tipi ve Form
# -------------------------------------------------------------
st.sidebar.header("Sigortalı Bilgileri")
sigortali_tipi = st.sidebar.radio("Sigortalı Tipi Seçiniz", ("Bireysel", "Kurumsal"))

# Veri ve artefaktlar
df = load_dataset()
ui = get_ui_options(df)
model_bireysel, model_kurumsal = load_models()

# Form
st.header("Teklif Bilgileri")

# İl seçimi form dışında - dinamik ilçe güncellemesi için
il = st.selectbox("İl", ui["İL"], key="il_secimi")

with st.form("teklif_form"):
    if sigortali_tipi == "Bireysel":
        c1, c2, c3 = st.columns(3)
    else:
        c1, c2 = st.columns(2)  # Kurumsal için 2 kolon yeterli
    
    with c1:
        hasarsizlik = st.selectbox(
            "Hasarsızlık İndirim Kademesi", ui["HASARSIZLIK İNDİRİMİ KADEMESİ"]
        )
        trafik_kodu = st.selectbox(
            "Trafik Basamak Kodu", ui["TRAFİK BASAMAK KODU"]
        )
        marka = st.selectbox("Marka", ui["MARKA"])
        arac_yasi = st.number_input(
            "Araç Yaşı",
            min_value=int(ui["ARAÇ_YAŞI_RANGE"][0]),
            max_value=int(ui["ARAÇ_YAŞI_RANGE"][1]),
            value=int(ui["ARAÇ_YAŞI_RANGE"][0]),
        )
    
    with c2:
        yakit = st.selectbox("Yakıt Tipi", ui["YAKIT TİPİ"])
        
        # Portföy Ayrımı - hem bireysel hem kurumsal için
        if sigortali_tipi == "Bireysel":
            portfoy = st.selectbox("Portföy Ayrımı", ui["PORTFÖY AYRIMI"])
        else:
            # Kurumsal için tüm portföy seçeneklerini göster
            portfoy = st.selectbox("Portföy Ayrımı", ui["PORTFÖY AYRIMI"])
        
        # İlçe seçimi - sadece bireysel için ve seçilen ile göre
        if sigortali_tipi == "Bireysel":
            ilce_listesi = ui["ILCE_MAP"].get(il, [])
            if len(ilce_listesi) > 0:
                ilce = st.selectbox("İlçe", ilce_listesi, key="ilce_secimi")
            else:
                ilce = st.selectbox("İlçe", [""], key="ilce_secimi")
        else:
            ilce = ""  # Kurumsal için ilçe yok
            
        # Kurumsal için teklif primi burada
        if sigortali_tipi == "Kurumsal":
            prim = st.number_input(
                "Teklif Primi (TL)",
                min_value=int(ui["PRIM_RANGE"][0]),
                max_value=int(max(ui["PRIM_RANGE"][1], 1_000_000)),
                value=int(min(max(ui["PRIM_RANGE"][0], 1000), 100000)),
                step=100,
            )
    
    # Üçüncü kolon sadece bireysel için
    if sigortali_tipi == "Bireysel":
        with c3:
            yas = st.number_input(
                "Yaş",
                min_value=18,  # Minimum yaş 18
                max_value=int(ui["YAŞ_RANGE"][1]) if isinstance(ui["YAŞ_RANGE"], tuple) else 100,
                value=18,  # Başlangıç değeri 18
            )
            # Bireysel için teklif primi yaş altında
            prim = st.number_input(
                "Teklif Primi (TL)",
                min_value=int(ui["PRIM_RANGE"][0]),
                max_value=int(max(ui["PRIM_RANGE"][1], 1_000_000)),
                value=int(min(max(ui["PRIM_RANGE"][0], 1000), 100000)),
                step=100,
            )
    else:
        # Kurumsal için yaş yok
        yas = 0  # Kurumsal için yaş bilgisi yok
    
    # Optimal prim önerisi butonu
    optimize_button = st.form_submit_button("🎯 Optimal Prim Öner")
    
    submitted = st.form_submit_button("Teklif Onay Durumunu Tahmin Et")


# Optimal prim önerisi fonksiyonu
def find_optimal_premium(hasarsizlik_numeric, trafik_kodu, arac_yasi, marka_encoded, yakit_encoded, il_encoded, model):
    """
    En yüksek onay olasılığını veren prim miktarını bulur
    
    HESAPLAMA YÖNTEMİ:
    1. 500 TL - 50,000 TL arası 500'er TL artışlarla test eder (100 farklı değer)
    2. Her prim değeri için:
       - Log dönüşümü uygular: log(1 + prim)
       - Özellik vektörü oluşturur
       - Model ile onay olasılığını hesaplar
    3. En yüksek olasılık veren prim değerini döndürür
    
    KULLANILAN DEĞİŞKENLER:
    - Hasarsızlık indirimi (0-1 arası normalize)
    - Trafik basamak kodu (sayısal)
    - Araç yaşı (sayısal)
    - Prim (log dönüşümlü)
    - Marka (hash encoding)
    - Yakıt tipi (hash encoding)
    - İl (hash encoding)
    """
    best_prim = 1000
    best_proba = 0
    tested_values = []
    
    # Farklı prim değerlerini test et (500-50000 TL arası, 500'er artış)
    for test_prim in range(500, 50000, 500):
        prim_log = np.log1p(float(test_prim))
        features = np.array([
            hasarsizlik_numeric,
            float(trafik_kodu),
            float(arac_yasi),
            prim_log,
            marka_encoded,
            yakit_encoded,
            il_encoded
        ]).reshape(1, -1)
        
        try:
            proba = model.predict_proba(features)[0][1]
            tested_values.append((test_prim, proba))
            if proba > best_proba:
                best_proba = proba
                best_prim = test_prim
        except:
            continue
    
    return best_prim, best_proba, tested_values

if optimize_button:
    try:
        # Özellik hazırlama
        hasarsizlik_numeric = float(hasarsizlik.replace("%", "")) / 100 if "%" in hasarsizlik else 0.3
        
        # İlk defa sigortalı kontrolü - hasarsızlık ve trafik kodu override
        if "İLK DEFA SİGORTALI" in portfoy.upper():
            hasarsizlik_numeric = 0.0  # İlk defa sigortalı için hasarsızlık 0
            trafik_kodu_final = 4  # İlk defa sigortalı için trafik kodu 4
        else:
            trafik_kodu_final = trafik_kodu
        
        marka_encoded = hash(marka) % 100
        yakit_encoded = hash(yakit) % 10
        il_encoded = hash(il) % 50
        prim_log = np.log1p(float(prim))
        
        model = model_bireysel if sigortali_tipi == "Bireysel" else model_kurumsal
        
        # Optimal prim bul
        optimal_prim, optimal_proba, tested_values = find_optimal_premium(
            hasarsizlik_numeric, trafik_kodu_final, arac_yasi, 
            marka_encoded, yakit_encoded, il_encoded, model
        )
        
        st.subheader(" Optimal Prim Önerisi")
        st.success(f"**Önerilen Prim: {optimal_prim:,} TL**")
        st.info(f"Bu prim ile onay olasılığı: **%{optimal_proba*100:.1f}**")
        
        # Hesaplama detayları
        with st.expander(" Hesaplama Detayları"):
            st.write("**Kullanılan Değişkenler:**")
            if "İLK DEFA SİGORTALI" in portfoy.upper():
                st.write(f"• Hasarsızlık İndirimi: {hasarsizlik_numeric:.2f} ⚠️ (İlk defa sigortalı için 0'a ayarlandı)")
                st.write(f"• Trafik Basamak Kodu: {trafik_kodu_final} ⚠️ (İlk defa sigortalı için 4'e ayarlandı)")
            else:
                st.write(f"• Hasarsızlık İndirimi: {hasarsizlik_numeric:.2f}")
                st.write(f"• Trafik Basamak Kodu: {trafik_kodu_final}")
            st.write(f"• Araç Yaşı: {arac_yasi}")
            st.write(f"• Marka (encoded): {marka_encoded}")
            st.write(f"• Yakıt Tipi (encoded): {yakit_encoded}")
            st.write(f"• İl (encoded): {il_encoded}")
            
            st.write("\n**Hesaplama Yöntemi:**")
            st.write("1. 500-50,000 TL arası 500'er TL artışla test")
            st.write("2. Her prim için log(1+prim) dönüşümü")
            st.write("3. Machine Learning modeli ile olasılık hesabı")
            st.write(f"4. Toplam {len(tested_values)} farklı değer test edildi")
            
            # En iyi 5 prim değerini göster
            if len(tested_values) > 0:
                sorted_values = sorted(tested_values, key=lambda x: x[1], reverse=True)[:5]
                st.write("\n**En İyi 5 Prim Değeri:**")
                for i, (prim, prob) in enumerate(sorted_values, 1):
                    st.write(f"{i}. {prim:,} TL → %{prob*100:.1f} onay şansı")
        
        # Mevcut prim ile karşılaştırma
        current_prim_log = np.log1p(float(prim))
        current_features = np.array([
            hasarsizlik_numeric, float(trafik_kodu), float(arac_yasi),
            current_prim_log, marka_encoded, yakit_encoded, il_encoded
        ]).reshape(1, -1)
        
        try:
            current_proba = model.predict_proba(current_features)[0][1]
            st.write(f"Mevcut prim ({prim:,} TL) ile onay olasılığı: **%{current_proba*100:.1f}**")
            
            if optimal_proba > current_proba:
                improvement = (optimal_proba - current_proba) * 100
                st.success(f" Optimal prim ile **%{improvement:.1f}** daha yüksek onay şansı!")
            else:
                st.info("Mevcut priminiz zaten optimal seviyede.")
        except:
            pass
            
    except Exception as e:
        st.error(f"Optimal prim hesaplanırken hata: {str(e)}")

if submitted:
    try:
        # Basit model için özellik vektörü hazırla
        hasarsizlik_numeric = float(hasarsizlik.replace("%", "")) / 100 if "%" in hasarsizlik else 0.3
        
        # İlk defa sigortalı kontrolü - hasarsızlık ve trafik kodu override
        if "İLK DEFA SİGORTALI" in portfoy.upper():
            hasarsizlik_numeric = 0.0  # İlk defa sigortalı için hasarsızlık 0
            trafik_kodu_final = 4  # İlk defa sigortalı için trafik kodu 4
        else:
            trafik_kodu_final = trafik_kodu
        
        prim_log = np.log1p(float(prim))
        
        # Kategorik değişkenler için basit encoding
        marka_encoded = hash(marka) % 100
        yakit_encoded = hash(yakit) % 10
        il_encoded = hash(il) % 50
        
        # Özellik vektörü oluştur
        if sigortali_tipi == "Bireysel":
            features = np.array([
                hasarsizlik_numeric,
                float(trafik_kodu_final),
                float(arac_yasi),
                prim_log,
                marka_encoded,
                yakit_encoded,
                il_encoded
            ]).reshape(1, -1)
        else:
            # Kurumsal için yaş ve ilçe yok
            features = np.array([
                hasarsizlik_numeric,
                float(trafik_kodu_final),
                float(arac_yasi),
                prim_log,
                marka_encoded,
                yakit_encoded,
                il_encoded
            ]).reshape(1, -1)
        
        # Model seçimi
        model = model_bireysel if sigortali_tipi == "Bireysel" else model_kurumsal
        
        # Tahmin yap
        pred = model.predict(features)
        proba = None
        try:
            proba = model.predict_proba(features)[0][1]
        except Exception:
            proba = None

        st.subheader("Tahmin Sonucu")
        if int(pred[0]) == 1:
            if proba is not None:
                st.success(f"✅ Teklif Onaylandı (Olasılık: %{proba*100:.2f})")
            else:
                st.success("✅ Teklif Onaylandı")
        else:
            if proba is not None:
                st.error(f"❌ Teklif Reddedildi (Red Olasılığı: %{(1-proba)*100:.2f})")
            else:
                st.error("❌ Teklif Reddedildi")
                
        # Debug bilgisi
        with st.expander("📋 Tahmin Detayları"):
            st.write(f"**Sigortalı Tipi:** {sigortali_tipi}")
            st.write(f"**Portföy Ayrımı:** {portfoy}")
            st.write("\n**Model Girdi Değişkenleri:**")
            if "İLK DEFA SİGORTALI" in portfoy.upper():
                st.write(f"• Hasarsızlık İndirimi: {hasarsizlik_numeric:.2f} ⚠️ (İlk defa sigortalı için 0'a ayarlandı)")
                st.write(f"• Trafik Basamak Kodu: {trafik_kodu_final} ⚠️ (İlk defa sigortalı için 4'e ayarlandı)")
            else:
                st.write(f"• Hasarsızlık İndirimi: {hasarsizlik_numeric:.2f}")
                st.write(f"• Trafik Basamak Kodu: {trafik_kodu_final}")
            st.write(f"• Araç Yaşı: {arac_yasi}")
            st.write(f"• Teklif Primi (log): {prim_log:.2f} (Orijinal: {prim:,} TL)")
            st.write(f"• Marka (encoded): {marka_encoded} ({marka})")
            st.write(f"• Yakıt Tipi (encoded): {yakit_encoded} ({yakit})")
            st.write(f"• İl (encoded): {il_encoded} ({il})")
            if sigortali_tipi == "Bireysel":
                st.write(f"• Yaş: {yas}")
                st.write(f"• İlçe: {ilce}")
            
            st.write("\n**Model Bilgisi:**")
            st.write(f"• Kullanılan Model: {'Bireysel' if sigortali_tipi == 'Bireysel' else 'Kurumsal'} Voting Classifier")
            st.write("• Alt Modeller: Logistic Regression + Random Forest")
            st.write("• Encoding: Hash-based kategorik değişken dönüşümü")

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.info("Lütfen tüm alanların doğru doldurulduğundan emin olun.")

# Sidebar yardım
st.sidebar.info(
    """
    1) Sol menüden sigortalı tipini seçin (Bireysel/Kurumsal)
    2) Tüm alanları doldurun (seçenekler dataset'ten otomatik alınır)
    3) "Teklif Onay Durumunu Tahmin Et" butonuna tıklayın
    """
)