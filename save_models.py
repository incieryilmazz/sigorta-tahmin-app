#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model kaydetme scripti - Jupyter notebook'taki egitilmis modelleri kaydetmek icin
Bu scripti notebook'tan sonra calistirin veya notebook'un sonuna bu kodu ekleyin
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def save_trained_models():
    """
    Notebook'ta egitilmis modelleri kaydetmek icin bu fonksiyonu kullanin
    """
    
    print("Model kaydetme islemi baslatiliyor...")
    
    # Veri yukleme ve hazirlama (notebook'taki koddan)
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1254', 'latin1']
    df = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv("Case_Study_Data.xlsx - DATA.csv", encoding=encoding)
            print(f"Veri seti basariyla yuklendi ({encoding} encoding)")
            break
        except Exception as e:
            continue
    
    if df is None:
        print("Veri seti yuklenemedi - tum encoding'ler denendi")
        return
    
    # Veri temizleme
    df = df.replace("", np.nan)
    df.columns = [c.strip() for c in df.columns]
    
    # Orijinal kolon adlariyla calis (encoding sorunlari nedeniyle)
    print("Kolon adlari:", df.columns.tolist())
    
    # Hedef degiskeni duzenle - orijinal kolon adini kullan
    target_col = [col for col in df.columns if 'ONAY DURUMU' in col][0]
    df[target_col] = (df[target_col].astype(str).str.upper() == "P").astype(int)
    
    # Portfoy ayirimi kolonu bul
    portfoy_col = [col for col in df.columns if 'PORTF' in col and 'AYRIMI' in col][0]
    
    # Bireysel ve kurumsal ayirma
    df_bireysel = df[df[portfoy_col] != "KURUMSAL"].copy()
    df_kurumsal = df[df[portfoy_col] == "KURUMSAL"].copy()
    
    print(f"Bireysel kayit sayisi: {len(df_bireysel)}")
    print(f"Kurumsal kayit sayisi: {len(df_kurumsal)}")
    
    # BIREYSEL MODEL
    print("\n=== BIREYSEL MODEL EGITIMI ===")
    
    # Bireysel icin ozellik muhendisligi
    df_bireysel_processed = df_bireysel.copy()
    
    # Kolon adlarini dinamik olarak bul
    cols = df.columns.tolist()
    hasarsizlik_col = cols[1]  # 'HASARSIZLIK �ND�R�M� KADEMES�'
    prim_col = cols[10]        # 'TEKL�F PR�M�'
    teklif_no_col = cols[0]    # 'TEKL�F NUMARASI'
    marka_col = cols[3]        # 'MARKA'
    yakit_col = cols[6]        # 'YAKIT T�P�'
    il_col = cols[7]           # '�L'
    ilce_col = cols[8]         # '�L�E'
    yas_col = cols[9]          # 'YA�'
    
    # Hasarsizlik indirimini sayisal hale getir
    df_bireysel_processed[hasarsizlik_col] = (
        df_bireysel_processed[hasarsizlik_col]
        .astype(str).str.replace("%", "").astype(float) / 100
    )
    
    # Teklif primini duzenle
    df_bireysel_processed[prim_col] = (
        df_bireysel_processed[prim_col]
        .astype(str).str.replace(".", "").str.replace(",", ".")
        .astype(float)
    )
    
    # Log donusum
    df_bireysel_processed[prim_col] = np.log1p(df_bireysel_processed[prim_col])
    
    # Gereksiz sutunlari cikar
    columns_to_drop = [teklif_no_col, "MODEL YILI"]
    df_bireysel_processed = df_bireysel_processed.drop(columns=columns_to_drop, errors='ignore')
    
    # One-hot encoding
    categorical_cols = [marka_col, yakit_col, portfoy_col, il_col, ilce_col]
    df_bireysel_processed = pd.get_dummies(
        df_bireysel_processed,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )
    
    # X ve y ayirma
    X_bireysel = df_bireysel_processed.drop(columns=[target_col])
    y_bireysel = df_bireysel_processed[target_col]
    
    # Train-test split
    X_train_bireysel, X_test_bireysel, y_train_bireysel, y_test_bireysel = train_test_split(
        X_bireysel, y_bireysel, test_size=0.3, random_state=42, stratify=y_bireysel
    )
    
    # Bireysel icin Voting Classifier olustur (XGBoost olmadan)
    log_reg_bireysel = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    rf_bireysel = RandomForestClassifier(
        n_estimators=100, min_samples_split=2, min_samples_leaf=4,
        max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1
    )
    
    # Pipeline ile StandardScaler dahil et
    voting_bireysel = VotingClassifier(
        estimators=[
            ('lr', Pipeline([('scaler', StandardScaler()), ('classifier', log_reg_bireysel)])),
            ('rf', rf_bireysel)
        ],
        voting='soft'
    )
    
    # Modeli egit
    voting_bireysel.fit(X_train_bireysel, y_train_bireysel)
    print("Bireysel voting classifier egitildi")
    
    # KURUMSAL MODEL
    print("\n=== KURUMSAL MODEL EGITIMI ===")
    
    # Kurumsal icin ozellik muhendisligi
    df_kurumsal_processed = df_kurumsal.copy()
    
    # Hasarsizlik indirimini sayisal hale getir
    df_kurumsal_processed[hasarsizlik_col] = (
        df_kurumsal_processed[hasarsizlik_col]
        .astype(str).str.replace("%", "").astype(float) / 100
    )
    
    # Teklif primini duzenle
    df_kurumsal_processed[prim_col] = (
        df_kurumsal_processed[prim_col]
        .astype(str).str.replace(".", "").str.replace(",", ".")
        .astype(float)
    )
    
    # Log donusum
    df_kurumsal_processed[prim_col] = np.log1p(df_kurumsal_processed[prim_col])
    
    # Kurumsal icin gereksiz sutunlari cikar (YAS ve ILCE dahil)
    columns_to_drop = [teklif_no_col, "MODEL YILI", yas_col, ilce_col]
    df_kurumsal_processed = df_kurumsal_processed.drop(columns=columns_to_drop, errors='ignore')
    
    # One-hot encoding
    categorical_cols_kurumsal = [marka_col, yakit_col, portfoy_col, il_col]
    df_kurumsal_processed = pd.get_dummies(
        df_kurumsal_processed,
        columns=categorical_cols_kurumsal,
        drop_first=True,
        dtype=int
    )
    
    # X ve y ayirma
    X_kurumsal = df_kurumsal_processed.drop(columns=[target_col])
    y_kurumsal = df_kurumsal_processed[target_col]
    
    # Train-test split
    X_train_kurumsal, X_test_kurumsal, y_train_kurumsal, y_test_kurumsal = train_test_split(
        X_kurumsal, y_kurumsal, test_size=0.3, random_state=42, stratify=y_kurumsal
    )
    
    # Kurumsal icin Voting Classifier olustur (XGBoost olmadan)
    log_reg_kurumsal = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    rf_kurumsal = RandomForestClassifier(
        n_estimators=100, min_samples_split=2, min_samples_leaf=4,
        max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1
    )
    
    # Pipeline ile StandardScaler dahil et
    voting_kurumsal = VotingClassifier(
        estimators=[
            ('lr', Pipeline([('scaler', StandardScaler()), ('classifier', log_reg_kurumsal)])),
            ('rf', rf_kurumsal)
        ],
        voting='soft'
    )
    
    # Modeli egit
    voting_kurumsal.fit(X_train_kurumsal, y_train_kurumsal)
    print("Kurumsal voting classifier egitildi")
    
    # MODELLERI KAYDET
    print("\n=== MODELLERI KAYDETME ===")
    
    try:
        # Bireysel model kaydet
        joblib.dump(voting_bireysel, 'voting_bireysel.pkl')
        print("voting_bireysel.pkl kaydedildi")
        
        # Kurumsal model kaydet
        joblib.dump(voting_kurumsal, 'voting_kurumsal.pkl')
        print("voting_kurumsal.pkl kaydedildi")
        
        print("\nTum modeller basariyla kaydedildi!")
        print("Artik Streamlit uygulamanizi calistirabilisiniz.")
        
    except Exception as e:
        print(f"Model kaydetme hatasi: {e}")

if __name__ == "__main__":
    save_trained_models()
