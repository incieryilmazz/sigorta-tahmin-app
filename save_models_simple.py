#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basit model kaydetme scripti
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def save_models():
    print("Model kaydetme basladi...")
    
    # Veri yukle
    try:
        df = pd.read_csv("Case_Study_Data.xlsx - DATA.csv", encoding='utf-8')
        print("Veri yuklendi")
    except:
        print("Veri yuklenemedi")
        return
    
    # Veri temizle
    df = df.replace("", np.nan)
    df.columns = [c.strip() for c in df.columns]
    
    # Kolon indekslerini kullan (encoding sorunlari icin)
    cols = df.columns.tolist()
    print("Kolonlar:", cols)
    
    # Hedef degiskeni duzenle (P=1, T=0)
    target_col = cols[13]  # Son kolon
    df[target_col] = (df[target_col].astype(str).str.upper() == "P").astype(int)
    
    # Sadece bireysel verilerle calis (kurumsal cok az)
    portfoy_col = cols[11]
    df_work = df[df[portfoy_col] != "KURUMSAL"].copy()
    print(f"Calisma veri sayisi: {len(df_work)}")
    
    # Basit ozellik secimi
    # Sayisal kolonlar: indeks 1 (hasarsizlik), 2 (trafik), 4 (arac yasi), 10 (prim)
    # Kategorik: 3 (marka), 6 (yakit), 7 (il), 11 (portfoy)
    
    # Hasarsizlik indirimini duzenle
    hasarsizlik_col = cols[1]
    df_work[hasarsizlik_col] = (
        df_work[hasarsizlik_col].astype(str)
        .str.replace("%", "").str.replace("nan", "0")
        .astype(float) / 100
    )
    
    # Prim kolonu duzenle
    prim_col = cols[10]
    df_work[prim_col] = (
        df_work[prim_col].astype(str)
        .str.replace(".", "").str.replace(",", ".")
        .str.replace("nan", "1000")
        .astype(float)
    )
    df_work[prim_col] = np.log1p(df_work[prim_col])
    
    # Basit ozellik matrisi olustur
    features = []
    feature_names = []
    
    # Sayisal ozellikler
    features.append(df_work[hasarsizlik_col].fillna(0).values)
    feature_names.append("hasarsizlik")
    
    features.append(df_work[cols[2]].fillna(40).values)  # trafik kodu
    feature_names.append("trafik")
    
    features.append(df_work[cols[4]].fillna(5).values)   # arac yasi
    feature_names.append("arac_yasi")
    
    features.append(df_work[prim_col].fillna(8).values)  # log prim
    feature_names.append("prim")
    
    # Kategorik ozellikler - label encoding
    le_marka = LabelEncoder()
    marka_encoded = le_marka.fit_transform(df_work[cols[3]].fillna("DIGER").astype(str))
    features.append(marka_encoded)
    feature_names.append("marka")
    
    le_yakit = LabelEncoder()
    yakit_encoded = le_yakit.fit_transform(df_work[cols[6]].fillna("BENZIN").astype(str))
    features.append(yakit_encoded)
    feature_names.append("yakit")
    
    le_il = LabelEncoder()
    il_encoded = le_il.fit_transform(df_work[cols[7]].fillna("ISTANBUL").astype(str))
    features.append(il_encoded)
    feature_names.append("il")
    
    # Ozellik matrisini olustur
    X = np.column_stack(features)
    y = df_work[target_col].values
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Pozitif ornekler: {y.sum()}, Negatif ornekler: {len(y) - y.sum()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Model olustur
    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, class_weight="balanced", 
        random_state=42, n_jobs=-1
    )
    
    # Voting classifier
    voting_model = VotingClassifier(
        estimators=[
            ('lr', Pipeline([('scaler', StandardScaler()), ('classifier', log_reg)])),
            ('rf', rf)
        ],
        voting='soft'
    )
    
    # Modeli egit
    print("Model egitiliyor...")
    voting_model.fit(X_train, y_train)
    print("Model egitildi")
    
    # Test et
    train_score = voting_model.score(X_train, y_train)
    test_score = voting_model.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Modelleri kaydet
    print("Modeller kaydediliyor...")
    
    # Bireysel model (ana model)
    joblib.dump(voting_model, 'voting_bireysel.pkl')
    print("voting_bireysel.pkl kaydedildi")
    
    # Kurumsal model (ayni model - veri az oldugu icin)
    joblib.dump(voting_model, 'voting_kurumsal.pkl')
    print("voting_kurumsal.pkl kaydedildi")
    
    # Encoder'lari da kaydet (ihtiyac olabilir)
    encoders = {
        'marka': le_marka,
        'yakit': le_yakit,
        'il': le_il,
        'feature_names': feature_names,
        'columns': cols
    }
    joblib.dump(encoders, 'encoders.pkl')
    print("encoders.pkl kaydedildi")
    
    print("\nTum modeller basariyla kaydedildi!")
    print("Streamlit uygulamasini calistirabilisiniz.")

if __name__ == "__main__":
    save_models()
