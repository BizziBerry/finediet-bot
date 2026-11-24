"""
Обучает модель классификации товаров из чеков и сохраняет её в model.joblib.

    python train_model.py --data receipt_items.csv        (или df_cleaned_filtered.xlsx)

Данные:  https://www.kaggle.com/datasets/bizziberri/purchase-data
Модель:  TF-IDF по словам (1-2) и символам (2-5) + LinearSVC.
         На отложенных 20% в ноутбуке на Kaggle: accuracy 0.912, macro F1 0.688 (MLP из проекта: 0.853 / 0.561).
Вероятности: softmax(оценки SVM / T), температура T подбирается на отложенной части.
Размер: мелкие веса обнуляются и хранятся разреженно - модель в разы меньше, точность проверяется.
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from model import ReceiptClassifier, clean


def make_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=60_000, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2,
                                 max_features=150_000, dtype=np.float32)),
    ])


def fit(texts, y, C):
    vec = make_vectorizer()
    X = vec.fit_transform(texts)
    svm = LinearSVC(C=C).fit(X, y)
    return vec, svm


def prune(coef, thr):
    c = coef.copy()
    c[np.abs(c) < thr] = 0
    return sparse.csr_matrix(c, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="receipt_items.csv")
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--prune", type=float, default=0.02, help="веса меньше этого по модулю обнуляются")
    args = ap.parse_args()

    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_excel(args.data)
    # to_numpy(dtype=object): в pandas 3 с pyarrow .values возвращает не numpy-массив
    X = df["Название"].map(clean).to_numpy(dtype=object)
    y = df["Категория"].astype(str).to_numpy(dtype=object)
    t0 = time.time()

    # 1) проверка на отложенных 20% + подбор температуры и порога обнуления
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vec, svm = fit(X_tr, y_tr, args.C)
    Z = vec.transform(X_te) @ svm.coef_.T + svm.intercept_
    p = svm.classes_[Z.argmax(1)]
    print(f"отложенные 20%: accuracy {accuracy_score(y_te, p):.3f}, "
          f"macro F1 {f1_score(y_te, p, average='macro', zero_division=0):.3f}  [{(time.time() - t0) / 60:.1f} мин]")

    idx = np.searchsorted(svm.classes_, y_te)
    def nll(T):
        z = Z / T
        z = z - z.max(1, keepdims=True)
        return -np.mean(z[np.arange(len(z)), idx] - np.log(np.exp(z).sum(1)))
    temps = np.geomspace(0.005, 5.0, 60)
    T = float(temps[np.argmin([nll(t) for t in temps])])
    m = ReceiptClassifier(vec, prune(svm.coef_, args.prune), svm.intercept_, svm.classes_, T)
    pp = m.predict_proba(X_te)
    conf, acc = pp.max(1), (svm.classes_[pp.argmax(1)] == y_te)
    print(f"температура T={T:.3f}; средняя уверенность {conf.mean():.3f} при точности {acc.mean():.3f}; "
          f"после обнуления весов < {args.prune}: accuracy {acc.mean():.3f}, "
          f"ненулевых весов {m.coef.nnz / np.prod(m.coef.shape):.1%}")

    # 2) итоговая модель на всех данных
    vec, svm = fit(X, y, args.C)
    final = ReceiptClassifier(vec, prune(svm.coef_, args.prune), svm.intercept_, svm.classes_, T)
    joblib.dump(final, args.out, compress=3)
    print(f"сохранено: {args.out}  [{(time.time() - t0) / 60:.1f} мин]")


if __name__ == "__main__":
    main()
