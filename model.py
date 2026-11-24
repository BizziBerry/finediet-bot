"""Компактная модель для бота: TF-IDF (слова + символьные n-граммы) + линейный SVM,
вероятности через softmax с температурой, веса хранятся разреженно в float32."""
import re

import numpy as np
from scipy import sparse


def clean(name) -> str:
    """Нормализация, одинаковая при обучении и в боте: нижний регистр, одиночные пробелы."""
    return re.sub(r"\s+", " ", str(name)).strip().lower()


class ReceiptClassifier:
    def __init__(self, vectorizer, coef, intercept, classes, temperature):
        self.vectorizer = vectorizer
        self.coef = sparse.csr_matrix(coef, dtype=np.float32)          # (классы x признаки)
        self.intercept = np.asarray(intercept, dtype=np.float32)
        self.classes_ = np.asarray(classes)
        self.temperature = float(temperature)

    def decision_function(self, texts):
        X = self.vectorizer.transform([clean(t) for t in texts]).astype(np.float32)
        return np.asarray((X @ self.coef.T).todense()) + self.intercept

    def predict_proba(self, texts):
        z = self.decision_function(texts) / self.temperature
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, texts):
        return self.classes_[self.decision_function(texts).argmax(axis=1)]
