import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD

# 1. Завантаження та підготовка даних
df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]
df = df.dropna()

# Кодування категоріальних змінних
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Вхідні та вихідні змінні
X = df.drop('survived', axis=1)
y = df['survived']

# Масштабування ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Поділ на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Побудова моделі MLP
def build_model(optimizer='adam', use_dropout=True, use_l2=False):
    model = Sequential()
    regularizer = l2(0.01) if use_l2 else None

    model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizer))
    if use_dropout:
        model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizer))
    if use_dropout:
        model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Навчання моделі з різними оптимізаторами
results = {}
for opt_name in ['adam', 'sgd']:
    print(f"\nНавчання з оптимізатором: {opt_name.upper()}")
    model = build_model(optimizer=opt_name, use_dropout=True, use_l2=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.1)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    acc = accuracy_score(y_test, y_pred)
    results[opt_name] = acc
    print(f"Точність з {opt_name.upper()}: {acc:.4f}")

# 4. Порівняння
print("\n--- Результати ---")
for opt, acc in results.items():
    print(f"{opt.upper()} точність: {acc:.4f}")