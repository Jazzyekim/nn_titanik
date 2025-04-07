import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from model_torch.titanik import TitanicMLP, TitanicSLP

titanic_df = sns.load_dataset('titanic')

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
titanic_df = titanic_df[features + ['survived']]
titanic_df = titanic_df.dropna()
titanic_df = pd.get_dummies(titanic_df, columns=['sex'], drop_first=True)

X = titanic_df.drop('survived', axis=1).values
y = titanic_df['survived'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch
from torch.utils.data import TensorDataset, DataLoader

# Перетворення у тензори
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


def train_model(model='mlp', optimizer_type='adam', l2_lambda=0.0):
    model = {
        'mlp': TitanicMLP(),
        'slp': TitanicSLP()
    }[model]
    criterion = nn.BCELoss()
    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), weight_decay=l2_lambda),
        'sgd': torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=l2_lambda)
    }[optimizer_type]

    epochs = 100
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model


from sklearn.metrics import accuracy_score


def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_class = (y_pred > 0.5).int()
        acc = accuracy_score(y_test_tensor.int(), y_pred_class)
        print(f"Accuracy: {acc:.4f}")


print("=== Multi Layer Perceptron ===")
model_adam = train_model(model='mlp', optimizer_type='adam')
evaluate_model(model_adam)

print("=== Single Layer Perceptron ===")
model_adam = train_model(model='slp')
evaluate_model(model_adam)
