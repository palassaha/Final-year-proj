import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("parkinsons\parkinsons.data")  # Replace with your actual file

# Select relevant features
features = [
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'PPE'
]

X = df[features]
y = df['status']  # 1 for PD, 0 for healthy

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=feature, hue='status', kde=True, element='step', palette='Set1')
    plt.title(f"{feature} Distribution by Status")
    plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', s=70)
plt.title("PCA of Selected Voice Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

plt.figure(figsize=(14, 10))
for i, feature in enumerate(features):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(data=df, x='status', y=feature, palette='pastel')
    plt.title(feature)
    plt.tight_layout()
plt.suptitle("Feature Distribution by Status", y=1.02)
plt.show()
