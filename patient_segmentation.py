# ============================================================
# PATIENT SEGMENTATION USING K-MEANS AND HIERARCHICAL CLUSTERING
# ============================================================

print("==============================================")
print("   PATIENT SEGMENTATION PROJECT")
print("==============================================")

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

print("\n[1/8] Loading required libraries...")

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

print("Libraries loaded successfully.")


# ------------------------------------------------------------
# 2. CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

print("\n[2/8] Creating output folders...")

os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("graphs folder ready.")
print("results folder ready.")


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

print("\n[3/8] Loading dataset...")

file_name = "patient_segmentation_dataset.csv"

if not os.path.exists(file_name):
    print("\nERROR: Dataset file not found!")
    print("Make sure patient_segmentation_dataset.csv")
    print("is inside the Patient_Segmentation folder.")
    exit()

df = pd.read_csv(file_name)

print("Dataset loaded successfully.")

print("\nDataset Information")
print("-------------------")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 records:")
print(df.head())


# ------------------------------------------------------------
# 4. BASIC DATA CLEANING
# ------------------------------------------------------------

print("\n[4/8] Checking and cleaning dataset...")

print("\nMissing values before preprocessing:")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

# Remove duplicate records
duplicates = df.duplicated().sum()

if duplicates > 0:
    print("\nDuplicate records found:", duplicates)
    df = df.drop_duplicates()
    print("Duplicate records removed.")
else:
    print("\nNo duplicate records found.")

print("\nFinal dataset size:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

print("\n[5/8] Creating exploratory data analysis graphs...")


# Age distribution
if "Age" in df.columns:

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=20, kde=True)
    plt.title("Patient Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig("graphs/age_distribution.png")
    plt.close()

    print("Created: age_distribution.png")


# BMI distribution
if "BMI" in df.columns:

    plt.figure(figsize=(8, 5))
    sns.histplot(df["BMI"], bins=20, kde=True)
    plt.title("Patient BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig("graphs/bmi_distribution.png")
    plt.close()

    print("Created: bmi_distribution.png")


# Annual visits
if "Annual_Visits" in df.columns:

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Annual_Visits"], bins=15, kde=True)
    plt.title("Annual Hospital Visits")
    plt.xlabel("Annual Visits")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig("graphs/annual_visits.png")
    plt.close()

    print("Created: annual_visits.png")


# Insurance distribution
if "Insurance_Type" in df.columns:

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Insurance_Type")
    plt.title("Insurance Type Distribution")
    plt.xlabel("Insurance Type")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("graphs/insurance_distribution.png")
    plt.close()

    print("Created: insurance_distribution.png")


# ------------------------------------------------------------
# 6. SELECT FEATURES FOR MACHINE LEARNING
# ------------------------------------------------------------

print("\n[6/8] Preparing features for clustering...")


# Numerical features
numerical_features = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Num_Chronic_Conditions",
    "Annual_Visits",
    "Avg_Billing_Amount",
    "Days_Since_Last_Visit",
    "Preventive_Care_Flag"
]

# Categorical features
categorical_features = [
    "Gender",
    "State",
    "Insurance_Type",
    "Primary_Condition"
]


# Keep only columns that actually exist
numerical_features = [
    column for column in numerical_features
    if column in df.columns
]

categorical_features = [
    column for column in categorical_features
    if column in df.columns
]


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# 7. PREPROCESSING
# ------------------------------------------------------------

print("\nPreprocessing numerical and categorical data...")


# Numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# Transform dataset
X = preprocessor.fit_transform(df)

# Convert sparse matrix if necessary
if hasattr(X, "toarray"):
    X = X.toarray()

print("Preprocessing completed.")

print("Processed data shape:", X.shape)


# ------------------------------------------------------------
# 8. K-MEANS CLUSTERING
# ------------------------------------------------------------

print("\n[7/8] Running K-Means clustering...")

print("\nTesting different K values...")

k_values = range(2, 11)

inertia_values = []
silhouette_values = []


for k in k_values:

    print("Testing K =", k)

    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_temp = kmeans_temp.fit_predict(X)

    inertia_values.append(kmeans_temp.inertia_)

    score = silhouette_score(X, labels_temp)

    silhouette_values.append(score)


# ------------------------------------------------------------
# FIND BEST K USING SILHOUETTE SCORE
# ------------------------------------------------------------

best_k = list(k_values)[np.argmax(silhouette_values)]

best_silhouette = max(silhouette_values)

print("\nBest K according to Silhouette Score:", best_k)
print("Best Silhouette Score:", round(best_silhouette, 4))


# ------------------------------------------------------------
# ELBOW GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.grid(True)

plt.tight_layout()

plt.savefig("graphs/elbow_method.png")

plt.close()

print("Created: elbow_method.png")


# ------------------------------------------------------------
# SILHOUETTE SCORE GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.grid(True)

plt.tight_layout()

plt.savefig("graphs/silhouette_scores.png")

plt.close()

print("Created: silhouette_scores.png")


# ------------------------------------------------------------
# FINAL K-MEANS MODEL
# ------------------------------------------------------------

print("\nRunning final K-Means model...")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X)

df["KMeans_Cluster"] = kmeans_labels


print("K-Means completed successfully.")


# ------------------------------------------------------------
# K-MEANS CLUSTER COUNTS
# ------------------------------------------------------------

cluster_counts = df["KMeans_Cluster"].value_counts().sort_index()

print("\nK-Means Cluster Distribution:")
print(cluster_counts)


plt.figure(figsize=(8, 5))

sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values
)

plt.title("Number of Patients in Each K-Means Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("graphs/kmeans_cluster_counts.png")

plt.close()

print("Created: kmeans_cluster_counts.png")


# ------------------------------------------------------------
# PCA FOR K-MEANS VISUALIZATION
# ------------------------------------------------------------

print("\nCreating PCA visualization for K-Means...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=30
)

plt.title("K-Means Patient Segmentation")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(scatter, label="Cluster")

plt.tight_layout()

plt.savefig("graphs/kmeans_pca.png")

plt.close()

print("Created: kmeans_pca.png")


# ------------------------------------------------------------
# 9. HIERARCHICAL CLUSTERING
# ------------------------------------------------------------

print("\n[8/8] Running Hierarchical Clustering...")

print("Creating dendrogram...")


# Use a sample for dendrogram if dataset is large
sample_size = min(500, len(X))

np.random.seed(42)

sample_indices = np.random.choice(
    len(X),
    size=sample_size,
    replace=False
)

X_sample = X[sample_indices]


# Ward linkage
linkage_matrix = linkage(
    X_sample,
    method="ward"
)


plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Patient Groups")
plt.ylabel("Distance")

plt.tight_layout()

plt.savefig("graphs/dendrogram.png")

plt.close()

print("Created: dendrogram.png")


# ------------------------------------------------------------
# FINAL HIERARCHICAL MODEL
# ------------------------------------------------------------

print("\nRunning Agglomerative Hierarchical Clustering...")

hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X)

df["Hierarchical_Cluster"] = hierarchical_labels

print("Hierarchical clustering completed successfully.")


# ------------------------------------------------------------
# HIERARCHICAL SILHOUETTE SCORE
# ------------------------------------------------------------

hierarchical_silhouette = silhouette_score(
    X,
    hierarchical_labels
)

print(
    "Hierarchical Silhouette Score:",
    round(hierarchical_silhouette, 4)
)


# ------------------------------------------------------------
# HIERARCHICAL PCA VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=hierarchical_labels,
    cmap="plasma",
    s=30
)

plt.title("Hierarchical Patient Segmentation")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig("graphs/hierarchical_pca.png")

plt.close()

print("Created: hierarchical_pca.png")


# ------------------------------------------------------------
# 10. ALGORITHM COMPARISON
# ------------------------------------------------------------

print("\nComparing clustering algorithms...")


kmeans_silhouette = silhouette_score(
    X,
    kmeans_labels
)


comparison = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Hierarchical Clustering"
    ],

    "Number_of_Clusters": [
        best_k,
        best_k
    ],

    "Silhouette_Score": [
        kmeans_silhouette,
        hierarchical_silhouette
    ]
})


print("\nAlgorithm Comparison:")
print(comparison)


comparison.to_csv(
    "results/algorithm_comparison.csv",
    index=False
)


# Comparison graph
plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison,
    x="Algorithm",
    y="Silhouette_Score"
)

plt.title("K-Means vs Hierarchical Clustering")

plt.xlabel("Algorithm")

plt.ylabel("Silhouette Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("graphs/algorithm_comparison.png")

plt.close()

print("Created: algorithm_comparison.png")


# ------------------------------------------------------------
# 11. CLUSTER PROFILES
# ------------------------------------------------------------

print("\nCreating cluster profiles...")


profile_columns = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Num_Chronic_Conditions",
    "Annual_Visits",
    "Avg_Billing_Amount",
    "Days_Since_Last_Visit",
    "Preventive_Care_Flag"
]


profile_columns = [
    column for column in profile_columns
    if column in df.columns
]


cluster_profiles = df.groupby(
    "KMeans_Cluster"
)[profile_columns].mean()


print("\nK-Means Cluster Profiles:")
print(cluster_profiles)


cluster_profiles.to_csv(
    "results/cluster_profiles.csv"
)


# ------------------------------------------------------------
# 12. SAVE FINAL PATIENT RESULTS
# ------------------------------------------------------------

print("\nSaving final patient segmentation results...")


df.to_csv(
    "results/patient_segmentation_results.csv",
    index=False
)


# ------------------------------------------------------------
# 13. SAVE SILHOUETTE RESULTS
# ------------------------------------------------------------

silhouette_results = pd.DataFrame({
    "K": list(k_values),
    "Silhouette_Score": silhouette_values,
    "Inertia": inertia_values
})


silhouette_results.to_csv(
    "results/silhouette_results.csv",
    index=False
)


# ------------------------------------------------------------
# 14. FINAL OUTPUT
# ------------------------------------------------------------

print("\n==============================================")
print("       PROJECT COMPLETED SUCCESSFULLY")
print("==============================================")

print("\nDataset:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nK-Means:")
print("Optimal K:", best_k)
print(
    "Silhouette Score:",
    round(kmeans_silhouette, 4)
)

print("\nHierarchical Clustering:")
print(
    "Silhouette Score:",
    round(hierarchical_silhouette, 4)
)

print("\nGenerated folders:")
print("1. graphs/")
print("2. results/")

print("\nGenerated graphs:")
print("- age_distribution.png")
print("- bmi_distribution.png")
print("- annual_visits.png")
print("- insurance_distribution.png")
print("- elbow_method.png")
print("- silhouette_scores.png")
print("- kmeans_cluster_counts.png")
print("- kmeans_pca.png")
print("- dendrogram.png")
print("- hierarchical_pca.png")
print("- algorithm_comparison.png")

print("\nGenerated result files:")
print("- cluster_profiles.csv")
print("- algorithm_comparison.csv")
print("- patient_segmentation_results.csv")
print("- silhouette_results.csv")

print("\n==============================================")
print("       ALL PROCESSING FINISHED")
print("==============================================")