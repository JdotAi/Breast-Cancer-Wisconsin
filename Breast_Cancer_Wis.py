import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import GridSearchCV

def prepare_data():
    data = pd.read_csv("data.csv")
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1
    )

    return X_train, X_test, y_train, y_test, X_scaled, y

def run_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf', probability=True, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return {"Model": "SVM", "Accuracy": accuracy, "F1 Score": f1, "Precision": precision, "Recall": recall}

def run_kmeans(X_train, X_test, y_train, y_test, X_scaled, y):
    model = KMeans(n_clusters=2, random_state=1)
    model.fit(X_scaled)
    clusters = model.labels_
    ari = adjusted_rand_score(y, clusters)

    plt.figure(figsize=(8, 6))
    for i in range(len(X_scaled)):
        color = 'red' if y.iloc[i] == 1 else 'green'
        plt.scatter(X_scaled[i, 0], X_scaled[i, 1], color=color, alpha=0.6, s=10)

    centroids = model.cluster_centers_
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], s=200, c='blue', marker='x')
        circle = plt.Circle((centroid[0], centroid[1]), radius=np.std(X_scaled[:, 0]), color='blue', fill=False, linestyle='dotted')
        plt.gca().add_artist(circle)

    plt.title("KMeans Clustering (Red: Malignant, Green: Benign)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("kmeans_clustering.png")

    return {"Model": "KMeans", "Adjusted Rand Index": ari}

def tune_neural_network(X_train, y_train):

    param_grid = {
        'hidden_layer_sizes': [(50,), (50, 25), (100, 50)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'activation': ['relu', 'tanh']
    }
    model = MLPClassifier(max_iter=1000, random_state=1)

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    return grid_search.best_estimator_

def run_neural_network(X_train, X_test, y_train, y_test):
    best_model = tune_neural_network(X_train, y_train)

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {
        "Model": "Neural Network",
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
    }


def display_results(results):
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_performance_comparison.csv", index=False)
    print("Results saved to 'model_performance_comparison.csv'")
    print(results_df)


def main():
    X_train, X_test, y_train, y_test, X_scaled, y = prepare_data()

    results = []
    results.append(run_svm(X_train, X_test, y_train, y_test))
    results.append(run_kmeans(X_train, X_test, y_train, y_test, X_scaled, y))
    results.append(run_neural_network(X_train, X_test, y_train, y_test))

    display_results(results)

if __name__ == "__main__":
    main()
