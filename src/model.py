import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from preprocess import preprocess_image, extract_minutiae

def train_model(X, y):
    # Normalize the feature vectors
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Save normalization parameters for later use during testing
    scaler_params = {'mean': scaler.mean_, 'std': scaler.scale_}
    with open('scaler_params.pkl', 'wb') as scaler_file:
        pickle.dump(scaler_params, scaler_file)

    knn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
    knn_model.fit(X_normalized)
    
    fingerprint_database = {label: [template for template, lbl in zip(X_normalized, y) if lbl == label] for label in set(y)}
    
    return knn_model, fingerprint_database, scaler

def load_scaler_params():
    # Load normalization parameters for testing
    with open('scaler_params.pkl', 'rb') as scaler_file:
        scaler_params = pickle.load(scaler_file)

    new_scaler = StandardScaler()
    new_scaler.mean_ = scaler_params['mean']
    new_scaler.scale_ = scaler_params['std']

    return new_scaler

# Function to match a fingerprint against the database
def match_fingerprint(query_template, knn_model, y_train, fingerprint_database, confidence_threshold=10, scaler=None):
    query_template_normalized = query_template  # Initialize the variable

    if scaler is not None:
        query_template_normalized = scaler.transform([query_template])[0]

    # Find the nearest neighbor
    distances, indices = knn_model.kneighbors([query_template_normalized], n_neighbors=1)

    # Check if any neighbors were found
    if not indices:
        print("Warning: No neighbors found. Consider verifying the result.")
        return None

    distance_to_nearest_neighbor = distances[0][0]

    # Get the label of the matched template
    matched_label = y_train[indices[0][0]]

    # Add detailed debug information
    if fingerprint_database and matched_label in fingerprint_database and fingerprint_database[matched_label]:
        print(f"Query Template: {query_template_normalized}")
        print(f"Training Template: {fingerprint_database[matched_label][0]}")
        print(f"Individual Feature Distances: {query_template_normalized - fingerprint_database[matched_label][0]}")
        print(f"Euclidean Distance: {distance_to_nearest_neighbor}")

    # Check if the distance is below the confidence threshold
    if distance_to_nearest_neighbor > confidence_threshold:
        print(f"Warning: Low confidence match (distance: {distance_to_nearest_neighbor}). Consider verifying the result.")
        return None

    return matched_label

# Function to match a single fingerprint image against the database
def match_single_image(image_path, knn_model, y_train, fingerprint_database, target_length=256, confidence_threshold=10, scaler=None):
    # Load and preprocess the single image
    fingerprint_image = preprocess_image(image_path, target_length)

    # Check if the image was loaded successfully
    if fingerprint_image is None:
        print("Error: Unable to load or preprocess the image.")
        return None

    # Extract minutiae from the preprocessed image
    minutiae = extract_minutiae(fingerprint_image, target_length)

    # Check if minutiae extraction was successful
    if minutiae is None:
        print("Error: Unable to extract minutiae from the image.")
        return None

    # Normalize the query template using the same scaler used during training
    if scaler is not None:
        minutiae_normalized = scaler.transform([minutiae])[0]

    # Match the fingerprint against the database
    matched_label = match_fingerprint(minutiae_normalized, knn_model, y_train, fingerprint_database, confidence_threshold)
    return matched_label

def read_new_folder(folder_path, model, target_length=256, confidence_threshold=10, scaler=None):
    # Load and extract minutiae from fingerprint images
    X, y = load_and_extract_minutiae(folder_path, target_length=target_length)

    # Initialize counters for approved, rejected, true positives, true negatives, false positives, and false negatives
    approved_count = 0
    rejected_count = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Create an empty fingerprint database
    fingerprint_database = {label: [] for label in set(y)}

    for query_template, true_label in zip(X, y):
        # Match the fingerprint against the database
        matched_label = match_fingerprint(query_template, model, y, fingerprint_database, confidence_threshold=confidence_threshold, scaler=scaler)

        # Check if the system approves the match
        if matched_label == true_label:
            approved_count += 1
            true_positives += 1
        else:
            rejected_count += 1
            false_negatives += 1

        # Update counters for false positives
        for label in set(y):
            if label != true_label and matched_label == label:
                false_positives += 1

        # Update counters for true negatives
        if matched_label is None and true_label not in fingerprint_database:
            true_negatives += 1

    return approved_count, rejected_count, true_positives, true_negatives, false_positives, false_negatives

def load_and_extract_minutiae(folder_path, target_length=256):
    X = []
    y = []

    for root, dirs, files in os.walk(folder_path):
        if not dirs:
            for file in files:
                if file.lower().endswith((".tif", ".bmp")):
                    image_path = os.path.join(root, file)
                    fingerprint_class = int(os.path.basename(root))

                    fingerprint_image = preprocess_image(image_path, target_length)
                    minutiae = extract_minutiae(fingerprint_image, target_length)

                    X.append(minutiae)
                    y.append(fingerprint_class)

    return np.array(X), np.array(y)