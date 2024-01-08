from sklearn.model_selection import train_test_split
from model import train_model, match_fingerprint, load_and_extract_minutiae, read_new_folder
from metrics import calculate_metrics, calculate_accuracy
import pickle

def main():
    # Initialize y_train as an empty list
    y_train = []

    # Adjust the target_length as needed
    target_length = 256
    folder_path = "C:/Users/salam/Desktop/Biometrics/Database"
    X, y = load_and_extract_minutiae(folder_path, target_length=target_length)

    # Check the size of the dataset
    dataset_size = len(X)

    # Specify a minimum number of samples for testing
    min_test_samples = 10  # Adjust this value based on the size of your dataset

    # Check if there are enough samples to perform the split
    if dataset_size >= min_test_samples + 1:
        # Adjust the test size dynamically based on the dataset size
        test_size = min(min_test_samples / dataset_size, 0.2)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Check if the sizes are consistent after the split
        if len(X_train) == len(y_train) and len(X_test) == len(y_test):
            # Train a Nearest Neighbors model
            knn_model, fingerprint_database, scaler = train_model(X_train, y_train)

            # Save the trained model, fingerprint database, and scaler
            with open('knn_model.pkl', 'wb') as model_file:
                pickle.dump(knn_model, model_file)

            with open('fingerprint_database.pkl', 'wb') as database_file:
                pickle.dump(fingerprint_database, database_file)

            with open('scaler.pkl', 'wb') as scaler_file:
                pickle.dump(scaler, scaler_file)

            # Testing the system with the test set
            correct_matches = 0
            for query_template, true_label in zip(X_test, y_test):
                matched_label = match_fingerprint(query_template, knn_model, y_train, fingerprint_database, confidence_threshold=10, scaler=scaler)
                if matched_label == true_label:
                    correct_matches += 1

            accuracy = correct_matches / len(X_test)
            print(f"Accuracy: {accuracy}")

        else:
            print("Inconsistent sizes after train-test split.")
    else:
        print("Not enough samples to perform a train-test split.")


    # Testing the system with the given altered images
    approved_altered, rejected_altered, true_positives_altered, true_negatives_altered, false_positives_altered, false_negatives_altered = read_new_folder("C:/Users/salam/Desktop/Biometrics/Altered/2",
                                                                                                                                                knn_model,
                                                                                                                                                target_length=256,
                                                                                                                                                confidence_threshold=40,
                                                                                                                                                scaler=scaler
                                                                                                                                            )

    accuracy_altered = calculate_accuracy(approved_altered, approved_altered + rejected_altered)
    print(f"Accuracy on the altered images: {accuracy_altered}")

    # Calculate and print metrics for the noisy set
    metrics_altered = calculate_metrics(true_positives_altered, true_negatives_altered, false_positives_altered, false_negatives_altered)

    # Print metrics for noisy images
    print("Metrics for altered Images:")
    print("Accuracy:", metrics_altered[0])
    print("FAR:", metrics_altered[1])
    print("FRR:", metrics_altered[2])
    print("Precision:", metrics_altered[3])
    print("Recall:", metrics_altered[4])
    print("F1 Score:", metrics_altered[5])




    #Testing the system with the noisy images
    approved_noise, rejected_noise, true_positives_noise, true_negatives_noise, false_positives_noise, false_negatives_noise = read_new_folder("C:/Users/salam/Desktop/Biometrics/noise/4",
                                                                                                                                                knn_model,
                                                                                                                                                target_length=256,
                                                                                                                                                confidence_threshold=40,
                                                                                                                                                scaler=scaler
                                                                                                                                            )

    accuracy_noise = calculate_accuracy(approved_noise, approved_noise + rejected_noise)
    print(f"Accuracy on the noisy images: {accuracy_noise}")

    # Calculate and print metrics for the noisy set
    metrics_noise = calculate_metrics(true_positives_noise, true_negatives_noise, false_positives_noise, false_negatives_noise)

    # Print metrics for noisy images
    print("Metrics for Noisy Images:")
    print("Accuracy:", metrics_noise[0])
    print("FAR:", metrics_noise[1])
    print("FRR:", metrics_noise[2])
    print("Precision:", metrics_noise[3])
    print("Recall:", metrics_noise[4])
    print("F1 Score:", metrics_noise[5])
    
    # Test the system with rotated images
    approved_rotation, rejected_rotation, true_positives_rotation, true_negatives_rotation, false_positives_rotation, false_negatives_rotation = read_new_folder("C:/Users/salam/Desktop/Biometrics/Rotated_Images/3",
                                                                                                                                                                knn_model,
                                                                                                                                                                target_length=256,
                                                                                                                                                                confidence_threshold=40,
                                                                                                                                                                scaler=scaler
                                                                                                                                                            )

    accuracy_rotation = calculate_accuracy(approved_rotation, approved_rotation + rejected_rotation)
    print(f"Accuracy on the rotated images: {accuracy_rotation}")

    # Calculate and print metrics for the rotated set
    metrics_rotation = calculate_metrics(true_positives_rotation, true_negatives_rotation, false_positives_rotation, false_negatives_rotation)
    print("\nMetrics for Rotated Images:")
    print("Accuracy:", metrics_rotation[0])
    print("FAR:", metrics_rotation[1])
    print("FRR:", metrics_rotation[2])
    print("Precision:", metrics_rotation[3])
    print("Recall:", metrics_rotation[4])
    print("F1 Score:", metrics_rotation[5])
    

if __name__ == "__main__":
    main()