import cv2
import numpy as np


def detect_and_crop_face(img):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        grayscale_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # For simplicity, we take the first detected face.
    (x, y, w, h) = faces[0]
    cropped_face = grayscale_img[y : y + h, x : x + w]

    return cv2.resize(cropped_face, (100, 100))


def choose_k(eigenvalues, variance_threshold=0.95):
    total_variance = sum(eigenvalues)
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)
    cumulative_variance = 0
    k = 0
    for eigenvalue in sorted_eigenvalues:
        cumulative_variance += eigenvalue
        k += 1
        if cumulative_variance / total_variance >= variance_threshold:
            break
    return k


def eigenfaces(train_images):
    data = [img.flatten() for img in train_images]
    data = np.array(data)

    mean_image = np.mean(data, axis=0)
    normalized_data = data - mean_image

    # Using SVD for faster computation
    u, s, vt = np.linalg.svd(normalized_data, full_matrices=False)

    # Get the eigenvalues from the square of singular values
    eigenvalues = s**2
    k = choose_k(eigenvalues)

    eigenvectors = vt[:k, :]

    return mean_image, eigenvectors


def project_onto_eigenfaces(img, mean_face, eigenvectors):
    return np.dot(img - mean_face, eigenvectors.T)


def recognize(test_image, train_images, mean_face, eigenvectors, labels):
    test_projection = project_onto_eigenfaces(
        test_image.flatten(), mean_face, eigenvectors
    )
    projections = [
        project_onto_eigenfaces(img.flatten(), mean_face, eigenvectors)
        for img in train_images
    ]

    min_distance = float("inf")
    min_distance_index = -1
    for i, proj in enumerate(projections):
        distance = np.linalg.norm(proj - test_projection)
        if distance < min_distance:
            min_distance = distance
            min_distance_index = i

    # If the distance is too large, we might say the face is unknown.
    # This threshold can be adjusted based on testing and use case.
    threshold = 5000
    if min_distance > threshold:
        return "Unknown"
    return labels[min_distance_index]
