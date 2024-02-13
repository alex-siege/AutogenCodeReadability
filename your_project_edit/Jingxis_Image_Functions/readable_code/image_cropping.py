#--------------------Cell--------------------
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install_package", package])

#--------------------Cell--------------------
packages = ['scipy', 'numpy', 'pandas', 'opencv-python']

#--------------------Cell--------------------
def install_packages(packages):
    """
    Iterates through a list of package names and installs each package while printing the install progress.
    """
    for i, package in enumerate(packages):
        progress = f"({i}/{len(packages)}) installing {package}."
        print(progress)
        install_package(package)
        print(f"({i+1}/{len(packages)}) {package} installed.")

install_packages(packages)

#--------------------Cell--------------------
# Function to load and display an image, if the image is not found it prints an error
def load_and_display_image(image_path):
    """
    Reads an image from the specified path and displays it using OpenCV. Prints a message if the image cannot be read.
    """
    import cv2
    
    image = cv2.imread(image_path)
    
    if image is not None:
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found or could not be read.")

load_and_display_image('../images/1.jpg')

#--------------------Cell--------------------
# Function to perform image processing and feature clustering
def cluster_image_features(image_path, show_preview=False):
    """
    Reads an image from the specified path, converts it to grayscale, detects and clusters features using ORB and K-means.
    Optionally displays a preview of each cluster crop if 'show_preview' is set to True.
    """
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    number_of_images = 0
    images = []
    
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        num_clusters = 3
        
        if len(keypoints) > num_clusters:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, _ = cv2.kmeans(np.array([kp.pt for kp in keypoints], dtype=np.float32),
                                       num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            clustered_points = {tuple(kp.pt): [] for kp in keypoints}
            
            for i, label in enumerate(labels):
                clustered_points[tuple(keypoints[i].pt)].append(keypoints[i])
            
            for cluster_keypoints in clustered_points.values():
                if cluster_keypoints:
                    cluster_points = np.array([kp.pt for kp in cluster_keypoints])
                    x_min, y_min = np.min(cluster_points, axis=0)
                    x_max, y_max = np.max(cluster_points, axis=0)
                    margin_x, margin_y = 80, 80
                    cropped_image = image[int(y_min-margin_y):int(y_max+margin_y), 
                                          int(x_min-margin_x):int(x_max+margin_x)]
                    
                    if cropped_image.size > 0:
                        number_of_images += 1
                        images.append(cropped_image)
                        
                        if show_preview:
                            cv2.imshow('Cropped Image - Cluster', cropped_image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
        else:
            print("Not enough keypoints to perform clustering.")
    else:
        print("Image not found or could not be read.")

    print(number_of_images)
    return images

cluster_image_features('../images/1.jpg')

#--------------------Cell--------------------

def compare_features_flann(image1, keypoints1, descriptors1,
                           image2, keypoints2, descriptors2, threshold=0.6):
    """
    Compares features between two images using the FLANN based matcher and 
    returns the good matching points along with the percentage of keypoints matched.

    Args:
        image1: First input image.
        keypoints1: Detected keypoints of the first image.
        descriptors1: Descriptors of the keypoints for the first image.
        image2: Second input image.
        keypoints2: Detected keypoints of the second image.
        descriptors2: Descriptors of the keypoints for the second image.
        threshold: Threshold for the distance ratio test.

    Returns:
        A tuple of good matching points and the match percentage.
    """
    flann_matcher = create_flann_matcher()
    matches = flann_matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_points = get_good_matches(matches, threshold)

    save_keypoint_images(image1, keypoints1, 'image-with-keypoints1.jpg')
    save_keypoint_images(image2, keypoints2, 'image-with-keypoints2.jpg')

    number_keypoints = min(len(keypoints1), len(keypoints2))
    match_percentage = calculate_match_percentage(len(good_points), number_keypoints)

    return good_points, match_percentage


def create_flann_matcher():
    """
    Creates a FLANN based matcher with predefined parameters.

    Returns:
        A cv2.FlannBasedMatcher object.
    """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def get_good_matches(matches, threshold):
    """
    Filters the matches using the Lowe's ratio test.

    Args:
        matches: A list of matches obtained from knnMatch.
        threshold: Threshold value for Lowe's ratio test.

    Returns:
        A list of good matches after applying Lowe's ratio test.
    """
    good_points = [m for m, n in matches if m.distance < threshold * n.distance]
    return good_points


def save_keypoint_images(image, keypoints, filename):
    """
    Saves an image with keypoints drawn on it.

    Args:
        image: The image upon which to draw keypoints.
        keypoints: The detected keypoints.
        filename: Filename for the output image.
    """
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f'../target/{filename}', img_with_keypoints)


def calculate_match_percentage(good_points_count, number_keypoints):
    """
    Calculates the match percentage based on good points and total number of keypoints.

    Args:
        good_points_count: Number of good matching points after Lowe's ratio test.
        number_keypoints: Total number of keypoints in the smaller set of keypoints.

    Returns:
        The match percentage.
    """
    return good_points_count / number_keypoints * 100 if number_keypoints else 0


#--------------------Cell--------------------
def run_sift(gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    return gray, kp, des 

def is_unique(image, unique_images):
    """
    Determines if the image is unique compared to the list of unique images using SIFT features and FLANN matcher.
    """
    img1, kp1, desc_1 = run_sift(image)
    
    for _, (img2, kp2, desc_2) in unique_images:
        if len(kp1) >= 2 and len(kp2) >= 2:
            good_points, percentage = compare_features_flann(img1, kp1, desc_1, img2, kp2, desc_2)
            if percentage < 0.001:
                return False
    return True

def show_unique_images(unique_images):
    """
    Displays all unique images in a separate window.
    """
    for base, _ in unique_images:
        cv2.imshow(f'Cropped Image - Cluster', base)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main code block
unique_images = []
for image in images:
    if not unique_images:
        # Add the first image to the unique_images list by default
        image_copy = image.copy()
        img_sift_params = run_sift(image)
        unique_images.append((image_copy, img_sift_params))
    elif is_unique(image, unique_images):
        # If the image is unique, add a copy of it to the list
        image_copy = image.copy()
        img_sift_params = run_sift(image)
        unique_images.append((image_copy, img_sift_params))

# Print the count of unique images
print(len(unique_images))

# Show the unique images
show_unique_images(unique_images)


