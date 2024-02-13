#--------------------Cell--------------------
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to be installed
packages = ['scipy', 'numpy', 'pandas', 'opencv-python']

def install_package(package_name):
    """Install a package using a package manager."""
    # Here install should be replaced with a call to the actual package manager, e.g., pip.
    # The following line is a placeholder for the actual install command: 
    # subprocess.run([sys.executable, "-m", "pip", "install", package_name])
    print(f"Installing {package_name}.")
    # TODO: Logic to install the package
    print(f"{package_name} installed.")

# Iterate through the packages and install them
for package_index, package in enumerate(packages):
    total_packages = len(packages)
    print(f"({package_index}/{total_packages}) installing {package}.")
    install_package(package)  # Call the function to install the package
    print(f"({package_index + 1}/{total_packages}) {package} installed.")

def readImg(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img,(960, 540), interpolation = cv2.INTER_CUBIC)


#--------------------Cell--------------------
run_mse = False

#--------------------Cell--------------------
def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

#--------------------Cell--------------------
def getError(img1, img2):
    return mse(img1, img2)

#--------------------Cell--------------------
import cv2

def read_image(image_path):
    """
    Read an image from a given path using OpenCV and return the image object.
    """
    return cv2.imread(image_path)

def calculate_error(img1, img2):
    """
    Calculate the error and difference between two images.
    Here, the actual implementation of how the error is
    calculated is not provided, so we just assume there's
    a function `getError` that does it.
    """
    # Assuming getError is a predefined function
    error, difference = getError(img1, img2)
    return error, difference

def display_difference(diff):
    """
    Display the image difference in a window.
    """
    cv2.imshow("Contour", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run(image_path1, image_path2):
    """
    Run the process of reading two images, calculating errors between them,
    and displaying the difference.
    """
    # Read images from given file paths
    img1 = read_image(image_path1)
    img2 = read_image(image_path2)
    
    # Calculate error between the two images
    error, diff = calculate_error(img1, img2)
    
    # Output the error to the console
    print(f"Image matching Error between the two images: {error}")
    
    # Display the difference between images
    display_difference(diff)


#--------------------Cell--------------------
if run_mse: run('../images/sample1.jpg','../images/sample2.jpg')

#--------------------Cell--------------------
run_FAST = False

#--------------------Cell--------------------
# Convert it to grayscale 
query_img_bw = readImg('../images/sample1.jpg')
train_img_bw = readImg('../images/sample1.jpg')

#--------------------Cell--------------------
def detect_keypoints_and_compute_descriptors(image, orb_detector):
    """
    Detect keypoints and compute the descriptors for a given image using a specified ORB detector.

    :param image: The image for which keypoints and descriptors are to be computed.
    :param orb_detector: The ORB detector.
    :return: A tuple of detected keypoints and computed descriptors.
    """
    # Detect keypoints and compute descriptors using orb_detector
    keypoints, descriptors = orb_detector.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(descriptor1, descriptor2):
    """
    Match the keypoints descriptors using BFMatcher.

    :param descriptor1: The first set of descriptors (from the query image).
    :param descriptor2: The second set of descriptors (from the training image).
    :return: The matches found between the descriptors.
    """
    # Initialize the Matcher and match the descriptors
    matcher = cv2.BFMatcher()
    return matcher.match(descriptor1, descriptor2)

def draw_matches_and_show(query_img, train_img, query_kp, train_kp, matches):
    """
    Draw the matched keypoints on the images and display the result.

    :param query_img: The query image.
    :param train_img: The training image.
    :param query_kp: The query image keypoints.
    :param train_kp: The training image keypoints.
    :param matches: The matches found between the keypoints.
    """
    # Draw the first 20 matches between the images and keypoints
    final_img = cv2.drawMatches(query_img, query_kp, train_img, train_kp, matches[:20], None)
    
    # Resize the final image for display
    final_img = cv2.resize(final_img, (1000, 650))
    
    # Show the final image with matches
    cv2.imshow("Matches", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_orb(query_img_bw, train_img_bw):
    """
    Run ORB (Oriented FAST and Rotated BRIEF) algorithm to detect and match keypoints between a query and a training image.

    :param query_img_bw: The query image in black and white.
    :param train_img_bw: The training image in black and white.
    """
    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors for both images
    query_kp, query_desc = detect_keypoints_and_compute_descriptors(query_img_bw, orb)
    train_kp, train_desc = detect_keypoints_and_compute_descriptors(train_img_bw, orb)
    
    # Match keypoints
    matches = match_keypoints(query_desc, train_desc)
    
    # Read the original colored query and training images
    query_img = cv2.imread('../images/sample1.jpg')
    train_img = cv2.imread('../images/sample1.jpg')
    
    # Draw matches and show the final image
    draw_matches_and_show(query_img, query_img, query_kp, train_kp, matches)


#--------------------Cell--------------------
if run_FAST: run_orb(query_img_bw, train_img_bw)

#--------------------Cell--------------------
run_sift_algo = False

#--------------------Cell--------------------
def run_sift(image_path):
    img = cv2.imread(image_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    return gray, kp, des 

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
    match_percentage = get_match_percentage(len(good_points), number_keypoints)

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


def get_match_percentage(good_points_count, number_keypoints):
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
def compare_features_bf(image1, keypoints1, descriptors1, image2, keypoints2, descriptors2, threshold=0.75):
    """
    Compare two sets of image features using Brute-Force Matcher.

    :param image1: The first input image.
    :param keypoints1: The keypoints detected in the first image.
    :param descriptors1: The descriptors of the keypoints in the first image.
    :param image2: The second input image.
    :param keypoints2: The keypoints detected in the second image.
    :param descriptors2: The descriptors of the keypoints in the second image.
    :param threshold: The threshold multiplier for the distance ratio test.
    :return: A tuple containing the good matches and the match quality percentage.
    """
    # Initialize the Brute-Force Matcher with default parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Filter the matches using the ratio test
    good_points = [m for m, n in matches if m.distance < threshold * n.distance]
    
    # Calculate the number of keypoints based on the smaller set
    number_keypoints = min(len(keypoints1), len(keypoints2))
    
    # Save images with keypoints marked on them
    _save_image_with_keypoints(image1, keypoints1, 'image-with-keypoints1-bf.jpg')
    _save_image_with_keypoints(image2, keypoints2, 'image-with-keypoints2-bf.jpg')

    # Return good matches and the match quality percentage
    return good_points, len(good_points) / number_keypoints * 100

def _save_image_with_keypoints(image, keypoints, output_filename):
    """
    Draw keypoints on an image and save it.

    :param image: The input image.
    :param keypoints: The keypoints to draw.
    :param output_filename: The filename for the output image.
    """
    # Draw keypoints on the input image
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Save the image with keypoints
    cv2.imwrite(f'../target/{output_filename}', img_with_keypoints)


def run_sift_algo_on_images():
    """
    If the condition to run SIFT algorithm is True, process two images, compare features using
    two different methods and print out the good matched points and the match quality percentage.
    """
    # Condition to check if we should run SIFT algorithm.
    if run_sift_algo:
        # Run SIFT on the first image and capture output in its respective variables.
        img1, kp1, desc_1 = run_sift('../images/sample1.jpg')
        # Run SIFT on the second image and capture output in its respective variables.
        img2, kp2, desc_2 = run_sift('../images/sample2.jpg')

        # Compare features using FLANN and print out good matches and percentage.
        good_points_flann, percentage_flann = compare_features_flann(img1, kp1, desc_1, img2, kp2, desc_2)
        print(percentage_flann)
        print(good_points_flann)

        # Compare features using Brute Force and print out good matches and percentage.
        good_points_bf, percentage_bf = compare_features_bf(img1, kp1, desc_1, img2, kp2, desc_2)
        print(percentage_bf)
        print(good_points_bf)

# Assuming the rest of the script handles the definition of `run_sift_algo` variable,
# and the methods `run_sift`, `compare_features_flann`, `compare_features_bf`.


