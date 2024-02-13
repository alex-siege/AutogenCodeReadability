#--------------------Cell--------------------
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#--------------------Cell--------------------
packages = ['scipy', 'numpy', 'pandas', 'opencv-python']

#--------------------Cell--------------------
id = 0
for pack in packages:
    print("("+str(id) + "/" + str(len(packages))+") "+"installing "+ pack + ".")
    install(pack)
    id += 1
    print("("+str(id)+ "/"+ str(len(packages))+ ") "+ pack + " installed.")

#--------------------Cell--------------------
import cv2

# Read an image using OpenCV
image = cv2.imread('../images/1.jpg')

# Check if the image was successfully loaded
if image is not None:
    # Display the image (optional)
    cv2.imshow('Image', image)
    cv2.waitKey(0)  # Wait for any key to close the window
    cv2.destroyAllWindows()
else:
    print("Image not found or could not be read.")

#--------------------Cell--------------------
import cv2
import numpy as np

# Read an image using OpenCV
image = cv2.imread('../images/1.jpg')
number_of_images = 0
images = []
show_preview = False
# Check if the image was successfully loaded
if image is not None:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors in the image
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Set the number of clusters (change as needed)
    num_clusters = 3
    
    # Apply K-means clustering to group keypoints into clusters
    if len(keypoints) > num_clusters:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(np.array([kp.pt for kp in keypoints], dtype=np.float32), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Create a dictionary to store keypoints for each cluster
        clustered_points = {tuple(kp.pt): [] for kp in keypoints}
        
        # Assign each keypoint to its corresponding cluster
        for i, label in enumerate(labels):
            clustered_points[tuple(keypoints[i].pt)].append(keypoints[i])
        
        # Crop around the bounding boxes of the clustered keypoints
        for cluster_keypoints in clustered_points.values():
            if len(cluster_keypoints) > 0:
                # Extract keypoints for computing bounding box
                cluster_points = np.array([kp.pt for kp in cluster_keypoints])
                
                # Compute bounding box for the cluster keypoints
                x_min, y_min = np.min(cluster_points, axis=0)
                x_max, y_max = np.max(cluster_points, axis=0)
                
                # Extract the cropped region for each cluster
                crop_margin_x = 80  # Define the crop margin around the bounding box (adjust as needed)
                crop_margin_y = 80  # Define the crop margin around the bounding box (adjust as needed)
                
                cropped_image = image[int(y_min - crop_margin_y):int(y_max + crop_margin_y),
                                      int(x_min - crop_margin_x):int(x_max + crop_margin_x)]
                
                # Display or save the cropped image for each cluster
                if cropped_image.size > 0 and cropped_image.size > 0:
                    number_of_images += 1
                    images.append(cropped_image)
                    
                    if show_preview:
                        #print(cropped_image)
                        cv2.imshow(f'Cropped Image - Cluster', cropped_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        show_preview = False
                # Save the cropped image
                #cv2.imwrite('cropped_image_cluster.jpg', cropped_image)
    else:
        print("Not enough keypoints to perform clustering.")

else:
    print("Image not found or could not be read.")

print(number_of_images)


#--------------------Cell--------------------
def compare_features_flann(_img1,_kp1,_dsc1,_img2,_kp2,_dsc2,_thres=0):

    # FLANN parameters
    FLANN_INDEX_KDTREE = 3
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(_dsc1, _dsc2, k=2)
    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0,] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good_points = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            #matches_mask[i] = [1, 0]
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(_kp1) <= len(_kp2):
        number_keypoints = len(_kp1)
    else:
        number_keypoints = len(_kp2)

    # Marking the keypoint on the image using circles
    img=cv2.drawKeypoints(_img1 ,_kp1 , _img1 , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('../target/image-with-keypoints1.jpg', img)
    img=cv2.drawKeypoints(_img2 ,_kp2 , _img2 , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('../target/image-with-keypoints2.jpg', img)

    return good_points , len(good_points) / number_keypoints * 100

#--------------------Cell--------------------
def run_sift(gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    return gray, kp, des 

#--------------------Cell--------------------
unique_img = []
for image in images:
    if len(unique_img) <= 0:
        img_copy = image.copy()
        img1, kp1, desc_1 = run_sift(image)
        unique_img.append((img_copy,(img1, kp1, desc_1)))
    for unique_image in unique_img:
        img_copy = image.copy()
        img1, kp1, desc_1 = run_sift(image)
        img2, kp2, desc_2 = unique_image[1]
        
        if len(kp1) >=2 and len(kp2) >= 2:
            goodpoints, percentage = compare_features_flann(img1,kp1,desc_1,img2,kp2,desc_2)
            if percentage < 0.001:
                unique_img.append((img_copy,(img1, kp1, desc_1)))

#--------------------Cell--------------------
print(len(unique_img))

#--------------------Cell--------------------
for image_1 in unique_img:
    base = image_1[0]
    img, _, _ = image_1[1]
    cv2.imshow(f'Cropped Image - Cluster', base)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

