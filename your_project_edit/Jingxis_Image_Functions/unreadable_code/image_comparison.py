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
import numpy as np

#--------------------Cell--------------------
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
def run(img1path, img2path):
    img1 = readImg(img1path)
    img2 = readImg(img2path)
    error, diff = getError(img1, img2)
    print("Image matching Error between the two images:", error)
    cv2.imshow("Contour", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#--------------------Cell--------------------
if run_mse: run('../images/sample1.jpg','../images/sample2.jpg')

#--------------------Cell--------------------
run_FAST = False

#--------------------Cell--------------------
# Convert it to grayscale 
query_img_bw = readImg('../images/sample1.jpg')
train_img_bw = readImg('../images/sample1.jpg')

#--------------------Cell--------------------
def run_orb(query_img_bw, train_img_bw):
    # Initialize the ORB detector algorithm 
    orb = cv2.ORB_create() 
    
    # Now detect the keypoints and compute 
    # the descriptors for the query image 
    # and train image 
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 
    
    # Initialize the Matcher for matching 
    # the keypoints and then match the 
    # keypoints 
    matcher = cv2.BFMatcher() 
    matches = matcher.match(queryDescriptors,trainDescriptors) 
    
    # draw the matches to the final image 
    # containing both the images the drawMatches() 
    # function takes both images and keypoints 
    # and outputs the matched query image with 
    # its train image 
    query_img = cv2.imread('../images/sample1.jpg') 
    train_img = cv2.imread('../images/sample1.jpg') 
    final_img = cv2.drawMatches(query_img, queryKeypoints,  
    train_img, trainKeypoints, matches[:20],None) 
    
    final_img = cv2.resize(final_img, (1000,650)) 
    
    # Show the final image 
    cv2.imshow("Matches", final_img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

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
def compare_features_bf(_img1,_kp1,_dsc1,_img2,_kp2,_dsc2,_thres = 0):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(_dsc1, _dsc2, k=2)
    # Apply ratio test
    good_points = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            #matches_mask[i] = [1, 0]
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(_kp1) <= len(_kp2):
        number_keypoints = len(_kp1)
    else:
        number_keypoints = len(_kp2)

    # print("Keypoints 1ST Image: " + str(len(_kp1)))
    # print("Keypoints 2ND Image: " + str(len(_kp2)))
    # print("GOOD Matches:", len(good_points))
    # print("How good it's the match: ", len(good_points) / number_keypoints * 100)

    # Marking the keypoint on the image using circles
    img=cv2.drawKeypoints(_img1 ,_kp1 , _img1 , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('../target/image-with-keypoints1-bf.jpg', img)
    img=cv2.drawKeypoints(_img2 ,_kp2 , _img2 , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('../target/image-with-keypoints2-bf.jpg', img)

    return good_points , len(good_points) / number_keypoints * 100

#--------------------Cell--------------------
if run_sift_algo:
    img1, kp1, desc_1 = run_sift('../images/sample1.jpg')
    img2, kp2, desc_2 = run_sift('../images/sample2.jpg')

    goodpoints, percentage = compare_features_flann(img1,kp1,desc_1,img2,kp2,desc_2)
    
    print(percentage)
    print(goodpoints)

    goodpoints, percentage = compare_features_bf(img1, kp1, desc_1, img2, kp2, desc_2)

    print(percentage)
    print(goodpoints)

