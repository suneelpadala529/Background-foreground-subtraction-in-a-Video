import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats as st
import glob
import os

videoCapture = cv2.VideoCapture(r"/home/suneel/ML/umcp.mpg");
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 176)
fps  = videoCapture.get(cv2.CAP_PROP_FPS);
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)));

print("\nfps, size -> ", fps, size)  
success, frame = videoCapture.read();
count = 0;

# Create the frames from the given video
while success:
    # Saves the frames with frame-count 
    cv2.imwrite(r"/home/suneel/ML/frames/frame%d.jpg" % count, frame)
    count += 1;   
    #print(frame.shape)  
    success, frame = videoCapture.read();

print("We have frames:", count);

# Create numpy arrrays for RGB from the frames generated
frame_Array = np.zeros((count, size[0], size[1]), dtype = int) 
frame_Array_bg = np.zeros((count, size[0], size[1]), dtype = float) 
frm = 0

while frm < count:
    frm_cur = cv2.imread(r"/home/suneel/ML/frames/frame{}.jpg".format(frm), cv2.IMREAD_GRAYSCALE)
    frame_Array[frm] = np.copy(frm_cur)
    frame_Array_bg[frm] = np.copy(frm_cur)
    frm = frm + 1 

print(frm)

# Model parameters
r_result = 255*(np.ones((count, size[0], size[1]), dtype = int))  # 0 - background, 1 - foreground
threshold = 0.65
num_K = 5
alpha = 0.2
print(size)

for i in range(0,size[0]):
    print(i)

    for j in range(0,size[1]):
        # For one pixel i, j -> prepare the GMM with the first 1000 frames 	    
        r = frame_Array[0:count, i, j]

        # Initialize -> Create mean, variance, SD, weight array for each pixel
        mean_var_dist_weight = np.zeros((num_K, 5), dtype = float) 
        mean_var_dist_weight[0,0] = np.mean(r)
        mean_var_dist_weight[0,1] = np.var(r)
        mean_var_dist_weight[0,2] = 2.5 * np.sqrt(mean_var_dist_weight[0,1])
        mean_var_dist_weight[0,3] = 1
        mean_var_dist_weight[0,4] = mean_var_dist_weight[0,3] / np.sqrt(mean_var_dist_weight[0,1])
        
        rho = 0
        frm = 0		
        for frm in range(0,count-1):
            r_temp = frame_Array[frm, i, j]

            for index in range(0, num_K):        
                if ((mean_var_dist_weight[index,3] == 0) or (index == num_K - 1)): 
                    mean_var_dist_weight[index,0] = r_temp
                    mean_var_dist_weight[index,1] =  mean_var_dist_weight[index,1] * 10
                    mean_var_dist_weight[index,2] = 2.5 * np.sqrt(mean_var_dist_weight[index,1])
                    mean_var_dist_weight[index,3] = mean_var_dist_weight[index,3] / 10
                    mean_var_dist_weight[index,4] = mean_var_dist_weight[index,3] / np.sqrt(mean_var_dist_weight[index,1])

                if ((r_temp < mean_var_dist_weight[index,0] + mean_var_dist_weight[index,2]) and 
                    (r_temp > mean_var_dist_weight[index,0] - mean_var_dist_weight[index,2])):
                    
                    sd = st.norm.pdf(mean_var_dist_weight[index,0], r_temp, np.sqrt(mean_var_dist_weight[index,1]))
                    rho = alpha * sd

                    mean_var_dist_weight[index,0] = (1 - rho) * mean_var_dist_weight[index,0] + rho * (r_temp)
                    mean_var_dist_weight[index,1] =  (1 - rho) * mean_var_dist_weight[0,1] + rho * (r_temp - mean_var_dist_weight[index,0]) ** 2
                    mean_var_dist_weight[index,2] = 2.5 * np.sqrt(mean_var_dist_weight[index,1])
                    mean_var_dist_weight[index,3] += alpha 
                    mean_var_dist_weight[index,4] = mean_var_dist_weight[index,3] / np.sqrt(mean_var_dist_weight[index,1])

            # udpate weights here and break            
            for index in range(0, num_K):
                mean_var_dist_weight[index,3] = (1 - alpha) * mean_var_dist_weight[index,3]

            # sort 'mean_var_dist_weight' after every input point based on 'weight'  
            newarray = sorted(mean_var_dist_weight, key=lambda x:x[4])
            mean_var_dist_weight = np.vstack(newarray)
            sum = 0.0
            max_K = 0
            # verify background or foreground
            for index in range(0, num_K):
                sum = sum + mean_var_dist_weight[index,3]
                if (sum > threshold):
                    max_K = index
                    break
            
            # verify background or foreground
            sum = 0
            for index in range(max_K,num_K):
                if ((r_temp < mean_var_dist_weight[index,0] + mean_var_dist_weight[index,2]) and
                    (r_temp > mean_var_dist_weight[index,0] - mean_var_dist_weight[index,2])):
                    r_result[frm][i][j] = 0
                    
                sum = mean_var_dist_weight[index, 0] * mean_var_dist_weight[index, 4]
                break
            frame_Array_bg[frm][i][j] = sum

print("We have background and Foreground")

#create video from frames
try:
    os.makedirs("/home/suneel/ML/out_img/")
except Exception as e:
    print(e)

#Create images from resultant matrices
for index in range(r_result.shape[0]):
    cv2.imwrite("/home/suneel/ML/out_img/new_file"+str(index)+".jpeg",r_result[index])
for index in range(frame_Array_bg.shape[0]):
    cv2.imwrite("/home/suneel/ML/out_img/new_frame"+str(index)+".jpeg",frame_Array_bg[index])

#Create videos from Images

      