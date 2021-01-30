import cv2

fps=30
size=(352,240)
out = cv2.VideoWriter(r"/home/suneel/ML/bg.avi",cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
imag=[]
for filename in range(0,999):
    img = cv2.imread(r"/home/suneel/ML/out_img/new_file{}.jpeg".format(filename))
    imag.append(img)
for filename in range(0,999):
    out.write(imag[filename])

out_1 = cv2.VideoWriter(r"/home/suneel/ML/fg.avi",cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

imag=[]
for filename in range(0,999):
    img = cv2.imread(r"/home/suneel/ML/out_img/new_frame{}.jpeg".format(filename))
    imag.append(img)

for filename in range(0,999):
    out_1.write(imag[filename])

out.release()
out_1.release()
print("created")