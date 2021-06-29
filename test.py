import cv2



cv2.namedWindow("win",cv2.WINDOW_NORMAL)
img = cv2.imread('maze.jpeg',1)

for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		if img[i][j][0] == 113:
			print(i,j)

cv2.imshow("win",img)
cv2.waitKey(0)