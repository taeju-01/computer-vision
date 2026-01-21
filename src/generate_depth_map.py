####기본적인DepthMap생성코드(OpenCV활용)
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# 이미지로드
image= cv2.imread('test_image.jpg')
# 그레이스케일변환
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 깊이맵생성(가상의깊이적용)
depth_map= cv2.applyColorMap(gray, cv2.COLORMAP_JET)
# 결과출력
cv2_imshow(image)
cv2_imshow(depth_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
