####심화코드: DepthMap을기반으로3D 포인트클라우드생성
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# 이미지로드
image= cv2.imread('test_image.jpg')
# 그레이스케일변환
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# DepthMap생성
depth_map= cv2.applyColorMap(gray, cv2.COLORMAP_JET)
# 3D 포인트클라우드변환
h, w= depth_map.shape[:2]
X, Y= np.meshgrid(np.arange(w), np.arange(h))
Z= gray.astype(np.float32)  # Depth값을Z축으로사용
# 3D 좌표생성
points_3d = np.dstack((X, Y, Z))
# 결과출력
cv2_imshow(depth_map)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
