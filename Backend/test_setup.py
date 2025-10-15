import cv2
import numpy as np
img = np.zeros((480,640,3), dtype='uint8')
cv2.putText(img, 'OpenCV GUI test', (50,240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
cv2.imshow('test', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
print('OK')