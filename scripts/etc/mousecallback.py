import region_growing as rd
import cv2



cls = rd.RegionGrowing()
img = cv2.resize(cv2.imread('../../image/rock/spiritsol118navcam.jpg'),(512,512))
cv2.imread('../../image/rock/sol729.jpg')
cv2.imshow('Input image',img)    
cv2.setMouseCallback('Input image',cls.mouse_event)

cv2.waitKey(-1)
cv2.destroyAllWindows()