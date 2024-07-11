import cv2


cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

num = 0


while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()


    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(r'C:\Users\mayan\AppData\Local\Programs\Python\Python311\images\cam1\imgl_' + str(num) + '.png', img)
        cv2.imwrite(r'C:\Users\mayan\AppData\Local\Programs\Python\Python311\images\cam2\imgr_' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)