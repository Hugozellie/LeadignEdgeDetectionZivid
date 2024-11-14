import zivid as z
import cv2 as cv
import numpy as np
import datetime as dt
import math

def connectCamera():
    # try and connect to the next available camera
    try:
        print("connecting")
        app = z.Application()
        cameras = app.cameras()
        for c in cameras:
            print(c.state)
        camera = app.connect_camera()
        print("connected")

        return camera
    # if no camera is available, exit the program
    except:
        print("unable to connect camera.\nCheck if camera is not busy.")
        exit()

def captureImage(camera: z.Camera):
    # set the settings as seen bellow
    print("setting settings")
    settings_2d = z.Settings2D()
    settings_2d.acquisitions.append(z.Settings2D.Acquisition())
    settings_2d.acquisitions[0].aperture = 1.8
    settings_2d.acquisitions[0].exposure_time = dt.timedelta(microseconds=10000)
    settings_2d.acquisitions[0].gain = 1.0
    settings_2d.acquisitions[0].brightness = 0.1
    settings_2d.acquisitions[0].gamma = 1.0
    print("settings set")

    with camera.capture(settings_2d) as frame_2d:
        # capture and save image as .png
        image = frame_2d.image_rgba()
        image.save("/home/hugo/projects/Zivid/Images/captured.png")
        print("captured")

def viewSettings(settings: z.settings.Settings):
    # prints the current list of settings
    print(settings)

def loadImage(path: str):
    # loads the image from the given path
    return cv.imread(path)

def viewImages(*img):
    # views all images given
    for i in range(len(img)):
        view = cv.resize(img[i],(0,0), fx=0.8, fy=0.8)
        cv.imshow(f"output {i}", view)
    cv.waitKey(0)
    cv.destroyAllWindows()

def saveImages(*img):
    # saves all images given
    for i in range(len(img)):
        filename = str("captured" + str(i) + ".png")
        cv.imwrite(filename, img[i])

def bgr2gray(convert):
    # converts a bgr image to a grayscale image
    return cv.cvtColor(convert, cv.COLOR_BGR2GRAY)

def bgr2hsv(convert):
    # converts a bgr imagse to a hsv image
    return cv.cvtColor(convert, cv.COLOR_BGR2HSV)

def maskImg(img,lower,upper):
    # masks an image using the given parameters
    imgBlur = cv.GaussianBlur(img,(11,11),0)
    img_mask = cv.inRange(imgBlur,lower,upper)

    return img_mask

def gradientImg(img):
    # uses canny edge detection to show all the edges
    imgBlur = cv.GaussianBlur(img,(5,5),0)
    canny = cv.Canny(image=imgBlur, threshold1=100, threshold2=175)

    return canny

def getAllContours(img, mask):
    # draws all large enough countours in the image
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, (0,255,0), 1)
    valid = []

    for cont in contours:
        m = cv.moments(cont)
        area = m["m00"]

        if area > 5000:
            valid.append(cont)
            cx = int(m["m10"] / area)
            cy = int(m["m01"] / area)

            cv.circle(img, (cx,cy), 2, (128,0,128), -1)

    return img, valid

def getCenterContour(img, contours):
    # shows the center most and largest contour in the image
    centers = []
    dist = []

    for cont in contours:
        m = cv.moments(cont)
        area = m["m00"]

        if area > 5000:
            cx = int(m["m10"] / area)
            cy = int(m["m01"] / area)

            centers.append((cx,cy))

    # get all centerpoints of the contours in a list
    for i in range(len(centers)):
        cx = centers[i][0]
        cy = centers[i][1]
        imgCenterH = int(img.shape[0] / 2)
        imgCenterW = int(img.shape[1] / 2) 

        dist.append(math.dist([cx,cy],[imgCenterW,imgCenterH]))

    # select only the closest one from the middle
    index = dist.index(min(dist))
    
    # show the closest middlepoint and contour
    cv.drawContours(img, contours, index, (0,255,0), 1)
    cv.circle(img, (centers[index][0],centers[index][1]), 2, (255,0,0), -1)
    return img

camera = connectCamera()
captureImage(camera)

og = loadImage("/home/hugo/projects/Zivid/Images/captured.png")
imgGray = bgr2gray(og.copy())
imgMask = maskImg(imgGray.copy(),15,80)
imgHSV = bgr2hsv(og.copy())
imgGradient = gradientImg(imgGray.copy())
imgContours,contours = getAllContours(og.copy(),imgMask)
# imgContour = getCenterContour(og.copy(),contours)
viewImages(imgGray,imgMask,imgGradient,imgContours)
saveImages(imgGray,imgMask,imgGradient,imgContours)