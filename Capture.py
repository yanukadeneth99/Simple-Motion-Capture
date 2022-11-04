# Imports
import cv2
import imutils
import time

# Initialising the Camera
cam = cv2.VideoCapture(0)

# Sleep for a second waiting for the camera to load
time.sleep(1)

# Keeps the first Image frame
firstFrame = None

# Standardizing the area to prevent the img size changing per device (camera)
area = 500

while True:
    # Reading Camera Frame
    _, img = cam.read()

    # Setting Text to display
    text = "Normal"

    # Resizing Image
    img = imutils.resize(img, width=area)

    # Converting Image to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smoothens Image
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    # Set the first frame as Gaussian Image
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    # Get the absolute difference between the first and current frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)

    # Applying Threshold and Dilating to remove any pixel issues
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    # Collecting the counts of contours
    cnts = cv2.findContours(
        threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Looping over every count
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue

        # Get the Dimentions of the Bounding Box
        (x, y, w, h) = cv2.boundingRect(c)

        # Draw the box
        cv2.rectangle(img, (x, y), (x + w, y+h), (0, 255, 0), 2)

        # Change Text
        text = "Moving Object Detected"

    # Print out the text
    print(text)

    # Render the text onto screen
    cv2.putText(img, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the Frame
    cv2.imshow("Camera Feed", img)

    # Break the Loop if Q key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Releasing Camera and Destroying all Windows
cam.release()
cv2.destroyAllWindows()
