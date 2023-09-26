import cv2


def Backgroud_detector(rgb_image, threshold=65, iteration_start=5, percent=8/100):
    img_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    sample = img_gray.copy()
    sample[sample >= threshold] = 255

    r, c = sample.shape
    _, thresh = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh[int(r*percent):int(r*(1-percent)), int(c*percent):int(c*(1-percent))] = 0
    thresh[thresh == 255] = 1

    thresh = cv2.erode(thresh, None, iterations=iteration_start)
    thresh = cv2.dilate(thresh, None, iterations=iteration_start+5)
    thresh = cv2.erode(thresh, None, iterations=iteration_start+15)
    thresh = cv2.dilate(thresh, None, iterations=iteration_start+10)

    return thresh
