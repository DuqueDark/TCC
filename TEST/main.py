#C:\Program Files\Tesseract-OCR


from pytesseract import pytesseract
import cv2

path_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract'
pytesseract.tesseract_cmd = path_tesseract

def plate(source):
    img = cv2.imread(source)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('teste',img)
    
    _, img_binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    #cv2.imshow('binary', img_binary)
    
    img_defocus = cv2.GaussianBlur(img_binary, (5,5), 0)
    #cv2.imshow('defocus', img_defocus)
    
    contours, hierarchy = cv2.findContours(img_defocus, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    
    for c in contours:
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(aprox) == 4:
                (x, y, alt, lar) = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + alt, y + lar), (0, 255, 0), 2)
                roi = img[y:y + lar, x:x + alt]
                cv2.imwrite('out/roi.png', roi)
    
    #cv2.imshow('IMG contours',img)

# treating photo
def treating_photo():
    img = cv2.imread(r'out\roi.png')

    img_size = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('IMG resize',img_size)

    img_gray = cv2.cvtColor(img_size, cv2.COLOR_BGR2GRAY)
    cv2.imshow('IMG gray',img_gray)

    _, img_binary = cv2.threshold(img_gray, 165, 255, cv2.THRESH_BINARY)
    cv2.imshow('IMG binary',img_binary)
    
    # img_desfoque = cv2.GaussianBlur(img_binary, (5, 5), 0)
    # cv2.imshow('IMG desfoque',img_desfoque)

    cv2.imwrite('out/roi-ocr.png', img_binary)

# read
def ocr_reading():
    img_text = cv2.imread('out/roi-ocr.png')

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

    return pytesseract.image_to_string(img_text, lang='eng', config=config)


def main():
    
    source = r'img\pic.jpg'
    
    plate(source)
    treating_photo()
    print(ocr_reading())
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
