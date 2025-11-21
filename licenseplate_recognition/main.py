import cv2
import pytesseract
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def detect_plate_number(image_path):
    # Load the image
    image = cv2.imread(r'C:\Users\Dell\OneDrive\Desktop\python\licenseplate_recognition\car1.jpg')
    if image is None:
        print("Error: Image not found or path is incorrect.")
        return None
    # Display original image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edges = cv2.Canny(blurred, 100, 200)
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    plate_contour = None
    # Loop through contours to find rectangular shape
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # Likely a rectangle (plate)
            plate_contour = approx
            break
    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = gray[y:y + h, x:x + w]
        # Threshold for better OCR
        _, thresh = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Use Tesseract OCR to extract text
        plate_number = pytesseract.image_to_string(thresh, config='--psm 8')
        return plate_number.strip()
    else:
        print("License plate contour not found.")
        return None
# --- Run the function ---
image_path = r"C:\Users\Dell\OneDrive\Desktop\python\licenseplate_recognition\car1.jpg"
plate_number = detect_plate_number(image_path)
print("Detected Plate Number:", plate_number)

