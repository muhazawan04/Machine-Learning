import cv2
import numpy as np

def process_and_display_image(input_image_path):
    # Read the image
    img = cv2.imread(input_image_path)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (301,359))

    # Display the original image
    
    

    # Define the range of red color in RGB
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([255, 255, 255])
    # Create a binary mask identifying red pixels
    red_mask = cv2.inRange(img_rgb, lower_red, upper_red)

    # Create a black image with the same dimensions as the original image
    black_image = np.zeros_like(img_rgb)

    # Set red pixels to white in the black image
    black_image[red_mask != 0] = [255, 255, 255]

    # Display the processed image
    cv2.imshow('Original Image', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow('Processed Image', cv2.cvtColor(black_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_path = 'D:\\Mohaz\\Projects\\Machine_learning with python\Machine-Learning\\thalasemia\\Raw Data\\4.jpg'
process_and_display_image(input_path)


