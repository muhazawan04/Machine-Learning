import cv2
import numpy as np

# Define a function to calculate variable section widths based on a list of percentages
def calculate_variable_widths(image_dimension, percentages):
    section_widths = []
    for percentage in percentages:
        section_width = int(image_dimension * (percentage / 100))
        section_widths.append(section_width)
    return section_widths

# Function to perform contrast stretching
def contrast_stretching(img, min_out, max_out):
    min_in, max_in = np.min(img), np.max(img)
    stretched_img = ((img - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
    return np.uint8(stretched_img)

# Read the image
image = cv2.imread("D:\\Mohaz\\Projects\\Machine_learning with python\Machine-Learning\\thalasemia\Raw Data\\15.jpg")
image = cv2.resize(image, (301, 359))
lower_red = np.array([60, 60, 150])
upper_red = np.array([255, 255, 255])
    # Create a binary mask identifying red pixels
red_mask = cv2.inRange(image, lower_red, upper_red)

    # Create a black image with the same dimensions as the original image
enhanced_image = np.zeros_like(image)

    # Set red pixels to white in the black image
enhanced_image[red_mask != 0] = [255, 255, 255]


# Define the list of percentages for variable section widths in both x and y axes
percentages_x = [4, 6, 6, 6, 35, 8, 15, 10, 10]
percentages_y = [14, 12.5, 12, 12, 12, 13, 12, 13.5]

# Calculate the section widths based on percentages and image width and height
section_widths_x = calculate_variable_widths(image.shape[1], percentages_x)
section_widths_y = calculate_variable_widths(image.shape[0], percentages_y)

# Set the threshold for average pixel value
avg = np.mean(enhanced_image)
threshold = avg- avg*0.08  # Adjust this value as needed

# Initialize empty list to store red sections and section boundaries
red_sections = []
section_boundaries_x = []
section_boundaries_y = []

# Loop through each section in x-axis
for i, section_width_x in enumerate(section_widths_x):
    if i == 0:
        start_x = 0
    else:
        start_x = section_boundaries_x[i - 1]  # Use the previous section boundary as the starting point
    end_x = start_x + section_width_x

    section_boundaries_x.append(end_x)

    # Loop through each section in y-axis
    for j, section_width_y in enumerate(section_widths_y):
        if j == 0:
            start_y = 0
        else:
            start_y = section_boundaries_y[j - 1]  # Use the previous section boundary as the starting point
        end_y = start_y + section_width_y

        section_boundaries_y.append(end_y)

        # Extract the current section from the enhanced image
        section = enhanced_image[start_y:end_y, start_x:end_x]

        # Calculate the average pixel value
        avg_pixel_value = np.mean(section)

        # Check if the average pixel value is below the threshold
        if avg_pixel_value < threshold:
            red_sections.append((j, i))

            # Determine the label based on the column index
            if i == 6:
                label = "major" 
            else:
                label = " "

            # Draw bounding box around the detected section with label
            cv2.rectangle(enhanced_image, (start_x, start_y), (end_x, end_y), (0, 0, 0), 2)
            cv2.putText(enhanced_image, label, (start_x + 5, start_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Print the indices of the red sections and section boundaries
print(f"Red sections: {red_sections}")

# Display the enhanced grayscale image
cv2.imshow("Enhanced Grayscale Image", enhanced_image)
cv2.imshow("Original", image)

cv2.waitKey(0)
