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

# Initialize the camera
cap = cv2.VideoCapture('rtsp://admin:Kfnfiffe12354@10.110.130.223/Streaming/Channels/101')  # 0 corresponds to the default camera, you can change it if you have multiple cameras
#cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Display the video stream
    cv2.imshow("Video Stream", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Check if 'p' is pressed to capture an image
    if key == ord('p'):
        # Convert the frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance the grayscale image using contrast stretching
        enhanced_frame = contrast_stretching(grayscale_frame, 0, 255)

        # (The rest of your code for detecting and highlighting red sections remains the same)

        # Define the list of percentages for variable section widths in both x and y axes
        percentages_x = [4, 6, 6, 6, 35, 8, 15, 10, 10]
        percentages_y = [14, 12.5, 12, 12, 12, 13, 12, 13.5]

        # Calculate the section widths based on percentages and image width and height
        section_widths_x = calculate_variable_widths(frame.shape[1], percentages_x)
        section_widths_y = calculate_variable_widths(frame.shape[0], percentages_y)

        # Set the threshold for average pixel value
        threshold = 180  # Adjust this value as needed

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
                section = enhanced_frame[start_y:end_y, start_x:end_x]

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
                    cv2.rectangle(enhanced_frame, (start_x, start_y), (end_x, end_y), (0, 0, 0), 2)
                    cv2.putText(enhanced_frame, label, (start_x + 5, start_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Print the indices of the red sections and section boundaries
        print(f"Red sections: {red_sections}")

        # Display the enhanced grayscale image
        cv2.imshow("Enhanced Grayscale Image", enhanced_frame)

    # Check if 'q' is pressed to exit the program
    elif key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
