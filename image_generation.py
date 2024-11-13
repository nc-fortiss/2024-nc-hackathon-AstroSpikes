import numpy as np
import cv2

# Parameters
img_size = 240
square_size = 20
num_frames = 10
output_dir = 'event_frames'

# Create output directory if needed
import os
import csv
os.makedirs(output_dir, exist_ok=True)

# Initial position of the square
x_position = 0
y_position = img_size // 2 - square_size // 2

offset = 10

# Create CSV writer
csv_file = open(os.path.join(output_dir, 'positions.csv'), mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Frame sequence generation
for i in range(num_frames):
    # Create two-channel frame: one for positive (red) and one for negative (blue) changes
    frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Determine the new and old positions
    new_x_position = int(i * (img_size - square_size) / (num_frames - 1))
    
    # Draw the positive change (red) in the new position
    print(new_x_position+square_size-offset)
    frame[y_position:y_position + square_size, new_x_position-offset:new_x_position, 0] = 255  # Blue channel
    
    # Draw the negative change (blue) in the old position if not the first frame
    if i > 0:
        frame[y_position:y_position + square_size, new_x_position+square_size-offset:new_x_position + square_size, 2] = 255  # Red channel
        pass
    
    # Update the previous position to the new one for the next frame
    x_position = new_x_position
    
    # Save the frame as an image
    cv2.imwrite(f"{output_dir}/frame_{i:02d}.png", frame)
    print(f"Saved frame {i} at position {x_position}")
    # Save the x and y coordinates
    csv_writer.writerow([x_position, y_position])

print("All frames generated and saved.")
