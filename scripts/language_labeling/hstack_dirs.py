import os
import cv2
import numpy as np
import argparse

def pad_image(image, target_height):
    height, width, _ = image.shape
    if height < target_height:
        padding = np.ones((target_height - height, width, 3), dtype=np.uint8) * 255
        image = np.vstack((image, padding))
    return image

def sort_and_save_images(dir1, dir2, output_dir):
    # Get the list of .jpg files in the directories
    files1 = sorted([file for file in os.listdir(dir1) if file.endswith('.jpg')])
    files2 = sorted([file for file in os.listdir(dir2) if file.endswith('.jpg')])

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for file1, file2 in zip(files1, files2):
        # Read the images
        image1 = cv2.imread(os.path.join(dir1, file1))
        image2 = cv2.imread(os.path.join(dir2, file2))

        # Get the maximum height among the images
        max_height = max(image1.shape[0], image2.shape[0])

        # Pad the shorter image to match the height of the taller image
        image1 = pad_image(image1, max_height)
        image2 = pad_image(image2, max_height)

        # Concatenate the images horizontally
        combined_image = np.hstack((image1, image2))

        # Save the combined image to the output directory
        output_path = os.path.join(output_dir, f"combined_{file1}")
        cv2.imwrite(output_path, combined_image)

        print(f"Combined image saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sort and combine .jpg images from two directories.')
    parser.add_argument('dir1', type=str, help='Path to the first directory')
    parser.add_argument('dir2', type=str, help='Path to the second directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()

    sort_and_save_images(args.dir1, args.dir2, args.output_dir)
