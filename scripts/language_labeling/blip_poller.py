import argparse
import os
import os.path as osp
import time
import cv2
from PIL import Image

from blip_model import BLIP2

from add_text import add_text_to_image


def process_files(image_path: str, text_path: str, blip_model: BLIP2):
    try:
        img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)
        with open(text_path, "r") as f:
            prompt = f.read()

        pil_image = Image.fromarray(img_rgb)
        answer = blip_model.ask(pil_image, prompt)[0]

        response_name = image_path[:-3] + "response"
        with open(response_name, "w") as f:
            f.write(answer)
    except Exception as e:
        print("The following error occurred:")
        print(e)
        print("Continuing anyway.")

    # print("Prompt:")
    # print(prompt)
    # print(image_path)
    # print("Answer:")
    # print(answer)
    #
    # img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # img = add_text_to_image(img, "Prompt: " + prompt)
    # img = add_text_to_image(img, "Answer: " + answer)
    # filename = "visualizations/" + str(time.time()) + ".jpg"
    # cv2.imwrite(filename, img)


def monitor_directory(directory):
    ignored_files = set()

    # Ignore existing files in the directory on the first loop
    for filename in os.listdir(directory):
        ignored_files.add(filename)

    print("loading blip...")
    blip_model = BLIP2()

    while True:
        new_files = []

        for filename in os.listdir(directory):
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(directory, filename)
                text_path = os.path.splitext(image_path)[0] + ".txt"

                # Skip if the file is in the ignored_files set
                if filename in ignored_files:
                    continue

                # Skip if the text file doesn't exist
                if not os.path.exists(text_path):
                    ignored_files.add(filename)
                    continue

                new_files.append(filename)
                process_files(image_path, text_path, blip_model)

        # Add new files to the ignored_files set
        ignored_files.update(new_files)

        time.sleep(1)  # Sleep for 2 seconds before checking again


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor a directory for new image files."
    )
    parser.add_argument("directory", help="Path to the directory to monitor")
    args = parser.parse_args()

    monitor_directory(args.directory)
