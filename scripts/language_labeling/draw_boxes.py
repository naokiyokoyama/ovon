import argparse
import json

import cv2


def mouse_callback(event, x, y, flags, param):
    global top_left_pt, bottom_right_pt, recording, draw, selecting

    if event == cv2.EVENT_LBUTTONDOWN:
        if selecting:
            top_left_pt = (x, y)
            recording = True
            draw = False
            selecting = False
        else:
            bottom_right_pt = (x, y)
            recording = False
            draw = True
            selecting = True


def get_corners(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2

    # Calculate the top left and bottom right coordinates
    top_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))

    return top_left, bottom_right


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Bounding Box Recording")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("output_json", type=str, help="Path to the output json")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    image_copy = image.copy()

    top_left_pt = ()
    bottom_right_pt = ()
    recording = False
    selecting = True

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    results = []

    while True:
        cv2.imshow("Image", image_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if not recording and top_left_pt and bottom_right_pt:
            cv2.rectangle(image_copy, top_left_pt, bottom_right_pt, (0, 0, 255), 2)
            top_left_pt, bottom_right_pt = get_corners(top_left_pt, bottom_right_pt)
            cv2.imshow("Image", image_copy)
            cv2.waitKey(1)
            class_name = input("Class name: ")
            word_size, _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
            cv2.putText(
                image_copy,
                class_name,
                (top_left_pt[0] + 4, top_left_pt[1] + word_size[1] + 4),
                font,
                font_scale,
                (0, 0, 255),
                font_thickness,
                cv2.LINE_AA,
            )
            cv2.imshow("Image", image_copy)
            results.append(
                {
                    "class": class_name,
                    "top_left": top_left_pt,
                    "bottom_right": bottom_right_pt,
                }
            )
            top_left_pt = ()
            bottom_right_pt = ()

    cv2.destroyAllWindows()

    with open(args.output_json, "w") as file:
        json.dump({"results": results}, file, indent=4)

    print("Results saved to:", args.output_json)
