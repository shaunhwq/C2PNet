import os

import cv2
import torch
import numpy as np

from models.C2PNet import C2PNet


def shortside_resize(image: np.array, min_size: int = 256):
    h, w, _ = image.shape
    is_landscape = w > h
    aspect_ratio = h / w if not is_landscape else w / h

    new_shape = [int(min_size), int(min_size * aspect_ratio)]
    if is_landscape:
        new_shape = new_shape[::-1]
    return cv2.resize(image, new_shape)


def pre_process(image: np.array, device: str, min_size: int = 256):
    image = shortside_resize(image, min_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = image.unsqueeze(0).contiguous().to(device)
    return image


def post_process(model_output, input_hw):
    image_rgb = model_output.cpu().squeeze(0).permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Resize to original
    h, w, c = image_bgr.shape
    in_h, in_w = input_hw
    if not (in_h == h and in_w == w):
        image_bgr = cv2.resize(image_bgr, input_hw[::-1])
    return image_bgr


if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/31_hazy_video.mp4"
    device = "mps"
    weights_path = "weights/OTS.pkl"

    model = C2PNet(gps=3, blocks=19)
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights["model"])
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)

    while True:
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        in_tensor = pre_process(frame, device)
        with torch.no_grad():
            print("inference")
            model_outputs = model(in_tensor)
            print("done")
        out_image = post_process(model_outputs, frame.shape[:2])

        display_image = np.vstack([frame, out_image])

        cv2.imshow("output", display_image)
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
