import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background




if __name__ == "__main__":
    png = cv2.imread('media/pencil_small.png', cv2.IMREAD_UNCHANGED)
    # png = cv2.cvtColor(png, cv2.COLOR_RGB2BGR)
    cap = cv2.VideoCapture(0)
    while True:

        _, frame = cap.read()
        image = overlay_transparent(frame, png, 50, 50)
        cv2.imshow('f', image)

        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            break
