import cv2
import matplotlib.pyplot as plt


def draw_detected_cells(image_color, results):
    for center_x, center_y, radius, _ in results:
        cv2.circle(image_color, (center_x, center_y), radius, (0, 255, 0), 2)
    return image_color


def plot_image(image, num_cells):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.title(f"Matches after NMS: {num_cells}")
    plt.axis("off")
    plt.show()


def show_cells_on_image(color_image, found_cells):
    cells_on_image = draw_detected_cells(color_image, found_cells)
    plot_image(cells_on_image, len(found_cells))
