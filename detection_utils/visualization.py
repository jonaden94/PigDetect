import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2  # For reading the image file

def draw_bboxes(image_path, bboxes, scores, score_thresh, save_path=None, linewidth=2, show_scores=True, figsize=(15, 15), score_fontsize=6):
    """
    Draws bounding boxes on an image based on the provided score threshold and saves or displays the image with controlled size.
    
    Args:
        image_path (str): Path to the image file on which to draw.
        bboxes (list of list of float): List of bounding boxes, each specified as [x_min, y_min, x_max, y_max].
        scores (list of float): List of scores corresponding to each bounding box.
        score_thresh (float): Threshold to filter bboxes with associated scores below this value.
        save_path (str, optional): If provided, the image will be saved to this path.
        linewidth (int): Thickness of the bounding box edges.
        show_scores (bool): If True, display the scores on the top left corner inside each bounding box.
        figsize (tuple): Width and height of the figure in inches.
        score_fontsize (int): Font size of the score text.
    """
    # Load the image from the given path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Calculate the appropriate figure size to maintain the aspect ratio
    height, width, _ = image.shape
    aspect_ratio = width / height
    fig_width, fig_height = figsize
    fig_height = fig_width / aspect_ratio if aspect_ratio > 1 else fig_height
    fig_width = fig_height * aspect_ratio if aspect_ratio < 1 else fig_width

    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.imshow(image)

    # Loop through each bbox and score
    for bbox, score in zip(bboxes, scores):
        if score >= score_thresh:
            # Create a Rectangle patch with random color
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                     edgecolor=np.random.rand(3,), facecolor='none')
            ax.add_patch(rect)
            if show_scores:
                ax.text(bbox[0], bbox[1], f"{score:.2f}", color='white', fontsize=score_fontsize, ha='left', va='top',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    plt.axis('off')  # Turn off axis numbers and ticks

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Image saved to {save_path}")
    else:
        plt.show()

    plt.close()