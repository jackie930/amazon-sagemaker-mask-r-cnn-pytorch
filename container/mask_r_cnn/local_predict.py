#!/usr/bin/env python3
import PIL
from helper import *
import cv2
import torch
import torchvision

def overlay_class_names(image, predictions, class_dict):
    """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
    scores = predictions["scores"].tolist()
    labels = predictions["labels"].tolist()
    a = dict(sorted(class_dict.items(), key=lambda item: item[1]))
    keys = list(a.keys())
    labels = [keys[int(i) - 1] for i in labels]
    boxes = predictions['boxes']
    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

    cv2.imwrite('./res.jpg', image)

    return image


def compute_colors_for_labels(labels, palette=None):
    """
        Simple function that adds fixed colors depending on the class
        """
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = labels[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors


def overlay_boxes(image, predictions):
    """
        Adds the predicted boxes on top of the image
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
    labels = predictions["labels"]
    boxes = predictions['boxes']
    colors = compute_colors_for_labels(labels).tolist()
    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    cv2.imwrite('./res_bbox.jpg', image)

    return image


def overlay_mask(image, predictions):
    """
        Adds the instances contours for each predicted object.
        Each label has a different color.
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
    masks = predictions["masks"].ge(0.5).mul(255).byte().numpy()
    labels = predictions["labels"]
    colors = compute_colors_for_labels(labels).tolist()
    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    cv2.imwrite('./res_mask.jpg', composite)

    return composite

def select_top_predictions(predictions, threshold):
    idx = (predictions["scores"] > threshold).nonzero().squeeze(1)
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions

def main(num_classes,model_path,img_path):
    model = get_model_instance_segmentation(num_classes)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    #predict
    image = PIL.Image.open(img_path)
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_tensor = image_tensor.to(device)
    output = model([image_tensor])
    top_predictions = select_top_predictions(output[0], 0.5)
    top_predictions = {k: v.cpu() for k, v in top_predictions.items()}
    #print result
    print ("precition: ", top_predictions)
    #visulize
    cv_img = np.array(image)
    result = overlay_boxes(cv_img, top_predictions)
    result = overlay_mask(result, top_predictions)



if __name__ == '__main__':
    main(51, './save_model/mask_rcnn_model_saved_test', '../../data/modelc/pic/10080.JPG', )




