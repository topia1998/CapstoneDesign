import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import skimage
from skimage import io, transform

def video_partition():
    count = []
    for i in range(1, 100):
        filepath = 'C:\\Users\\topia1998\\%d.mp4' % i
        if os.path.isfile(filepath):
            video = cv2.VideoCapture(filepath)

            if not video.isOpened():
                print("Could not Open :", filepath)
                exit(0)

            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)

            cnt = 0

            while(video.isOpened()):
                if (cnt + 1) * (int)(fps) > length:
                    break 
                ret, image = video.read()
                if(int(video.get(1)) % (int)(fps) == 0):
                    createFolder("C:\\Users\\topia1998\\video%d" % i)
                    cv2.imwrite("C:\\Users\\topia1998\\video%d\\frame%d.png" % (i, cnt), image)

                    cnt += 1
            video.release()
        else:
            break
        count.append(cnt)

    return count, i - 1

def load_image(image_path):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    img = skimage.img_as_float(io.imread(image_path))
    if len(img.shape) == 2:
        img = np.array([img, img, img]).swapaxes(0, 2)
    return img

def rescale(img, input_height, input_width):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    aspect = img.shape[1] / float(img.shape[0])
    if (aspect > 1):
    # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = transform.resize(img, (input_width, res))
    if (aspect < 1):
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = transform.resize(img, (res, input_height))
    if (aspect == 1):
        imgScaled = transform.resize(img, (input_width, input_height))
    return imgScaled

def crop_center(img, cropx, cropy):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

def normalize(img, mean=128, std=128):
    img = (img * 256 - mean) / std
    return img

def prepare(img_uri):
        img = load_image(img_uri)
        img = rescale(img, 300, 300)
        img = transform.resize(img, (300, 300))
        img = normalize(img)
        return img


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error:Creating directory.' + directory)

def ssd():
    k, v = video_partition()
    import torch
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    ssd_model.to('cuda')
    ssd_model.eval()

    target = cv2.imread("/content/target.png")

    for j in range(1, v + 1):
        count = 1
        uris = list()
        for i in range(k[j - 1]):
            uris.append('/content/video%d/frame%d.png' % (j, i))

        inputs = [prepare(uri) for uri in uris]
        tensor = utils.prepare_tensor(inputs)

        with torch.no_grad():
            detections_batch = ssd_model(tensor)

        results_per_input = utils.decode_results(detections_batch)
        best_results_per_input = [utils.pick_best(results, 0.6) for results in results_per_input]
        classes_to_labels = utils.get_coco_object_dictionary()
        createFolder("/content/result%d" % j)
        cv2.imwrite("/content/result%d/target.png" % j, target)
        for image_idx in range(len(best_results_per_input)):
            tmp = plt.imread('/content/video%d/frame%d.png' % (j, image_idx))
            image = inputs[image_idx] / 2 + 0.5
            bboxes, classes, confidences = best_results_per_input[image_idx]
            for idx in range(len(bboxes)):
                left, bot, right, top = bboxes[idx]
                x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
                if classes_to_labels[classes[idx] - 1] == 'person':
                    tmp = rescale(tmp, 300, 300)
                    tmp = transform.resize(tmp, (300, 300))
                    if y < 0:
                        y = 0
                    if x < 0:
                        x = 0
                    cropped = tmp[int(y):int(y+h), int(x):int(x+w)]
                    plt.imsave("/content/result%d/%d.png" % (j, count), cropped)
                    count += 1

ssd()