from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import re
import math
import cv2

def main(args):
    images,image_names = load_and_align_data(args.image_dir, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)

            nrof_images = len(image_names)
            imagesPart = []
            embeddingsTotal = []
            # embeddingsTotal = [[] for i in range(nrof_images)]
            index = 0

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for i in range(nrof_images):
                imagesPart.append(images[i])
                if i % 50 == 49:
                    # Run forward pass to calculate embeddings
                    feed_dict = {images_placeholder: imagesPart, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    embeddingsTotal.extend(emb)
                    imagesPart = []
            if len(imagesPart) != 0:
                feed_dict = {images_placeholder: imagesPart, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)
                embeddingsTotal.extend(emb)
            embeddingsTotal = np.array(embeddingsTotal)
            if args.subtract_mean:
                mean = np.mean(embeddingsTotal, axis=0)
                embeddingsTotal = embeddingsTotal - mean

            count = 0
            totalCount = 0
            for i in range(nrof_images):
                name_i = image_names[i]
                matchObj1 = re.match(r'(.*)_(.*).jpg', name_i, re.M | re.I)
                # print (matchObj1.group(1),matchObj1.group(2))
                if matchObj1.group(2) != "2":
                    continue
                # print (matchObj1.group(2))
                totalCount = totalCount + 1
                index_i = matchObj1.group(1)
                min_dist = 10000
                min_index = i
                for j in range(nrof_images):
                    name_j = image_names[j]
                    matchObj2 = re.match(r'(.*)_(.*).jpg', name_j, re.M | re.I)
                    if matchObj2.group(2) != "1":
                        continue
                    if args.distance_metric == 0:
                        dist = np.sqrt(np.sum(np.square(np.subtract(embeddingsTotal[i, :], embeddingsTotal[j, :]))))
                    else:
                        dot = np.sum(np.multiply(embeddingsTotal[i, :], embeddingsTotal[j, :]))
                        norm = np.linalg.norm(embeddingsTotal[i, :]) * np.linalg.norm(embeddingsTotal[j, :])
                        similarity = dot / norm
                        dist = np.arccos(similarity) / math.pi
                    if dist < min_dist and j != i:
                        min_dist = dist
                        min_index = j
                name_j = image_names[min_index]
                print("match:")
                print(name_i)
                print(name_j)
                matchObj2 = re.match(r'(.*)_(.*).jpg', name_j, re.M | re.I)
                index_j = matchObj2.group(1)
                if index_i == index_j:
                    count = count + 1
            print(count)
            print(count/(totalCount + 0.0))

def load_and_align_data(image_dir, image_size, margin, gpu_memory_fraction):
    image_paths = []
    image_names = []
    # print(image_dir)
    for file in os.listdir(image_dir):
        # print(file)
        image_paths.append(os.path.join(image_dir, file))
        image_names.append(file)
    # for files in os.walk(image_dir):
    #     print(files)
    #     image_paths.append(files)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        # print(image)
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, but not remove ", image)
            cropped = img
        else:
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

            #尝试人脸对齐
            # print(_)
            eye_center = ((_[0][0] + _[1][0]) / 2, (_[5][0] + _[6][0]) / 2)
            dy = _[6][0] - _[5][0]
            dx = _[1][0] - _[0][0]
            angle = cv2.fastAtan2(dy, dx)
            rot = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
            cropped = cv2.warpAffine(cropped, rot, dsize=(cropped.shape[1], cropped.shape[0]))

        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images,image_names

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_dir', type=str, help='dir of Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    #32 pixels with an image size of 160 pixels corresponds to a margin of 44 pixels with an image size of 182, which is the image size that has been used for training of the model below.
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--distance_metric', type=int,
                        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--subtract_mean',
                        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))