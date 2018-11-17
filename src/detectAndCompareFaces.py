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
import math
import cv2

def main():
    g1 = tf.Graph()  # 加载到Session 1的graph
    g2 = tf.Graph()  # 加载到Session 2的graph
    sess1 = tf.Session(graph=g1)  # Session1
    sess2 = tf.Session(graph=g2)  # Session2

    global pnet, rnet, onet

    with sess1.as_default():
        with g1.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess1, None)

    images,image_names = load_and_align_data("data/images",160,32,0.9,sess1,pnet, rnet, onet)
    nrof_images = len(image_names)
    video_full_path = "rtsp://admin:thusun2017.3@192.168.2.222:554/h264/ch1/main"
    cap = cv2.VideoCapture(video_full_path)
    print(cap.isOpened())
    frame_count = 1
    success = True

    with sess2.as_default():
        with g2.as_default():
            # Load the model
            facenet.load_model("data/facenet/20180402-114759/20180402-114759.pb")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess2.run(embeddings, feed_dict=feed_dict)
            # mean = np.mean(emb, axis=0)

    count = 0
    while count < 50:
        count += 1
        success, frame = cap.read()
        # print('Read a new frame: ', success)
        if not success:
            continue
        with sess1.as_default():
            with g1.as_default():
                [canFind,imgs] = load_and_align_single_img(count,frame,160,32,0.9,sess1,pnet, rnet, onet)
                if not canFind:
                    continue
        # print("can find face")
        print("the ",count ,"th detect")

        with sess2.as_default():
            with g2.as_default():
                feed_dict = {images_placeholder: imgs, phase_train_placeholder: False}
                embs = sess2.run(embeddings, feed_dict=feed_dict)
                for emFind in embs:
                    minDist = 10000
                    minIndex = -1
                    for j in range(nrof_images):
                        # dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                        dot = np.sum(np.multiply(emFind , emb[j, :]))
                        norm = np.linalg.norm(emFind) * np.linalg.norm(emb[j, :])
                        similarity = dot / norm
                        # print (similarity)
                        dist = np.arccos(similarity) / math.pi
                        if dist < minDist:
                            minDist = dist
                            minIndex = j
                        print ("find person: ",image_names[j])

    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    #         # Load the model
    #         facenet.load_model("data/facenet/20180408-102900")

    cap.release()

def load_and_align_single_img(count ,img, image_size, margin, gpu_memory_fraction,sess,pnet, rnet, one):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    # print(bounding_boxes)
    # print("_ : ", _)
    if len(bounding_boxes) < 1:
        print("can't detect face, remove ")
        return [False,False]
    img_list = []
    nrof_faces = bounding_boxes.shape[0]
    det = bounding_boxes[:, 0:4]
    det_arr = []
    img_size = np.asarray(img.shape)[0:2]
    for i in range(nrof_faces):
        det_arr.append(np.squeeze(det[i]))
    for i, det in enumerate(det_arr):
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        # scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        output_filename_n = "{}{}_{}{}".format("data/faces/",count, i, ".jpg")
        misc.imsave(output_filename_n, cropped)
        eye_center = ((_[0][i] + _[1][i]) / 2, (_[5][i] + _[6][i]) / 2)
        dy = _[6][i] - _[5][i]
        dx = _[1][i] - _[0][i]
        angle = cv2.fastAtan2(dy, dx)
        rot = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        cropped = cv2.warpAffine(cropped, rot, dsize=(cropped.shape[1], cropped.shape[0]))
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return [True,images]


def load_and_align_data(image_dir, image_size, margin, gpu_memory_fraction,sess,pnet, rnet, onet):
    # print (image_paths)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # print('Creating networks and loading parameters')
    # with tf.Graph().as_default():
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #     with sess.as_default():
    #         pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    image_paths = []
    image_names = []
    for file in os.listdir(os.path.expanduser(image_dir)):
        image_paths.append(os.path.join(image_dir, file))
        image_names.append(file)

    tmp_image_paths = copy.copy(image_paths)
    print (tmp_image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        # print(bounding_boxes)
        # print("_ : ", _)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        eye_center = ((_[0] + _[1]) / 2, (_[5] + _[6]) / 2)
        dy = _[6] - _[5]
        dx = _[1] - _[0]
        angle = cv2.fastAtan2(dy, dx)
        rot = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        cropped = cv2.warpAffine(cropped, rot, dsize=(cropped.shape[1], cropped.shape[0]))
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images,image_names

if __name__ == '__main__':
    main()