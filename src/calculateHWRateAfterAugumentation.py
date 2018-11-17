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
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)

            images, image_names, image_can_not_names, image_replace = load_and_align_data(args.image_dir, args.image_size, args.margin,
                                                      args.gpu_memory_fraction, args.replace_dir_1 , args.replace_dir_2)

            print (image_can_not_names)
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
                # for e in emb:
                #     for f in e:
                #         embeddingsTotal[index].append(f)
                #     index = index + 1
            embeddingsTotal = np.array(embeddingsTotal)
            # print ("len: ",len(embeddingsTotal[0]))
            if args.subtract_mean:
                mean = np.mean(embeddingsTotal, axis=0)
                embeddingsTotal = embeddingsTotal - mean

            count_first = 0
            count_first_five = 0
            totalCount = 0
            for i in range(nrof_images):
                name_i = image_names[i]
                matchObj1 = re.match(r'(.*)_(.*).jpg', name_i, re.M | re.I)
                # print (matchObj1.group(1),matchObj1.group(2))
                if matchObj1.group(2) != "2":
                    continue
                totalCount = totalCount + 1
                index_i = matchObj1.group(1)
                if name_i in image_can_not_names:
                    replace_list = image_replace[name_i]
                    hasReplace = False
                    if not (len(replace_list) == 0):
                        hasReplace = True
                    if hasReplace:
                        feed_dict = {images_placeholder: replace_list, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        first_list = {}
                        first_five_list = {}
                        lens = len(replace_list)
                        for i in range(lens):
                            max_in_min_dist = 0
                            min_index_list = []
                            min_dist = 10000
                            min_index = i
                            for j in range(nrof_images):
                                name_j = image_names[j]
                                matchObj2 = re.match(r'(.*)_(.*).jpg', name_j, re.M | re.I)
                                if matchObj2.group(2) != "1":
                                    continue
                                if args.distance_metric == 0:
                                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], embeddingsTotal[j, :]))))
                                else:
                                    dot = np.sum(np.multiply(emb[i, :], embeddingsTotal[j, :]))
                                    norm = np.linalg.norm(emb[i, :]) * np.linalg.norm(embeddingsTotal[j, :])
                                    similarity = dot / norm
                                    dist = np.arccos(similarity) / math.pi
                                if dist < min_dist and j != i:
                                    min_dist = dist
                                    min_index = j
                                if len(min_index_list) < 5:
                                    obj = {"index": j, "dist": dist}
                                    min_index_list.append(obj)
                                    if dist > max_in_min_dist:
                                        max_in_min_dist = dist
                                else:
                                    if dist < max_in_min_dist:
                                        for obj in min_index_list:
                                            if obj["dist"] == max_in_min_dist:
                                                min_index_list.remove(obj)
                                                max_in_min_dist = dist
                                                break
                                        min_index_list.append({"index": j, "dist": dist})
                                        for obj in min_index_list:
                                            if obj["dist"] > max_in_min_dist:
                                                max_in_min_dist = obj["dist"]
                            for obj in min_index_list:
                                min_index_sp = obj["index"]
                                name_j = image_names[min_index_sp]
                                if name_j in first_five_list:
                                    first_five_list[name_j] += 1
                                else:
                                    first_five_list[name_j] = 1
                            name_j = image_names[min_index]
                            if name_j in first_list:
                                first_list[name_j] += 1
                            else:
                                first_list[name_j] = 1

                        max_first_name = name_i
                        max_first_count = 0
                        for key in first_list:
                            c = first_list[key]
                            if c > max_first_count:
                                max_first_name = key
                                max_first_count = c
                        matchObj2 = re.match(r'(.*)_(.*).jpg', max_first_name, re.M | re.I)
                        index_j = matchObj2.group(1)
                        if index_i == index_j:
                            count_first = count_first + 1
                            if isInCanNotDetect(index_i) or isInCanNotDetect(index_j):
                                print("first match", name_i, max_first_name)

                        first_five_list = sorted(first_five_list.items(), key=lambda d: d[1],reverse=True)
                        # print (first_five_list)
                        # print ("i name:" ,name_i)
                        countForIndex = 0
                        for i in range(5):
                            tmp_name = first_five_list[i][0]
                            if countForIndex >= 5:
                                break
                            # print("first five name for:", name_i , tmp_name)
                            matchObj2 = re.match(r'(.*)_(.*).jpg', tmp_name, re.M | re.I)
                            index_j = matchObj2.group(1)
                            if index_i == index_j:
                                count_first_five = count_first_five + 1
                                if isInCanNotDetect(index_i) or isInCanNotDetect(index_j):
                                    print("first five match", name_i, tmp_name)
                                break
                            countForIndex += 1

                        # for max in range(5):
                        #     tmp_name = first_five_list[0][0]
                        #     matchObj2 = re.match(r'(.*)_(.*).jpg', tmp_name, re.M | re.I)
                        #     index_j = matchObj2.group(1)
                        #     if index_i == index_j:
                        #         count_first_five = count_first_five + 1
                        #         if isInCanNotDetect(index_i):
                        #             print("first five match", name_i, name_j)
                        #         break
                        continue
                max_in_min_dist = 0
                # min_index = i
                # min_index_list = []
                min_index_list = []

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
                    if len(min_index_list) < 5:
                        obj = {"index":j , "dist":dist}
                        min_index_list.append(obj)
                        if dist > max_in_min_dist:
                            max_in_min_dist = dist
                    else:
                        if dist < max_in_min_dist:
                            for obj in min_index_list:
                                if obj["dist"] == max_in_min_dist:
                                    min_index_list.remove(obj)
                                    max_in_min_dist = dist
                                    break
                            min_index_list.append({"index": j , "dist": dist})
                            for obj in min_index_list:
                                if obj["dist"] > max_in_min_dist:
                                    max_in_min_dist = obj["dist"]
                    # if dist < min_dist and j != i:
                    #     min_dist = dist
                    #     min_index = j
                for obj in min_index_list:
                    min_index_sp = obj["index"]
                    name_j = image_names[min_index_sp]
                    matchObj2 = re.match(r'(.*)_(.*).jpg', name_j, re.M | re.I)
                    index_j = matchObj2.group(1)
                    if index_i == index_j:
                        count_first_five = count_first_five + 1
                        if isInCanNotDetect(index_i) or isInCanNotDetect(index_j):
                            print("first five match", name_i, name_j)
                        break
                # print("first five match:")
                # print(name_i)
                # print(name_j)
                # name_j = image_names[min_index]
                # print("first match:")
                # print(name_i)
                # print(name_j)
                name_j = image_names[min_index]
                matchObj2 = re.match(r'(.*)_(.*).jpg', name_j, re.M | re.I)
                index_j = matchObj2.group(1)
                if index_i == index_j:
                    count_first = count_first + 1
                    if isInCanNotDetect(index_i) or isInCanNotDetect(index_j):
                        print("first match",name_i,name_j)
            # print(count_first_five)
            print("first rate:",count_first / (totalCount + 0.0))
            print("first five rate:",count_first_five/(totalCount + 0.0))

def load_and_align_data(image_dir, image_size, margin, gpu_memory_fraction, replace_dir_1, replace_dir_2):
    image_paths = []
    image_names = []
    image_can_not_names = []
    image_replace = {}
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
        index = tmp_image_paths.index(image)
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            name = image_names[index]
            print("can't detect face, but not remove ", image , name)
            image_can_not_names.append(name)
            matchObj1 = re.match(r'(.*)_(.*).jpg', name, re.M | re.I)
            if matchObj1.group(2) == '2':
                id = matchObj1.group(1)
                dir = replace_dir_2 + id + '/'
                replace_list = []
                for file in os.listdir(dir):
                    tmp_name = os.path.join(dir, file)
                    tmp_img = misc.imread(os.path.expanduser(tmp_name), mode='RGB')
                    prewhitened = facenet.prewhiten(tmp_img)
                    replace_list.append(prewhitened)
                image_replace[name] = replace_list
                # tmp_name = os.path.join(dir, file)
                # tmp_img = misc.imread(os.path.expanduser(tmp_name), mode='RGB')
                aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                img_list.append(prewhitened)
            else:
                id = matchObj1.group(1)
                dir = replace_dir_1 + id + '/'
                has_file = False
                for file in os.listdir(dir):
                    tmp_name = os.path.join(dir, file)
                    tmp_img = misc.imread(os.path.expanduser(tmp_name), mode='RGB')
                    aligned = misc.imresize(tmp_img, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)
                    has_file = True
                    break
                if not has_file:
                    aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)
        else:
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

            # 尝试人脸对齐
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
    return images,image_names,image_can_not_names,image_replace

def isInCanNotDetect(i):
    if i in ["265","1107","1701","1344","1869","257","552","183","1993","1052","522","1560","806","1394","389","895","1727","1431","94","1369","178","717","1788","1450","1626","986","1195","202","331","1396","1151","665","1877","996","350","1724","1381"]:
        return True
    return False

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_dir', type=str, help='dir of Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--distance_metric', type=int,
                        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--subtract_mean',
                        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--replace_dir_1', type=str,
                        help='a directory containing the replace file dir of *_1.jpg file')
    parser.add_argument('--replace_dir_2', type=str,
                        help='a directory containing the replace file dir of *_1.jpg file')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))