from flask import request, Flask
from flask_cors import CORS

app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)

# Setup configuration file
import configparser

config = configparser.ConfigParser()
config.read('api_config.ini')

# Setup logs test
import logging

logging.basicConfig(filename=config['PATH']['LOGS'], level=logging.INFO,
                    format='%(asctime)s: %(levelname)s: %(message)s ')
logging.info('----------Started----------')

from elastic import update_doc, search_id

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
from json2xml import readfromstring, json2xml
import json
import re
import xml.etree.ElementTree as xml

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
# MODEL_NAME = 'faster_rcnn_nas_lowproposals_coco_2018_01_28'
MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28'
# MODEL_NAME = 'faster_rcnn_resnet101_lowproposals_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# ## Download Model

# In[5]:

'''
logging.info('Downloading'+ MODEL_FILE +'Model...')
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
logging.info('Downloaded')
'''


# ## Helper code

# In[9]:


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# TEST_IMAGE_FOLDER = config['PATH']['TEST_IMAGE_FOLDER']
# IMAGES_FOLDER_NAME = os.path.basename(TEST_IMAGE_FOLDER)
# TEST_IMAGE_PATHS = next(os.walk(TEST_IMAGE_FOLDER))[2]
# TEST_IMAGE_PATHS.sort()
# TEST_IMAGE_PATHS
OUTPUT_DIR_XML = config['PATH']['OUTPUT_DIR_XML']
OUTPUT_DIR_JSON = config['PATH']['OUTPUT_DIR_JSON']
ANNOTATED_IMAGE_DIR = config['PATH']['ANNOTATED_IMAGES']


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def save_json_xml(TEST_IMAGE_FOLDER, IMAGES_FOLDER_NAME, folder, category_index, image_path, output_dict):
    img = cv2.imread(os.path.join(TEST_IMAGE_FOLDER, image_path))
    image_height, image_width, depth = img.shape
    out_dict = {'folder': IMAGES_FOLDER_NAME,
                'filename': image_path,
                'path': os.path.join(TEST_IMAGE_FOLDER, image_path),
                'size': {'width': image_width, 'height': image_height, 'depth': depth}}
    for index, score in enumerate(output_dict['detection_scores']):
        if score < 0.3:
            continue
        label = category_index[output_dict['detection_classes'][index]]['name']
        ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]
        out_dict.update({'object' + str(index + 1): {'name': label,
                                                     'bndbox': {'xmin': int(xmin * image_width),
                                                                'ymin': int(ymin * image_height),
                                                                'xmax': int(xmax * image_width),
                                                                'ymax': int(ymax * image_height)}}
                         })
    data = readfromstring(json.dumps(out_dict))
    s = json2xml.Json2xml(data, wrapper="annotation").to_xml()
    s = re.sub("object\d+", 'object', s)
    fname = os.path.splitext(image_path)[0]

    OUTPUT_FOLDER_XML = os.path.join(folder, OUTPUT_DIR_XML)
    OUTPUT_FOLDER_JSON = os.path.join(folder, OUTPUT_DIR_JSON)

    # create dir for storing xml
    if not os.path.exists(OUTPUT_FOLDER_XML):
        os.makedirs(OUTPUT_FOLDER_XML)

    # create dir for storing json
    if not os.path.exists(OUTPUT_FOLDER_JSON):
        os.makedirs(OUTPUT_FOLDER_JSON)

    with open(os.path.join(OUTPUT_FOLDER_XML, fname + '.xml'), 'w') as output:
        output.write(s)
        output.close()
    with open(os.path.join(OUTPUT_FOLDER_JSON, fname + '.json'), 'w') as out:
        out.write(str(output_dict))
        out.close()


@app.route("/rosbag/image_annotate", endpoint="image_annotate", methods=['POST'])
def image_annotate():
    req_data = request.data
    req_data = req_data.decode("utf-8")
    req_data = json.loads(req_data)

    doc_id = req_data['doc_id']

    # search for existing data
    info_data = search_id(doc_id)
    info_dict = info_data['hits'][0]

    # remove doc_id from dictionary
    info_dict.pop('doc_id')

    TEST_IMAGE_FOLDER = req_data["image_dir"]
    IMAGES_FOLDER_NAME = os.path.basename(TEST_IMAGE_FOLDER)
    TEST_IMAGE_PATHS = next(os.walk(TEST_IMAGE_FOLDER))[2]
    TEST_IMAGE_PATHS.sort()

    # fetch bag_file_name
    folder = os.path.splitext(info_dict['path'])[0]

    # ## Load a (frozen) Tensorflow model into memory.

    # In[6]:

    logging.info('Load a (frozen) Tensorflow model into memory')
    model_load_start = time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    model_load_end = time.time()
    logging.info('Done')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    # In[7]:

    logging.info('Loading  label map')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    logging.info('Done')

    # # Detection

    object_detect_time = {}
    for image_path in TEST_IMAGE_PATHS:
        try:
            logging.info('Loading image into memory')
            image = Image.open(os.path.join(TEST_IMAGE_FOLDER, image_path))
            logging.info('Loaded')

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            image.close()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            logging.info('Expand dimensions since the model expects images to have shape: [1, None, None, 3]')
            image_np_expanded = np.expand_dims(image_np, axis=0)
            logging.info('Done')

            # Actual detection.
            logging.info('**********Actual Detection**********')
            object_detect_start = time.time()
            output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            object_detect_end = time.time()
            object_detect_time[image_path] = str(round(object_detect_end - object_detect_start, 2)) + 's'
            logging.info('**********Done**********')

            # Create json and xml files
            logging.info('Create json and xml files')
            save_json_xml(TEST_IMAGE_FOLDER, IMAGES_FOLDER_NAME, folder, category_index, image_path, output_dict)
            logging.info('Done')

            # Visualization of the results of a detection.
            logging.info('Visualization of the results of a detection')
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=5)
            plt.figure()

            ANNOTATED_IMAGE_FOLDER = os.path.join(folder, ANNOTATED_IMAGE_DIR)

            # create dir for storing annotated images
            if not os.path.exists(ANNOTATED_IMAGE_FOLDER):
                os.makedirs(ANNOTATED_IMAGE_FOLDER)

            plt.imsave(os.path.join(ANNOTATED_IMAGE_FOLDER, image_path), image_np)
            logging.info('**********Saved Image**********')

        except Exception as e:
            logging.log(level=logging.ERROR, msg=str(e))
            continue

    logging.info('--------------------SUMMARY--------------------')
    logging.info('Time taken to load model = ' + str(round(model_load_end - model_load_start, 2)) + 's')
    logging.info('Total number of images = ' + str(len(TEST_IMAGE_PATHS)))
    logging.info('Number of images successfully processed = ' + str(len(object_detect_time)))
    logging.info('Number of images left unprocessed = ' + str(len(TEST_IMAGE_PATHS) - len(object_detect_time)))
    logging.info('Time taken per image for object detection = ' + json.dumps(object_detect_time))
    # logging.info(f'Time taken to load model = {(model_load_end - model_load_start):.2f}')
    logging.info('----------Stopped----------')

    result = {
        "annotation": {
            "annotated_images_dir": os.path.join(folder, ANNOTATED_IMAGE_DIR),
            "logs_dir": config['PATH']['LOGS'],
            "xml_dir": os.path.join(folder, OUTPUT_DIR_XML),
            "json_dir": os.path.join(folder, OUTPUT_DIR_JSON)
        }
    }

    info_dict.update(result)
    response = update_doc(doc_id, info_dict)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
