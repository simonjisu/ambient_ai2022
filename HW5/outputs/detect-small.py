# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os
import time
import math
import json

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
import collections
from utils4smalls import tiles_location_gen, non_max_suppression, draw_object, reposition_bounding_box

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])


def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='./out.mp4')
    parser.add_argument('--length', type=int, default=3)

    parser.add_argument('--tile_sizes', required=True)
    parser.add_argument('--tile_overlap', type=int, default=15)
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--score_threshold', type=float, default=0.5)

    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera_idx)
        # Modifiy the resolution
        # Check the camera video resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)  
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        cap.set(cv2.CAP_PROP_FPS, 15)

    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) # frame rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # output video format arguments
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    frames = fps * args.length

    # tile settings
    img_size = (frame_height, frame_width)
    print(f'Image Size: {img_size}')
    
    ## tile size: e.g., 100x100, 30x30 -> [(100, 100), (30, 30)]
    tile_sizes = [
        list(map(int, tile_size.split('x'))) for tile_size in args.tile_sizes.split(',')
    ]
    for tile_size in tile_sizes:
        print(f'Number of Segmented: H={math.ceil(frame_height/tile_size[0])} W={math.ceil(frame_width/tile_size[1])}')
    records = collections.defaultdict()
    start = time.process_time()  # <------------------------------------------------------------------------- time
    records['program'] = [start]
    records['frames'] = []
    while cap.isOpened() and frames > 0:  # change condition
        frame_dict = {}   # record
        
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        objects_by_label = dict()

        tiles_in_one_frame = []
        for i, tile_size in enumerate(tile_sizes):
            tiles_dict = collections.defaultdict()
            tile_start = time.process_time()  # <------------------------------------------------------------------------- time
            tiles_dict['start'] = tile_start
            tiles_dict['sub'] = []
            for k, tile_location in enumerate(tiles_location_gen(img_size, list(tile_size), args.tile_overlap)):
                tiles_sub_dict = collections.defaultdict()
                tile_k_start = time.process_time()  # <------------------------------------------------------------------------- time
                tiles_sub_dict['start'] = tile_k_start
                # crop tile
                xmin, ymin, xmax, ymax = tile_location
                tile = cv2_im_rgb[xmin:xmax, ymin:ymax]
                # resize (inference size)
                resized_tile = cv2.resize(tile, inference_size)
                # calculate scale to roll back to origin image
                scale = min(resized_tile.shape[0]/tile.shape[0], resized_tile.shape[1]/tile.shape[1])
                
                tile_k_preprocessing = time.process_time()  # <------------------------------------------------------------------------- time
                tiles_sub_dict['end_preprocessing'] = tile_k_preprocessing
                # inference
                run_inference(interpreter, resized_tile.tobytes())
                
                tile_k_inference = time.process_time()  # <------------------------------------------------------------------------- time
                tiles_sub_dict['end_inference'] = tile_k_inference
                # get objects
                objs = get_objects(interpreter, args.score_threshold, (scale, scale))
                for obj in objs:
                    bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
                    bbox = reposition_bounding_box(bbox, tile_location)
                    label = labels.get(obj.id, '')
                    objects_by_label.setdefault(label, []).append(Object(label, obj.score, bbox))
                tile_k_get_objs = time.process_time()  # <------------------------------------------------------------------------- time
                tiles_sub_dict['end_get_object'] = tile_k_get_objs
                tiles_dict['sub'].append(tiles_sub_dict)
            tiles_in_one_frame.append(tiles_dict)
        frame_dict['tiles'] = tiles_in_one_frame
        # cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        # run_inference(interpreter, cv2_im_rgb.tobytes())
        # objs = get_objects(interpreter, args.threshold)[:args.top_k]
        # cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
        
        postprocessing_start = time.process_time()  # <------------------------------------------------------------------------- time
        frame_dict['postprocessing'] = [postprocessing_start]
        frame_dict['objects_by_label'] = objects_by_label
        frame_dict['objects'] = dict()
        for label, objects in objects_by_label.items():
            idxs = non_max_suppression(objects, args.iou_threshold)
            frame_dict['objects'][label] = []
            for idx in idxs:
                draw_object(cv2_im, objects[idx])
                frame_dict['objects'][label].append(objects[idx])
        postprocessing_end = time.process_time()  # <------------------------------------------------------------------------- time
        frame_dict['postprocessing'].append(postprocessing_end)
        # Write files
        out.write(frame)
        frames -= 1
        print(f'{frames} frames left')
        records['frames'].append(frame_dict)
        # cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    end = time.process_time()  # <------------------------------------------------------------------------- time
    records['program'].append(end)
    # 
    name = args.output.split('/')[-1].rstrip('.mp4')
    records_path = args.output.split('/')[:-1] + [f'{name}.json']
    records_path = '/' + os.path.join(*records_path)
    with open(records_path, 'w') as file:
        json.dump(records, file)

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()
