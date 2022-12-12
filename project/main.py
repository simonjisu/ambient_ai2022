import streamlit as st
import cv2
import numpy as np
from pathlib import Path
# import tempfile
import mediapipe as mp
import time
from moviepy.editor import VideoFileClip
# https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2021-11-03--streamlit-opencv-mediapipe/2021-11-03/#face-landmark-detection

st.set_page_config(
    page_title='Rehabilitation Treatment Guide System',
    page_icon='🏋️'
)

MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_SPEC = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
MP_DRAWING_STYLE = mp.solutions.drawing_styles
MP_POSE = mp.solutions.pose

IMG_PADDING = 100
IDX = 0
VIDEO_PATH = Path('.').resolve() / 'videos'
TASKS = ['shoulder']
VIDEO_DICT = {
    'shoulder': '00_shoulder',
}
st.write('''
# Rehabilitation Treatment Guide System

Please, first look at the following short video, then start to train!

## Arguments

* `Inference frequency(IF)`: Inferecne with MediaPipe Model for every IF number of frame
* `Angle Threshold`: Threshold for allowance of error angle degree between USER and TARGET
* `Box Padding`: Number of padding pixels to start based on TARGET person bounding box
* `Left Arm`: Check if you want to train the left arm
* `Do not start`: If you check this box, you will always not start the training program. For Debugging.
''')
with st.sidebar:
    
    TASK = st.selectbox('choose task', options=TASKS, key='task')
    INFERENCE_FREQ = st.select_slider('Inference frequency(IF)', options=[1, 3, 5, 7, 10, 15], value=5, key='inference_freq')
    ANGLE_THRES = st.slider('Angle Threshold', 1.5, 10.0, value=3.5, step=0.1, key='angle_thres')
    BOX_PADDING = st.select_slider('Box Padding', options=[15, 30, 45], value=30, key='box_padding')
    LEFT_ARM = st.checkbox('Left Arm?', value=False, key='left_arm')
    DO_NOT_CHECKBOX = st.checkbox('Do not start', value=False, key='not_start')

# BGR
COLOR_DICT = {
    'red': (0, 0, 255),
    'green': (0, 255, 0)
}

def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle 

def in_box(box, coor):
    # since the image is flipped coor i
    logic = box[0] < coor[0] < box[2] and box[1] < coor[1] < box[3]
    return logic

def check_in_box(box, judge_points):
    if isinstance(judge_points, dict):
        judge_points = list(judge_points.values())
    res = []
    for coor in judge_points:
        res.append(int(in_box(box, coor)))
    return sum(res) == len(judge_points)

def process_mediapipe(image, seg=False):
    with MP_POSE.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
        model_complexity=2, 
        enable_segmentation=seg
    ) as POSE:
        results = POSE.process(image)
    return results

def extract_bbox(image, box_padding, clip_width, clip_height):

    results = process_mediapipe(image, seg=True)
    mask_img = (results.segmentation_mask > 0.5).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask_img)
    bbox = (
        max(0, x-box_padding),  # width_left
        max(0, y-box_padding),  # height_top
        min(clip_width, x+w+box_padding),  # width_right
        min(clip_height, y+h+box_padding)  # height_bottom
    )
    return bbox

def extract_landmarks(image, rt_image=False):
    img_w, img_h, _ = image.shape
    results = process_mediapipe(image, seg=False)
    
    angles = np.zeros((1, 2))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Get coordinates
        left_shoulder = [
            landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].x*img_h, 
            landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y*img_w
        ]
        left_elbow = [
            landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].x*img_h,
            landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].y*img_w
        ]
        left_knee = [
            landmarks[MP_POSE.PoseLandmark.LEFT_KNEE.value].x*img_h,
            landmarks[MP_POSE.PoseLandmark.LEFT_KNEE.value].y*img_w
        ]
        left_hip = [
            landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].x*img_h,
            landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].y*img_w
        ]

        right_shoulder = [
            landmarks[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value].x*img_h,
            landmarks[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value].y*img_w
        ]
        right_elbow = [
            landmarks[MP_POSE.PoseLandmark.RIGHT_ELBOW.value].x*img_h,
            landmarks[MP_POSE.PoseLandmark.RIGHT_ELBOW.value].y*img_w
        ]
        right_knee = [
            landmarks[MP_POSE.PoseLandmark.RIGHT_KNEE.value].x*img_h,
            landmarks[MP_POSE.PoseLandmark.RIGHT_KNEE.value].y*img_w
        ]
        right_hip = [
            landmarks[MP_POSE.PoseLandmark.RIGHT_HIP.value].x*img_h,
            landmarks[MP_POSE.PoseLandmark.RIGHT_HIP.value].y*img_w
        ]
    else:
        left_shoulder, left_elbow, left_knee, left_hip = ((-1, -1),(-1, -1),(-1, -1),(-1, -1))
        right_shoulder, right_elbow, right_knee, right_hip = ((-1, -1),(-1, -1),(-1, -1),(-1, -1))
    
    angles[0, 0] = calculate_angle(left_elbow, left_shoulder, left_hip) #angle_1
    angles[0, 1] = calculate_angle(right_elbow, right_shoulder, right_hip)
    
    if LEFT_ARM:
        judge_points = {
            'left_shoulder': left_shoulder, 'left_elbow': left_elbow, 'left_knee': left_knee, 'left_hip': left_hip
        }
    else:
        judge_points = {
            'right_shoulder': right_shoulder, 'right_elbow': right_elbow, 'right_knee': right_knee, # 'right_hip': right_hip
        }
    
    if rt_image:
        MP_DRAWING.draw_landmarks(
            image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
            MP_DRAWING.DrawingSpec(color=tuple(reversed((245,117,66))), thickness=2, circle_radius=2), 
            MP_DRAWING.DrawingSpec(color=tuple(reversed((245,66,230))), thickness=2, circle_radius=2)
        )
    return angles, judge_points, image

@st.cache(allow_output_mutation=True)
def get_cemara_information(crop_video_path, box_padding):
    clip = VideoFileClip(crop_video_path).without_audio()
    # Generate bbox
    frame = clip.get_frame(0)
    clip_height, clip_width, _ = frame.shape
    first_image = frame.copy()
    if LEFT_ARM:
        first_image = cv2.flip(first_image, 1)
    video = cv2.VideoCapture(0)
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    if video_width > clip_width:
        crop_width = int((video_width - clip_width) / 2)
    else:
        crop_width = 0
    if video_height > clip_height:
        crop_height = int((video_height - clip_height) / 2)
    else:
        crop_height = 0

    infos = {
        'video_width': video_width, 'video_height': video_height, 'fps_input': fps_input,
        'clip_width': clip_width, 'clip_height': clip_height,
        'crop_width': crop_width, 'crop_height': crop_height
    }
    box = extract_bbox(first_image, 
        box_padding=box_padding, clip_width=clip_width, clip_height=clip_height)
    return infos, box

def crop_image(frame, infos):
    frame = frame[infos['crop_height']:(infos['video_height']-infos['crop_height']), 
        infos['crop_width']:(infos['video_width']-infos['crop_width'])].copy()
    return frame

fps = 0
i = 0

START_SET = st.sidebar.button('START A SET')
RESTART_SET = st.sidebar.button('RESTART')
fps_text = st.sidebar.markdown('')

INFOS, BOX = get_cemara_information(
    crop_video_path=str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_crop_720p.mp4')),
    box_padding=BOX_PADDING
)
FRAMES_THRES = 10

status_board = st.sidebar.markdown('')

if START_SET:
    col_left, col_right = st.columns([1, 1])
    angle_text = st.markdown('')
    FIT_BOX = False

    with col_left:
        st.header('USER')
        stframe = st.empty()
    
    with col_right:
        st.header('TARGET')
        gold_stframe = st.empty()

    prevTime = 0
    is_in_box_count = [0]*FRAMES_THRES # queue

    while not FIT_BOX:
        video = cv2.VideoCapture(0)
        while video.isOpened():
            i +=1
            ret, frame = video.read()
            # control size
            frame = cv2.flip(frame, 1)
            frame = crop_image(frame, infos=INFOS)
            angles, judge_points, frame = extract_landmarks(image=frame, rt_image=DO_NOT_CHECKBOX)
            
            is_in_box = check_in_box(BOX, judge_points)
            # record on status board
            status_board.write(f"<h2 style='color:red;'>[Frame Index {i}] Angle={angles[0, int(LEFT_ARM)]:.4f} </h2>", 
                unsafe_allow_html=True)

            currTime = time.time()
            fps = 1/(currTime - prevTime)
            fps_text.write(f"<h1 style='text-align: center; color:red;'> FPS: {int(fps)}</h1>", unsafe_allow_html=True)
            prevTime = currTime
            if not DO_NOT_CHECKBOX:
                color = COLOR_DICT['green'] if is_in_box else COLOR_DICT['red']
                cv2.rectangle(frame, BOX[:2], BOX[2:], color, 2)
            stframe.image(frame, channels='BGR', use_column_width=True)
            if not DO_NOT_CHECKBOX:
                is_in_box_count.append(int(is_in_box))
                if sum(is_in_box_count) == FRAMES_THRES:
                    break
                else:
                    is_in_box_count.pop(0)

        FIT_BOX = True
        prevTime = 0
        is_in_box_count = [0]*FRAMES_THRES
        
        print('is fit')

    wait_video = cv2.VideoCapture(str(VIDEO_PATH / 'intermediate.mp4'))
    gold_video = cv2.VideoCapture(str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_crop_720p.mp4')))
    
    while FIT_BOX:
        while wait_video.isOpened() and video.isOpened():
            # left column
            ret, frame = video.read()
            frame = cv2.flip(frame, 1)
            frame = crop_image(frame, infos=INFOS)
            color = COLOR_DICT['green'] if FIT_BOX else COLOR_DICT['red']
            cv2.rectangle(frame, BOX[:2], BOX[2:], color, 2)
            stframe.image(frame, channels='BGR', use_column_width=True)

            # right column
            w_ret, w_frame = wait_video.read()
            if not w_ret:
                wait_video.release()
                break
            gold_stframe.image(w_frame, channels='BGR', use_column_width=True)
            
            # fps
            currTime = time.time()
            fps = 1/(currTime - prevTime)
            fps_text.write(f"<h1 style='text-align: center; color:red;'> FPS: {int(fps)}</h1>", unsafe_allow_html=True)
            prevTime = currTime
        
        inference_idx = 0
        count = 0
        n_count = 0
        while gold_video.isOpened() and video.isOpened():
            # left
            ret, frame = video.read()
            frame = cv2.flip(frame, 1)
            frame = crop_image(frame, infos=INFOS)
            # right
            g_ret, g_frame = gold_video.read()
            if not g_ret:
                gold_video.release()
                break

            if inference_idx % INFERENCE_FREQ == 0:
                # left inference
                angles, _, frame = extract_landmarks(image=frame, rt_image=True)
                
                # right inference
                gold_angles, _, g_frame = extract_landmarks(image=g_frame, rt_image=True)
                # calculate angle
                corrects = np.abs(angles - gold_angles) < ANGLE_THRES
                if LEFT_ARM:
                    correct = corrects[0, 0]
                else:
                    correct = corrects[0, 1]
                count += int(correct)
                n_count += 1
                
                # fps
                currTime = time.time()
                fps = 1/(currTime - prevTime)
                fps_text.write(f"<h1 style='text-align: center; color:red;'> FPS: {int(fps)}</h1>", unsafe_allow_html=True)
                prevTime = currTime

                color = COLOR_DICT['green'] if is_in_box else COLOR_DICT['red']
                cv2.rectangle(frame, BOX[:2], BOX[2:], color, 2)
                stframe.image(frame, channels='BGR', use_column_width=True)
                gold_stframe.image(g_frame, channels='BGR', use_column_width=True)

                # record on status board
                status_board.write(f"<h2 style='color:red;'>[Frame Index {inference_idx}] \
                    Angle(elbow, shoulder, hip) match? {correct}<br>Left: {angles[0, int(LEFT_ARM)]:.4f} Right: {gold_angles[0, int(LEFT_ARM)]:.4f}  </h2>", 
                unsafe_allow_html=True)
            
            inference_idx += 1
            # accuracy
            acc = f'{(count / n_count):.4f}'
            angle_text.write(f"<h2 style='text-align: center; color:red;'> Count/Frame: {count} | Match Accuracy: {acc}</h2>", unsafe_allow_html=True)

        FIT_BOX = False

    if RESTART_SET:
        wait_video = cv2.VideoCapture(str(VIDEO_PATH / 'intermediate.mp4'))
        gold_video = cv2.VideoCapture(str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_crop_720p.mp4')))
        FIT_BOX = True
    
else:
    video_file = str(VIDEO_PATH / (VIDEO_DICT[TASK] + '.mp4'))
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.video(video_file, format="video/mp4", start_time=0)