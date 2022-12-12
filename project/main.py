import av
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pathlib import Path
# import tempfile
import mediapipe as mp
import time
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, TextClip, VideoClip, concatenate_videoclips, clips_array
# https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2021-11-03--streamlit-opencv-mediapipe/2021-11-03/#face-landmark-detection

MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_SPEC = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
MP_DRAWING_STYLE = mp.solutions.drawing_styles
MP_POSE = mp.solutions.pose

BOX_THRES = 30
IMG_PADDING = 100
IDX = 0
VIDEO_PATH = Path('.').resolve() / 'videos'
TASKS = ['shoulder']
VIDEO_DICT = {
    'shoulder': '00_shoulder',
}

with st.sidebar:
    TASK = st.selectbox('choose task', options=TASKS)
#     FPS = st.select_slider('Frame Per Seconds', options=[5, 15, 30], value=15)
    ANGLE_THRES = st.slider('Angle Threshold', 0.0, 5.0, value=0.5, step=0.1)
    ACC_THRES = st.slider('Accuracy Threshold', 50.0, 100.0, value=50.0, step=1.0)

# VIDEO_SPEC = {
#     'video': {
#         'width': {'ideal': 720, 'min': 720}, # add 100 px for left right padding
#         'height': {'ideal': 1280, 'min': 1280},
#         'frameRate': {'ideal': FPS, 'max': FPS}
#     },
#     'audio': False,
# }
# BGR
COLOR_DICT = {
    'red': (0, 0, 255),
    'green': (0, 255, 0)
}

@st.cache()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)

    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv2.resize(image,dim, interpolation=inter)

    return resized

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

# width_left, height_top, width_right, height_bottom
# BOXES_DICT = get_box_dict()
# ANGLE_DATA = np.loadtxt(str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_angles.txt')), 
#     dtype='str', comments='#', delimiter=',', skiprows=0, encoding='bytes')[:, 1:-2].astype(np.float64)

# def video_frame_callback(frame):
#     # https://github.com/whitphx/streamlit-webrtc
#     global START
#     global TASK
#     global IDX
    
#     image = frame.to_ndarray(format="bgr24")
#     img_w, img_h, _ = image.shape
#     results, angles, judge_points = process_mediapipe(image, img_w, img_h, seg=False)
#     # width_left, height_top, width_right, height_bottom
#     box = BOXES_DICT[TASK]
#     color = COLOR_DICT['green'] if START else COLOR_DICT['red']
#     cv2.rectangle(image, box[:2], box[2:], color, 2)
    
#     MP_DRAWING.draw_landmarks(
#         image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
#         MP_DRAWING.DrawingSpec(color=tuple(reversed((245, 117, 66))), thickness=2, circle_radius=2), 
#         MP_DRAWING.DrawingSpec(color=tuple(reversed((245, 66, 230))), thickness=2, circle_radius=2)
#     )

#     if judge_points is not None and check_in_box(box, judge_points):
#         START = True
#         print(START)
#         IDX += 1
#     else:
#         START = False
#         IDX = 0
#     # START TO RUN
#     if START:
#         # MP_DRAWING.draw_landmarks(
#         #     image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
#         #     MP_DRAWING.DrawingSpec(color=tuple(reversed((245,117,66))), thickness=2, circle_radius=2), 
#         #     MP_DRAWING.DrawingSpec(color=tuple(reversed((245,66,230))), thickness=2, circle_radius=2)
#         # )
#         # calculate the angle
#         diff = np.abs(ANGLE_DATA[IDX, :] - angles)
#         num_correct = (diff < ANGLE_THRES).sum()
#         print(f'[{IDX}] {num_correct}')
#         # Render detections
        
#     return av.VideoFrame.from_ndarray(image, format="bgr24")

# col_user, col_gold = st.columns([2, 1])

# st.write(ANGLE_DATA)
# st.write(ANGLE_DATA.shape, ANGLE_DATA.shape[0] / 30)
# webrtc_streamer(
#     key='task', 
#     video_frame_callback=video_frame_callback,
#     media_stream_constraints=VIDEO_SPEC
# )

# clip = VideoFileClip(str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_crop_720p.mp4'))).without_audio()
# # bbox = (291, 390, 529, 1280)
# frame = clip.get_frame(0)
# clip_height, clip_width, _ = frame.shape

# # Get Bounding Box
# BOX_THRES = 30
# with MP_POSE.Pose(
#         min_detection_confidence=0.5, 
#         min_tracking_confidence=0.5, 
#         model_complexity=2, 
#         enable_segmentation=True
#     ) as POSE:

#     results = POSE.process(frame)
# mask_img = (results.segmentation_mask > 0.5).astype(np.uint8)
# x, y, w, h = cv2.boundingRect(mask_img)
# bbox = (
#     max(0, x-BOX_THRES),  # width_left
#     max(0, y-BOX_THRES),  # height_top
#     min(clip_width, x+w+BOX_THRES),  # width_right
#     min(clip_height, y+h+BOX_THRES)  # height_bottom
# )



# with col_left:
#     stframe = st.empty()
    # temp_file = tempfile.NamedTemporaryFile(delete=False)
    # video = cv2.VideoCapture(0)
    # video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps_input = int(video.get(cv2.CAP_PROP_FPS))
    # crop_width = int((video_width - clip_width) / 2)
    # crop_height = int((video_height - clip_height) / 2)

# st.sidebar.write(f'[Video] width={video_width}, height={video_height}, fps={fps_input}')
# st.sidebar.write(f'[Clip] width={clip_width}, height={clip_height}, crop={crop_width, crop_height}')
# codec = cv2.VideoWriter_fourcc('a','v','c','1')
# out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width,height))


def process_mediapipe(image, seg=False):
    with MP_POSE.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
        model_complexity=2, 
        enable_segmentation=seg
    ) as POSE:
        results = POSE.process(image)
    return results

def extract_bbox(image):
    global BOX_THRES
    results = process_mediapipe(image, seg=True)
    mask_img = (results.segmentation_mask > 0.5).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask_img)
    bbox = (
        max(0, x-BOX_THRES),  # width_left
        max(0, y-BOX_THRES),  # height_top
        min(clip_width, x+w+BOX_THRES),  # width_right
        min(clip_height, y+h+BOX_THRES)  # height_bottom
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
    angles[0, 1]  = calculate_angle(right_elbow, right_shoulder, right_hip)
    judge_points = {
        'right_shoulder': right_shoulder, 'right_elbow': right_elbow, # 'right_knee': right_knee, # 'right_hip': right_hip
    }
    # judge_points = {
    #     'left_shoulder': left_shoulder, 'left_elbow': left_elbow, 'left_knee': left_knee, 'left_hip': left_hip
    # }
    if rt_image:
        MP_DRAWING.draw_landmarks(
            image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
            MP_DRAWING.DrawingSpec(color=tuple(reversed((245,117,66))), thickness=2, circle_radius=2), 
            MP_DRAWING.DrawingSpec(color=tuple(reversed((245,66,230))), thickness=2, circle_radius=2)
        )
    return angles, judge_points, image

clip = VideoFileClip(str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_crop_720p.mp4'))).without_audio()
# Generate bbox
frame = clip.get_frame(0)
clip_height, clip_width, _ = frame.shape

BOX_THRES = st.sidebar.select_slider('BBOX_THRES', options=[30, 60], value=30)
image = frame.copy()
BOX = extract_bbox(image)

# clips = []
# data = [
#     ('Ready for \nthe next set', 'red', 30, 3),
#     ('3', 'red', 70, 1),
#     ('2', 'red', 70, 1),
#     ('1', 'red', 70, 1),
# ]
# bg_white = np.ones_like(frame) * 255
# for k, (txt, color, fontsize, duration) in enumerate(data):
#     txt_clip = TextClip(txt, fontsize=fontsize, color=color, bg_color='transparent').set_pos('center')
#     bg_clip = ImageClip(frame)
#     wait_clip = CompositeVideoClip([bg_clip, txt_clip]).set_duration(duration)
#     wait_clip = wait_clip
#     clips.append(wait_clip)

fps = 0
i = 0

START_SET = st.sidebar.button('START A SET')
fps_text = st.sidebar.markdown('')

if START_SET:
    col_left, col_right = st.columns([1, 1])
    FIT_BOX = False

    with col_left:
        stframe = st.empty()
    
    with col_left:
        prevTime = 0
        is_in_box_count = [0]*5  # queue
        while not FIT_BOX:
            video = cv2.VideoCapture(0)
            video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(video.get(cv2.CAP_PROP_FPS))
            crop_width = int((video_width - clip_width) / 2)
            crop_height = int((video_height - clip_height) / 2)
            st.sidebar.write(f'[Video] width={video_width}, height={video_height}, fps={fps_input}')
            st.sidebar.write(f'[Clip] width={clip_width}, height={clip_height}, crop={crop_width, crop_height}')
            while video.isOpened():
                i +=1
                ret, frame = video.read()
                # control size
                frame = cv2.flip(frame, 1)
                frame = frame[crop_height:(video_height-crop_height), crop_width:(video_width-crop_width)].copy()
                _, judge_points, frame = extract_landmarks(image=frame, rt_image=False)
                
                is_in_box = check_in_box(BOX, judge_points)
                # print(f"right shoulder: {in_box(BOX, judge_points['right_shoulder'])}")
                # print(f"right elbow: {in_box(BOX, judge_points['right_elbow'])}")
                # print(is_in_box, judge_points, BOX)
                color = COLOR_DICT['green'] if is_in_box else COLOR_DICT['red']
                cv2.rectangle(frame, BOX[:2], BOX[2:], color, 2)
                
                currTime = time.time()
                fps = 1/(currTime - prevTime)
                fps_text.write(f"<h1 style='text-align: center; color:red;'> FPS: {int(fps)}</h1>", unsafe_allow_html=True)
                prevTime = currTime

                stframe.image(frame, channels='BGR', use_column_width=True)
                is_in_box_count.append(int(is_in_box))
                if sum(is_in_box_count) == 5:
                    break
                else:
                    is_in_box_count.pop(0)
            video.release()
            FIT_BOX = True
            print('is fit')
        
        while FIT_BOX:
            with col_right:
                gold_stframe = st.empty()
                wait_video = cv2.VideoCapture(str(VIDEO_PATH / 'intermediate.mp4'))
                wait_video_width = int(wait_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                wait_video_height = int(wait_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                wait_video_fps_input = int(wait_video.get(cv2.CAP_PROP_FPS))
                wait_idx = 0
                while wait_video.isOpened():
                    wait_idx +=1
                    wait_ret, wait_frame = wait_video.read()
                    if wait_ret:
                        gold_stframe.image(wait_frame, channels='BGR', use_column_width=True)
                    else:
                        break

    # gold_video = cv2.VideoCapture(str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_crop_720p.mp4')))
    # gold_video_width = int(gold_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # gold_video_height = int(gold_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # gold_video_fps_input = int(gold_video.get(cv2.CAP_PROP_FPS))
    # gold_idx = 0
    # while gold_video.isOpened():
    #     gold_idx +=1
    #     gold_ret, gold_frame = gold_video.read()
    #     if gold_ret:
    #         gold_stframe.image(gold_frame, channels='BGR', use_column_width=True)
    #     else:
    #         break

    # with col_left:
        
    #     prevTime = 0

    #     while video.isOpened():
    #         i +=1
    #         ret, frame = video.read()
    #         # control size
    #         frame = frame[crop_height:(video_height-crop_height), crop_width:(video_width-crop_width)]
    #         print(frame.shape)  
    #         # flip
    #         frame = cv2.flip(frame, 1)
    #         if not ret:
    #             continue

    #         currTime = time.time()
    #         fps = 1/(currTime - prevTime)
    #         fps_text.write(f"<h1 style='text-align: center; color:red;'> FPS: {int(fps)}</h1>", unsafe_allow_html=True)
    #         prevTime = currTime

    #         # frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
    #         # frame = image_resize(image=frame, width=720)
    #         stframe.image(frame, channels='BGR', use_column_width=True)
else:
    video_file = str(VIDEO_PATH / (VIDEO_DICT[TASK] + '.mp4'))
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.video(video_file, format="video/mp4", start_time=0)

# with col_left:
#     POSE = MP_POSE.Pose(
#         min_detection_confidence=0.5, 
#         min_tracking_confidence=0.5, 
#         model_complexity=2, 
#         enable_segmentation=False
#     )
#     prevTime = 0

#     while video.isOpened():
#         i +=1
#         ret, frame = video.read()
#         # control size
#         frame = frame[:, crop_width:(video_width-crop_width-1)]  
#         # flip
#         frame = cv2.flip(frame, 1)
#         if not ret:
#             continue
        
#         # color = COLOR_DICT['red']
#         # cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)
        
#         # Make detection
#         results = POSE.process(frame)
#         img_h, img_w, _ = frame.shape
#         frame.flags.writeable = True
#         angles = np.zeros((1, 2))
#         # Angles count
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
            
#             # Get coordinates
#             left_shoulder = [
#                 landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].x*img_h, 
#                 landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y*img_w
#             ]
#             left_elbow = [
#                 landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].x*img_h,
#                 landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].y*img_w
#             ]
#             left_knee = [
#                 landmarks[MP_POSE.PoseLandmark.LEFT_KNEE.value].x*img_h,
#                 landmarks[MP_POSE.PoseLandmark.LEFT_KNEE.value].y*img_w
#             ]
#             left_hip = [
#                 landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].x*img_h,
#                 landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].y*img_w
#             ]

#             right_shoulder = [
#                 landmarks[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value].x*img_h,
#                 landmarks[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value].y*img_w
#             ]
#             right_elbow = [
#                 landmarks[MP_POSE.PoseLandmark.RIGHT_ELBOW.value].x*img_h,
#                 landmarks[MP_POSE.PoseLandmark.RIGHT_ELBOW.value].y*img_w
#             ]
#             right_knee = [
#                 landmarks[MP_POSE.PoseLandmark.RIGHT_KNEE.value].x*img_h,
#                 landmarks[MP_POSE.PoseLandmark.RIGHT_KNEE.value].y*img_w
#             ]
#             right_hip = [
#                 landmarks[MP_POSE.PoseLandmark.RIGHT_HIP.value].x*img_h,
#                 landmarks[MP_POSE.PoseLandmark.RIGHT_HIP.value].y*img_w
#             ]
            
#             # Calculate angle & Store in (1, 4) vector
#             angles[0, 0] = calculate_angle(left_elbow, left_shoulder, left_hip) #angle_1
#             angles[0, 1] = calculate_angle(right_elbow, right_shoulder, right_hip)
#             # angles[0][2] = calculate_angle(left_knee, left_hip, right_hip)
#             # angles[0][3] = calculate_angle(right_knee, right_hip, left_hip)
        
#         MP_DRAWING.draw_landmarks(
#             frame, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
#             MP_DRAWING.DrawingSpec(color=tuple(reversed((245,117,66))), thickness=2, circle_radius=2), 
#             MP_DRAWING.DrawingSpec(color=tuple(reversed((245,66,230))), thickness=2, circle_radius=2)
#         )
#         # FPS Counter
#         currTime = time.time()
#         fps = 1/(currTime - prevTime)
#         fps_text.write(f"<h1 style='text-align: center; color:red;'> FPS: {int(fps)}</h1>", unsafe_allow_html=True)
#         prevTime = currTime

#         # frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
#         # frame = image_resize(image=frame, width=720)
#         stframe.image(frame, channels='BGR', use_column_width=False)

# with col_gold:

# with col_gold:
#     # if START:
#     #     video_file=str(VIDEO_PATH / (VIDEO_DICT[TASK] + '.mp4'))
        
#     video_file = str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_gold.mp4'))
#     st.video(video_file, format="video/mp4", start_time=0)
