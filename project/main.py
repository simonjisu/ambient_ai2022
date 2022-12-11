import av
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pathlib import Path
import mediapipe as mp
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, TextClip, VideoClip, concatenate_videoclips, clips_array

with st.sidebar:
    TASK = st.selectbox('choose task', options=['hipjoint', 'waist', 'trapezius'])
    FPS = st.select_slider('Frame Per Seconds', options=[5, 15, 30], value=30)
    ANGLE_THRES = st.slider('Angle Threshold', 0.0, 5.0, value=0.5, step=0.1)
    ACC_THRES = st.slider('Accuracy Threshold', 50.0, 100.0, value=50.0, step=1.0)

BOX_THRES = 30
START = False
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLE = mp.solutions.drawing_styles
MP_POSE = mp.solutions.pose

VIDEO_PATH = Path('.').resolve() / 'videos'
VIDEO_DICT = {
    'hipjoint': '01_hipjoint',
    'waist': '02_waist', 
    'trapezius': '03_trapezius'
}
BOXES_DICT = {
    'hipjoint': (0, 341, 398, 622), 
    'waist': (0, 327, 406, 552), 
    'trapezius': (0, 122, 296, 720)
}
VIDEO_SPEC = {
    'video': {
        'width': {'ideal': 406, 'min': 406}, 
        'height': {'ideal': 720, 'min': 720},
        'frameRate': {'ideal': FPS, 'max': FPS}
    },
    'audio': False,
}
# BGR
COLOR_DICT = {
    'red': (0, 0, 255),
    'green': (0, 255, 0)
}


ANGLE_DATA = np.loadtxt(str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_angles.txt')), 
    dtype='str', comments='#', delimiter=',', skiprows=0, encoding='bytes')[:, 1:-1].astype(np.float64)



def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 

def process_mediapipe(image, img_w, img_h):
    global TASK
    with MP_POSE.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5, 
            model_complexity=2, 
            enable_segmentation=True
        ) as POSE:
        # Make detection
        results = POSE.process(image)
        angles = np.zeros((1,4))
        try:
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
            
            # Calculate angle & Store in (1, 4) vector
            angles[0][0] = calculate_angle(left_elbow, left_shoulder, right_shoulder) #angle_1
            angles[0][1] = calculate_angle(right_elbow, right_shoulder, left_shoulder)
            angles[0][2] = calculate_angle(left_knee, left_hip, right_hip)
            angles[0][3] = calculate_angle(right_knee, right_hip, left_hip)
        except:
            (left_shoulder, right_shoulder, left_hip, right_hip) = ((-1, -1), (-1, -1), (-1, -1), (-1, -1))

    if TASK == 'hipjoint':
        judge_points = (left_shoulder, right_shoulder, left_hip, right_hip)
    else:
        judge_points = None
    
    return results, angles, judge_points

def in_box(box, coor):
    logic = box[0] < coor[0] < box[2] and box[2] < coor[1] < box[3]
    return logic

def check_in_box(box, judge_points):
    res = []
    for coor in judge_points:
        res.append(int(in_box(box, coor)))
    return sum(res) == len(judge_points)

# def is_fit(seg_mask):
#     global angle_threshold
#     mask_img = (seg_mask > angle_threshold).astype(np.uint8)
#     x, y, w, h = cv2.boundingRect(mask_img)
#     box_ = (x, y, x+w, y+h)
#     if in_box(box_, left_shoulder) and in_box(box_, right_shoulder) and in_box(box_, left_hip) and in_box(box_, right_hip):
#         #2: Angle Logic-to compare ground truth
#         if frame in frame_list: #frames to compare
#             ground_truth = read_ground_truth(file_name_) #Get Matrix of given exercise
#             frame_, angle_1, angle_2, angle_3, angle_4, angle_5, _ = ground_truth[frame]


def video_frame_callback(frame):
    # https://github.com/whitphx/streamlit-webrtc
    global START
    global TASK
    
    image = frame.to_ndarray(format="bgr24")
    img_w, img_h, _ = image.shape
    results, angles, judge_points = process_mediapipe(image, img_w, img_h)
    # width_left, height_top, width_right, height_bottom
    box = BOXES_DICT[TASK]
    color = COLOR_DICT['green'] if START else COLOR_DICT['red']
    cv2.rectangle(image, box[:2], box[2:], color, 2)

    MP_DRAWING.draw_landmarks(
        image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
        MP_DRAWING.DrawingSpec(color=tuple(reversed((245, 117, 66))), thickness=2, circle_radius=2), 
        MP_DRAWING.DrawingSpec(color=tuple(reversed((245, 66, 230))), thickness=2, circle_radius=2)
    )

    if judge_points is not None and check_in_box(box, judge_points):
        START = True
        print(START)
    # START TO RUN
    if START:
        MP_DRAWING.draw_landmarks(
            image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
            MP_DRAWING.DrawingSpec(color=tuple(reversed((245,117,66))), thickness=2, circle_radius=2), 
            MP_DRAWING.DrawingSpec(color=tuple(reversed((245,66,230))), thickness=2, circle_radius=2)
        )
        # Render detections
        with col_gold:
            st.write('### User')
            video_file = str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_gold.mp4'))
            st.video(video_file, format="video/mp4", start_time=0)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

col_user, col_gold = st.columns([1, 1])

st.write(ANGLE_DATA)
st.write(ANGLE_DATA.shape, ANGLE_DATA.shape[0] / 30)

with col_user:
    st.write('### User')
    webrtc_streamer(
        key='task', 
        video_frame_callback=video_frame_callback,
        media_stream_constraints=VIDEO_SPEC
    )

# with col_gold:
#     # if START:
#     #     video_file=str(VIDEO_PATH / (VIDEO_DICT[TASK] + '.mp4'))
        
#     video_file = str(VIDEO_PATH / (VIDEO_DICT[TASK] + '_gold.mp4'))
#     st.video(video_file, format="video/mp4", start_time=0)
