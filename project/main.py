import av
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pathlib import Path

VIDEO_PATH = Path('.').resolve() / 'videos' / 'trim'
VIDEO_DICT = {
    'hipjoint': '00_hipjoint.mp4',
    'waist': '02_waist.mp4', 
    'trapezius': '03_trapezius.mp4'
}

with st.sidebar:
    mode = st.selectbox('choose mode', options=['hipjoint', 'waist', 'trapezius'])

flip = st.checkbox("Flip")

def video_frame_callback(frame):
    # https://github.com/whitphx/streamlit-webrtc
    img = frame.to_ndarray(format="bgr24")
    # resize_img = cv2.resize(img, (406, 720))

    flipped = img[::-1,:,:] if flip else img

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")

col_gold, col_user = st.columns(2)
with col_gold:
    video_file =open(VIDEO_PATH / VIDEO_DICT[mode], mode='rb')

    st.video(video_file, format="video/mp4", start_time=0)

with col_user:
    webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
