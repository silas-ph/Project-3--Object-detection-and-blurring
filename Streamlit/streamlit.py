import streamlit as st
from PIL import Image, ImageFilter
import cv2
import tempfile
import os
from multiprocessing import Pool, cpu_count

# Dummy model function - replace with your actual model
def run_model(input_image):
    # For demonstration, we will just apply a filter to the image
    output_image = input_image.filter(ImageFilter.SMOOTH)
    return output_image

def video_to_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    return frame_paths

def frame_extraction_worker(params):
    frame, frames_dir, count = params
    frame_path = os.path.join(frames_dir, f'frame_{count:04d}.jpg')
    cv2.imwrite(frame_path, frame)
    return frame_path

def video_to_frames_parallel(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame, frames_dir, frame_count))
        frame_count += 1
    
    cap.release()
    
    with Pool(processes=cpu_count()) as pool:
        frame_paths = pool.map(frame_extraction_worker, frames)
    
    return frame_paths

def frames_to_video_worker(params):
    frame_file, frames_dir = params
    frame_path = os.path.join(frames_dir, frame_file)
    return cv2.imread(frame_path)

def frames_to_video_parallel(frames_dir, output_video_path, fps):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
    if not frame_files:
        raise ValueError("No frames found in the directory")

    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError(f"Could not read the first frame from {first_frame_path}")
        
    height, width, layers = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_paths = [(frame_file, frames_dir) for frame_file in frame_files]
    
    with Pool(processes=cpu_count()) as pool:
        frames = pool.map(frames_to_video_worker, frame_paths)
    
    for frame in frames:
        if frame is None:
            continue
        video_writer.write(frame)
    
    video_writer.release()

def process_frame(frame_path):
    frame = Image.open(frame_path)
    processed_frame = run_model(frame)
    processed_frame.save(frame_path)

def process_frames_parallel(frame_paths):
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_frame, frame_paths)

def streamlit_app():
    st.title("Video Processing App")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg4"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        frames_dir = tempfile.mkdtemp()
        
        st.write("Extracting frames...")
        frame_paths = video_to_frames_parallel(video_path, frames_dir)
        st.write(f"Extracted {len(frame_paths)} frames.")
        
        st.write("Processing frames...")
        process_frames_parallel(frame_paths)
        st.write("Frames processed.")
        
        output_video_path = os.path.join(frames_dir, 'output_video.mp4')
        st.write("Reassembling video...")
        try:
            frames_to_video_parallel(frames_dir, output_video_path, fps=30)
            st.write("Video reassembled.")
            
            with open(output_video_path, 'rb') as f:
                st.download_button(label="Download Processed Video", data=f, file_name="processed_video.mp4", mime="video/mp4")
                
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    streamlit_app()
