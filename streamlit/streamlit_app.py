import streamlit as st
import os
import sys
import tempfile
from PIL import Image
import numpy as np
import cv2

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our posture analysis modules
from app.services.posture_analyzer import PostureAnalyzer
from app.utils.pose_utils import PoseVisualizer


def main():
    st.set_page_config(
        page_title="Posture Analysis with YOLO",
        page_icon="🧍",
        layout="wide"
    )
    
    st.title("🧍 Posture Analysis with YOLO")
    st.markdown("Upload an image to analyze posture using your trained YOLO model")
    
    # Sidebar for model path
    st.sidebar.header("⚙️ Settings")
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="models/yolopose_v1.pt",
        help="Path to your trained YOLO pose model"
    )
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.info("Please place your yolopose_v1.pt model in the models/ directory")
        return
    
    # Initialize analyzer
    try:
        analyzer = PostureAnalyzer(model_path=model_path)
        visualizer = PoseVisualizer()
        st.success("✅ YOLO model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing a person to analyze their posture"
        )
        
        if uploaded_file is not None:
            # Display original image
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                # Convert to RGB if needed (JPEG doesn't support RGBA)
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
            
            # Analyze button
            if st.button("🔍 Analyze Posture", type="primary"):
                with st.spinner("Analyzing posture..."):
                    try:
                        # Analyze posture
                        analysis_result = analyzer.analyze_posture(temp_image_path)
                        
                        if analysis_result["success"]:
                            st.success(f"✅ Detected {analysis_result['num_people']} person(s)")
                            
                            # Debug: print the structure
                            st.write("Debug - Result structure:", analysis_result.keys())
                            
                            # Store results in session state for display in col2
                            st.session_state.pose_result = analysis_result
                            st.session_state.temp_image_path = temp_image_path
                            
                        else:
                            st.error(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_image_path):
                            os.unlink(temp_image_path)
    
    with col2:
        st.header("📊 Analysis Results")
        
        # Display results if available
        if hasattr(st.session_state, 'pose_result') and st.session_state.pose_result:
            pose_result = st.session_state.pose_result
            
            # Display results for each person
            # Check which key exists in the result
            if "pose_data" in pose_result:
                data_key = "pose_data"
            elif "posture_analysis" in pose_result:
                data_key = "posture_analysis"
            else:
                st.error(f"❌ Unexpected result structure. Available keys: {list(pose_result.keys())}")
                return
                
            for i, pose_data in enumerate(pose_result[data_key]):
                st.subheader(f"👤 Person {i + 1}")
                
                # Handle both new and old data structures
                if 'num_keypoints' in pose_data:
                    # New structure with keypoints only
                    num_keypoints = pose_data['num_keypoints']
                    keypoints = pose_data['keypoints']
                elif 'keypoints' in pose_data:
                    # Old structure with posture analysis
                    keypoints = pose_data['keypoints']
                    # Count confident keypoints
                    num_keypoints = len([kp for kp in keypoints.values() if kp['confidence'] > 0.1])
                else:
                    st.error("❌ No keypoints found in data")
                    continue
                
                st.metric("Detected Keypoints", f"{num_keypoints}/17")
                st.progress(num_keypoints / 17.0)
                
                # Display keypoints
                st.subheader("📍 Keypoints")
                
                # Create a table of keypoints
                keypoint_data = []
                for name, kp in keypoints.items():
                    if kp['confidence'] > 0.1:  # Only show confident keypoints
                        keypoint_data.append({
                            "Keypoint": name,
                            "X": f"{kp['x']:.1f}",
                            "Y": f"{kp['y']:.1f}",
                            "Confidence": f"{kp['confidence']:.3f}"
                        })
                
                if keypoint_data:
                    st.dataframe(keypoint_data, use_container_width=True)
                else:
                    st.warning("⚠️ No confident keypoints detected")
                
                # Show all keypoints in expander
                with st.expander("🔍 All Keypoints (including low confidence)"):
                    all_keypoints = []
                    for name, kp in keypoints.items():
                        all_keypoints.append({
                            "Keypoint": name,
                            "X": f"{kp['x']:.1f}",
                            "Y": f"{kp['y']:.1f}",
                            "Confidence": f"{kp['confidence']:.3f}",
                            "Status": "✅" if kp['confidence'] > 0.1 else "❌"
                        })
                    st.dataframe(all_keypoints, use_container_width=True)
                
                # Visualization
                st.subheader("🎨 Visualization")
                if uploaded_file is not None:
                    try:
                        # Create visualization using the uploaded file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_viz:
                            # Save uploaded file again for visualization
                            uploaded_image = Image.open(uploaded_file)
                            if uploaded_image.mode in ('RGBA', 'LA', 'P'):
                                uploaded_image = uploaded_image.convert('RGB')
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_input:
                                uploaded_image.save(tmp_input.name)
                                
                                visualizer.draw_pose_on_image(
                                    tmp_input.name, 
                                    pose_data['keypoints'], 
                                    tmp_viz.name
                                )
                                
                                # Display visualization
                                viz_image = Image.open(tmp_viz.name)
                                st.image(viz_image, caption="Pose Detection Visualization", use_container_width=True)
                                
                                # Clean up
                                os.unlink(tmp_viz.name)
                                os.unlink(tmp_input.name)
                            
                    except Exception as e:
                        st.error(f"❌ Error creating visualization: {e}")
        
        else:
            st.info("👆 Upload an image and click 'Analyze Posture' to see results here")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This system detects pose keypoints using your YOLO11x-pose model trained on COCO2017. "
        "For best results, ensure the person is clearly visible and well-lit in the image."
    )


if __name__ == "__main__":
    main() 