import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import cv2

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
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
            
            # Analyze button
            if st.button("🔍 Analyze Posture", type="primary"):
                with st.spinner("Analyzing posture..."):
                    try:
                        # Analyze posture
                        analysis_result = analyzer.analyze_posture(temp_image_path)
                        
                        if analysis_result["success"]:
                            st.success(f"✅ Analyzed {analysis_result['num_people']} person(s)")
                            
                            # Store results in session state for display in col2
                            st.session_state.analysis_result = analysis_result
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
        if hasattr(st.session_state, 'analysis_result') and st.session_state.analysis_result:
            analysis_result = st.session_state.analysis_result
            
            # Display results for each person
            for i, analysis in enumerate(analysis_result["posture_analysis"]):
                st.subheader(f"👤 Person {i + 1}")
                
                # Overall score with progress bar
                overall_score = analysis['overall_score']
                st.metric("Overall Posture Score", f"{overall_score:.2f}/1.0")
                st.progress(overall_score)
                
                # Individual metrics
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Head position
                    head_status = analysis['head_position']['status']
                    head_score = analysis['head_position'].get('score', 0)
                    if head_score is not None:
                        st.metric("Head Position", f"{head_score:.2f}")
                        if head_status == "good":
                            st.success("✅ Good head position")
                        else:
                            st.warning("⚠️ Needs improvement")
                    else:
                        st.info("ℹ️ Insufficient data")
                    
                    # Shoulder alignment
                    shoulder_status = analysis['shoulder_alignment']['status']
                    shoulder_score = analysis['shoulder_alignment'].get('score', 0)
                    if shoulder_score is not None:
                        st.metric("Shoulder Alignment", f"{shoulder_score:.2f}")
                        if shoulder_status == "good":
                            st.success("✅ Good shoulder alignment")
                        else:
                            st.warning("⚠️ Needs improvement")
                    else:
                        st.info("ℹ️ Insufficient data")
                
                with col_b:
                    # Back straightness
                    back_status = analysis['back_straightness']['status']
                    back_score = analysis['back_straightness'].get('score', 0)
                    if back_score is not None:
                        st.metric("Back Straightness", f"{back_score:.2f}")
                        if back_status == "good":
                            st.success("✅ Good back posture")
                        else:
                            st.warning("⚠️ Needs improvement")
                    else:
                        st.info("ℹ️ Insufficient data")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                recommendations = analysis.get('recommendations', [])
                if recommendations:
                    for rec in recommendations:
                        st.info(f"• {rec}")
                else:
                    st.success("🎉 Great posture! Keep it up!")
                
                # Show detailed metrics in expander
                with st.expander("📈 Detailed Metrics"):
                    if 'head_position' in analysis and analysis['head_position'].get('head_offset'):
                        st.write(f"**Head Offset:** {analysis['head_position']['head_offset']:.2f} pixels")
                    
                    if 'shoulder_alignment' in analysis and analysis['shoulder_alignment'].get('height_difference'):
                        st.write(f"**Shoulder Height Difference:** {analysis['shoulder_alignment']['height_difference']:.2f} pixels")
                    
                    if 'back_straightness' in analysis and analysis['back_straightness'].get('angle_difference'):
                        st.write(f"**Back Angle Difference:** {analysis['back_straightness']['angle_difference']:.2f} degrees")
                
                # Visualization
                st.subheader("🎨 Visualization")
                if hasattr(st.session_state, 'temp_image_path') and st.session_state.temp_image_path:
                    try:
                        # Create visualization
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_viz:
                            visualizer.draw_posture_analysis(
                                st.session_state.temp_image_path, 
                                analysis, 
                                tmp_viz.name
                            )
                            
                            # Display visualization
                            viz_image = Image.open(tmp_viz.name)
                            st.image(viz_image, caption="Posture Analysis Visualization", use_column_width=True)
                            
                            # Clean up
                            os.unlink(tmp_viz.name)
                            
                    except Exception as e:
                        st.error(f"❌ Error creating visualization: {e}")
        
        else:
            st.info("👆 Upload an image and click 'Analyze Posture' to see results here")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This system analyzes posture based on detected pose keypoints. "
        "For best results, ensure the person is clearly visible and well-lit in the image."
    )


if __name__ == "__main__":
    main() 