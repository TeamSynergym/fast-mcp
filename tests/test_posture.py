import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.posture_analyzer import PostureAnalyzer
from app.utils.pose_utils import PoseVisualizer


def test_posture_analysis():
    """Test the posture analyzer with sample images."""
    
    # Initialize the posture analyzer
    print("🔧 Initializing Posture Analyzer...")
    analyzer = PostureAnalyzer(model_path="models/yolopose_v1.pt")
    
    # Initialize the visualizer
    visualizer = PoseVisualizer()
    
    # Test with sample images
    test_images = [
        "data/test_images/sample1.jpg",
        "data/test_images/sample2.jpg",
        "data/test_images/sample3.jpg"
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Test image not found: {image_path}")
            continue
            
        print(f"\n📸 Analyzing posture in: {image_path}")
        
        try:
            # Analyze posture
            analysis_result = analyzer.analyze_posture(image_path)
            
            if analysis_result["success"]:
                print(f"✅ Successfully analyzed {analysis_result['num_people']} person(s)")
                
                # Print analysis for each person
                for i, analysis in enumerate(analysis_result["posture_analysis"]):
                    print(f"\n👤 Person {i + 1}:")
                    print(f"   Overall Score: {analysis['overall_score']:.2f}")
                    print(f"   Head Position: {analysis['head_position']['status']}")
                    print(f"   Shoulder Alignment: {analysis['shoulder_alignment']['status']}")
                    print(f"   Back Straightness: {analysis['back_straightness']['status']}")
                    
                    print("   Recommendations:")
                    for rec in analysis['recommendations']:
                        print(f"   • {rec}")
                
                # Create visualization
                output_path = f"data/test_images/analysis_{Path(image_path).stem}.jpg"
                visualizer.draw_posture_analysis(image_path, analysis_result["posture_analysis"][0], output_path)
                
            else:
                print(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")


def test_pose_detection_only():
    """Test only pose detection without posture analysis."""
    
    print("🔧 Testing pose detection only...")
    analyzer = PostureAnalyzer(model_path="models/yolopose_v1.pt")
    
    # Test with a single image
    test_image = "data/test_images/sample1.jpg"
    
    if not os.path.exists(test_image):
        print(f"⚠️  Test image not found: {test_image}")
        return
    
    try:
        # Detect pose
        pose_result = analyzer.detect_pose(test_image)
        
        if pose_result["success"]:
            print(f"✅ Detected pose for {pose_result['num_people']} person(s)")
            
            # Print keypoint information
            for i, keypoints in enumerate(pose_result["keypoints"]):
                print(f"\n👤 Person {i + 1} keypoints:")
                print(f"   Number of keypoints: {len(keypoints)}")
                print(f"   Keypoint shape: {keypoints.shape}")
                
                # Show first few keypoints
                for j in range(min(5, len(keypoints))):
                    x, y, conf = keypoints[j]
                    print(f"   Keypoint {j}: ({x:.1f}, {y:.1f}) - confidence: {conf:.2f}")
        else:
            print(f"❌ Pose detection failed: {pose_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error in pose detection: {e}")


def create_test_directory():
    """Create test directory structure."""
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created test directory: {test_dir}")
    print(f"📁 Created models directory: {models_dir}")
    print("\n📝 Next steps:")
    print("1. Place your yolopose_v1.pt model in the models/ directory")
    print("2. Add some test images to data/test_images/ directory")
    print("3. Run the test script")


if __name__ == "__main__":
    print("🧪 Starting Posture Analysis Tests")
    print("=" * 50)
    
    # Create test directories
    create_test_directory()
    
    # Check if model exists
    model_path = "models/yolopose_v1.pt"
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model not found at {model_path}")
        print("Please place your yolopose_v1.pt model in the models/ directory")
        exit(1)
    
    # Run tests
    print("\n" + "=" * 50)
    test_pose_detection_only()
    
    print("\n" + "=" * 50)
    test_posture_analysis()
    
    print("\n✅ Tests completed!") 