#!/usr/bin/env python3
"""
Custom analysis example for pad-analytics package.

This example demonstrates:
1. Direct image processing functions
2. Feature extraction from PAD images
3. Building custom analysis pipelines
"""

from pad_analytics import regionRoutine, pixelProcessing, fileManagement
import numpy as np
import cv2

def analyze_pad_image(image_path):
    """
    Custom analysis of a PAD image.
    
    Args:
        image_path: Path to PAD image file
    
    Returns:
        dict: Extracted features from the image
    """
    print(f"\nAnalyzing image: {image_path}")
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print("   Error: Could not load image")
            return None
        
        print(f"   Image shape: {img.shape}")
        
        # Example: Extract average colors from specific regions
        # Define some example regions (in real use, these would be PAD lanes)
        regions = [
            [(100, 100), (150, 150)],  # Region 1: top-left to bottom-right
            [(200, 100), (250, 150)],  # Region 2
            [(300, 100), (350, 150)],  # Region 3
        ]
        
        features = {}
        
        for i, (top_left, bottom_right) in enumerate(regions):
            # Extract pixels from region
            pixels = []
            for y in range(top_left[1], bottom_right[1]):
                for x in range(top_left[0], bottom_right[0]):
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        pixels.append((y, x))
            
            if pixels:
                # Calculate average RGB values
                avg_rgb = pixelProcessing.avgPixels(pixels, img)
                features[f'region_{i+1}_rgb'] = avg_rgb
                
                # Calculate average HSV values
                avg_hsv = pixelProcessing.avgPixelsHSV(pixels, img)
                features[f'region_{i+1}_hsv'] = avg_hsv
                
                # Calculate average LAB values
                avg_lab = pixelProcessing.avgPixelsLAB(pixels, img)
                features[f'region_{i+1}_lab'] = avg_lab
                
                print(f"   Region {i+1} - RGB: {avg_rgb}, HSV: {avg_hsv}, LAB: {avg_lab}")
        
        return features
        
    except Exception as e:
        print(f"   Error in analysis: {e}")
        return None


def create_feature_vector(features):
    """
    Convert extracted features to a feature vector for ML models.
    
    Args:
        features: Dictionary of extracted features
    
    Returns:
        np.array: Feature vector
    """
    if not features:
        return None
    
    # Flatten all color values into a single vector
    vector = []
    
    # Sort keys to ensure consistent ordering
    for key in sorted(features.keys()):
        values = features[key]
        if isinstance(values, (list, tuple)):
            vector.extend(values)
        else:
            vector.append(values)
    
    return np.array(vector)


def main():
    print("=== PAD Analytics Custom Analysis Example ===\n")
    
    # 1. Generate index for a PAD layout
    print("1. Generating PAD lane index...")
    index = fileManagement.genIndex(regions=3)  # 3 regions per lane
    print(f"   Generated index with {len(index)} elements")
    print(f"   First few elements: {index[:10]}")
    
    # 2. Create a synthetic test image
    print("\n2. Creating synthetic PAD image for testing...")
    # Create a simple test image with colored regions
    test_img = np.zeros((400, 500, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate PAD lanes
    cv2.rectangle(test_img, (100, 100), (150, 150), (255, 0, 0), -1)  # Blue
    cv2.rectangle(test_img, (200, 100), (250, 150), (0, 255, 0), -1)  # Green
    cv2.rectangle(test_img, (300, 100), (350, 150), (0, 0, 255), -1)  # Red
    
    # Save test image
    test_image_path = "test_pad_image.png"
    cv2.imwrite(test_image_path, test_img)
    print(f"   Test image saved to: {test_image_path}")
    
    # 3. Analyze the test image
    print("\n3. Analyzing test image...")
    features = analyze_pad_image(test_image_path)
    
    if features:
        # 4. Create feature vector
        print("\n4. Creating feature vector...")
        feature_vector = create_feature_vector(features)
        print(f"   Feature vector shape: {feature_vector.shape}")
        print(f"   Feature vector: {feature_vector[:10]}...")  # Show first 10 values
        
        # 5. Example: Use features for custom analysis
        print("\n5. Custom analysis results:")
        
        # Check if blue channel is dominant in region 1
        rgb_1 = features.get('region_1_rgb', [0, 0, 0])
        if rgb_1[2] > rgb_1[0] and rgb_1[2] > rgb_1[1]:
            print("   ✓ Region 1 shows blue dominance (expected)")
        
        # Check if green channel is dominant in region 2
        rgb_2 = features.get('region_2_rgb', [0, 0, 0])
        if rgb_2[1] > rgb_2[0] and rgb_2[1] > rgb_2[2]:
            print("   ✓ Region 2 shows green dominance (expected)")
        
        # Check if red channel is dominant in region 3
        rgb_3 = features.get('region_3_rgb', [0, 0, 0])
        if rgb_3[0] > rgb_3[1] and rgb_3[0] > rgb_3[2]:
            print("   ✓ Region 3 shows red dominance (expected)")
    
    print("\n=== Example Complete ===")
    print("\nNote: In real usage, you would:")
    print("- Load actual PAD images from the API or local files")
    print("- Use proper lane detection algorithms")
    print("- Feed extracted features to ML models")
    print("- Compare results with known standards")


if __name__ == "__main__":
    main()