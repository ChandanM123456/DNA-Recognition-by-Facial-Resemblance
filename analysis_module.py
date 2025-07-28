import face_recognition
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
from PIL import Image # ADDED THIS LINE

# --- Configuration ---
CAPTURED_FACES_BASE_DIR = "captured_faces"
ANALYSIS_RESULTS_CSV = "analysis_results.csv"

# Define weights for each feature for the overall similarity score
# These weights are illustrative and can be adjusted based on desired importance
FEATURE_WEIGHTS = {
    'nose_length': 0.20,
    'nose_width': 0.15,
    'lip_thickness_upper': 0.10,
    'lip_thickness_lower': 0.10,
    'mouth_width': 0.15, # Using mouth_width, as mouth_aspect_ratio was showing nan
    'left_eye_width': 0.10,
    'right_eye_width': 0.10,
    'inter_eye_distance': 0.05,
    'chin_height': 0.05
    # Removed 'eye_aspect_ratio', 'face_width', 'face_height' as they were leading to nan
    # and might require more complex/robust calculation or additional landmarks.
}

# Ensure weights sum to 1 (or adjust them proportionally if they don't initially)
total_weight = sum(FEATURE_WEIGHTS.values())
if abs(total_weight - 1.0) > 1e-6: # Check if not approximately 1
    print(f"Warning: FEATURE_WEIGHTS do not sum to 1. They sum to {total_weight:.2f}. Normalizing...")
    for feature, weight in FEATURE_WEIGHTS.items():
        FEATURE_WEIGHTS[feature] = weight / total_weight
    print(f"Normalized weights sum to: {sum(FEATURE_WEIGHTS.values()):.2f}")


# --- Helper Functions for Face Analysis ---

def get_face_landmarks(image_path):
    """
    Loads an image and returns landmarks for the first detected face.
    Returns (face_landmarks, pil_image) or (None, None) if no face is found.
    """
    try:
        image_np = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image_np)

        if not face_locations:
            print(f"  No face found in {image_path}")
            return None, None

        # Assume the first detected face is the main subject
        face_landmarks_list = face_recognition.face_landmarks(image_np, [face_locations[0]])
        
        if face_landmarks_list:
            return face_landmarks_list[0], Image.fromarray(image_np)
        else:
            print(f"  Could not get landmarks for face in {image_path}")
            return None, None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x1, y1) and (x2, y2)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_aspect_ratio(p1, p2, p3, p4, p5, p6):
    """
    Calculates the eye or mouth aspect ratio.
    Points are typically ordered: horizontal points (p1, p4) and vertical points (p2,p3,p5,p6).
    """
    A = calculate_distance(p2, p6)
    B = calculate_distance(p3, p5)
    C = calculate_distance(p1, p4)
    if C == 0: return 0.001 # Avoid division by zero
    return (A + B) / (2.0 * C)

def get_feature_measurements(landmarks):
    """
    Extracts various facial feature measurements (distances, ratios) from landmarks.
    """
    measurements = {}

    # Eye Aspect Ratio (EAR) - Re-enabled for completeness if it starts working
    if 'left_eye' in landmarks and len(landmarks['left_eye']) == 6:
        l_eye = landmarks['left_eye']
        measurements['eye_aspect_ratio_left'] = calculate_aspect_ratio(*l_eye)
        measurements['left_eye_width'] = calculate_distance(l_eye[0], l_eye[3])
    if 'right_eye' in landmarks and len(landmarks['right_eye']) == 6:
        r_eye = landmarks['right_eye']
        measurements['eye_aspect_ratio_right'] = calculate_aspect_ratio(*r_eye)
        measurements['right_eye_width'] = calculate_distance(r_eye[0], r_eye[3])
    
    # Average EAR if both exist
    if 'eye_aspect_ratio_left' in measurements and 'eye_aspect_ratio_right' in measurements:
        measurements['eye_aspect_ratio'] = (measurements['eye_aspect_ratio_left'] + measurements['eye_aspect_ratio_right']) / 2
    elif 'eye_aspect_ratio_left' in measurements:
        measurements['eye_aspect_ratio'] = measurements['eye_aspect_ratio_left']
    elif 'eye_aspect_ratio_right' in measurements:
        measurements['eye_aspect_ratio'] = measurements['eye_aspect_ratio_right']


    # Mouth Aspect Ratio (MAR) and Lip Thickness
    if 'mouth' in landmarks and len(landmarks['mouth']) == 20:
        mouth = landmarks['mouth']
        
        v1 = calculate_distance(mouth[2], mouth[10])
        v2 = calculate_distance(mouth[3], mouth[9])
        v3 = calculate_distance(mouth[4], mouth[8])

        h = calculate_distance(mouth[0], mouth[6])

        if h > 0:
            measurements['mouth_aspect_ratio'] = (v1 + v2 + v3) / (3.0 * h)
            measurements['mouth_width'] = h
        else:
            measurements['mouth_aspect_ratio'] = 0.001
            measurements['mouth_width'] = 0.001

        if len(landmarks['mouth']) >= 20:
            upper_lip_points_outer = [mouth[2], mouth[3], mouth[4]]
            upper_lip_points_inner = [mouth[14], mouth[15], mouth[16]]
            
            thicknesses = []
            for i in range(3):
                thicknesses.append(calculate_distance(upper_lip_points_outer[i], upper_lip_points_inner[i]))
            measurements['lip_thickness_upper'] = np.mean(thicknesses) if thicknesses else 0.001

            lower_lip_points_outer = [mouth[8], mouth[9], mouth[10]]
            lower_lip_points_inner = [mouth[18], mouth[19], mouth[17]]

            thicknesses = []
            for i in range(3):
                thicknesses.append(calculate_distance(lower_lip_points_outer[i], lower_lip_points_inner[i]))
            measurements['lip_thickness_lower'] = np.mean(thicknesses) if thicknesses else 0.001


    # Nose Length and Width
    if 'nose_bridge' in landmarks and len(landmarks['nose_bridge']) >= 3 and \
       'nose_tip' in landmarks and len(landmarks['nose_tip']) >= 1 and \
       'nose_bottom' in landmarks and len(landmarks['nose_bottom']) >= 2:

        nose_top_point = landmarks['nose_bridge'][0]
        nose_bottom_point = landmarks['nose_bottom'][2]
        measurements['nose_length'] = calculate_distance(nose_top_point, nose_bottom_point)
        
        nostril_left = landmarks['nose_bottom'][0]
        nostril_right = landmarks['nose_bottom'][3]
        measurements['nose_width'] = calculate_distance(nostril_left, nostril_right)


    # Inter-eye distance (distance between centers of eyes)
    if 'left_eye' in landmarks and 'right_eye' in landmarks:
        left_eye_center = np.mean(landmarks['left_eye'], axis=0)
        right_eye_center = np.mean(landmarks['right_eye'], axis=0)
        measurements['inter_eye_distance'] = calculate_distance(left_eye_center, right_eye_center)

    # Face Width (from temple to temple or cheekbone to cheekbone)
    if 'chin' in landmarks and len(landmarks['chin']) == 17:
        measurements['face_width'] = calculate_distance(landmarks['chin'][0], landmarks['chin'][16])
        
        # Face Height (from chin to mid-forehead - needs eyebrow/face top points)
        if 'left_eyebrow' in landmarks and 'right_eyebrow' in landmarks:
            eyebrow_mid_x = int((landmarks['left_eyebrow'][3][0] + landmarks['right_eyebrow'][1][0]) / 2)
            eyebrow_mid_y = int((landmarks['left_eyebrow'][3][1] + landmarks['right_eyebrow'][1][1]) / 2)
            mid_eyebrow_point = (eyebrow_mid_x, eyebrow_mid_y)
            
            chin_point = landmarks['chin'][8]
            measurements['face_height'] = calculate_distance(chin_point, mid_eyebrow_point)
            
            # Chin Height (from mouth bottom to chin bottom)
            if 'mouth' in landmarks and len(landmarks['mouth']) >= 9:
                mouth_bottom_outer = landmarks['mouth'][9]
                measurements['chin_height'] = calculate_distance(mouth_bottom_outer, chin_point)
    
    # Ensure all values are non-zero to avoid division by zero or log(0) issues later
    for k, v in measurements.items():
        if v == 0:
            measurements[k] = 0.001

    return measurements

def normalize_feature(value, min_val, max_val):
    """Normalizes a feature value to a 0-1 range."""
    if max_val == min_val: return 0.5 # Avoid division by zero, return mid-point
    return (value - min_val) / (max_val - min_val)

def calculate_similarity_percentage(baby_val, parent_val, min_range, max_range, invert=False):
    """
    Calculates similarity percentage between baby and parent feature values.
    Higher percentage means more similar.
    min_range, max_range: Expected min/max values for this feature across population.
    invert: If True, higher difference means higher similarity (e.g., for ratios, or if values are "dissimilarity" metrics)
            (For Euclidean distances, typically False. For ratios, sometimes true if ratios should be close to 1.)
    """
    # Normalize values to 0-1 relative to the expected population range
    norm_baby = normalize_feature(baby_val, min_range, max_range)
    norm_parent = normalize_feature(parent_val, min_range, max_range)

    # Calculate absolute difference between normalized values
    difference = abs(norm_baby - norm_parent)
    
    # Convert difference to similarity: 0 difference = 100% similarity, 1 difference = 0% similarity
    similarity = (1 - difference) * 100
    
    return similarity

# --- Population-based ranges for normalization (example values, adjust as needed) ---
FEATURE_RANGES = {
    'nose_length': {'min': 50, 'max': 100},
    'nose_width': {'min': 20, 'max': 60},
    'lip_thickness_upper': {'min': 5, 'max': 30},
    'lip_thickness_lower': {'min': 5, 'max': 35},
    'mouth_width': {'min': 50, 'max': 120},
    'left_eye_width': {'min': 20, 'max': 50},
    'right_eye_width': {'min': 20, 'max': 50},
    'inter_eye_distance': {'min': 40, 'max': 100},
    'chin_height': {'min': 30, 'max': 80},
    # Kept these here in case they become relevant later, even if not weighted for overall score initially
    'eye_aspect_ratio': {'min': 0.15, 'max': 0.35}, 
    'mouth_aspect_ratio': {'min': 0.3, 'max': 0.8},
    'face_width': {'min': 100, 'max': 250},
    'face_height': {'min': 150, 'max': 300},
}


def calculate_face_similarity(baby_measurements, parent_measurements):
    """
    Calculates similarity between baby and parent based on various facial features.
    Returns a dictionary of individual feature similarities and an overall weighted similarity.
    """
    if not baby_measurements or not parent_measurements:
        print("  Warning: Missing baby or parent measurements for similarity calculation. Returning 0.")
        return {"overall_similarity": 0, "feature_similarities": {}}

    feature_similarities = {}
    weighted_sum = 0
    total_effective_weight = 0

    for feature_key, weight in FEATURE_WEIGHTS.items():
        if feature_key in baby_measurements and feature_key in parent_measurements and \
           feature_key in FEATURE_RANGES:
            
            baby_val = baby_measurements[feature_key]
            parent_val = parent_measurements[feature_key]
            ranges = FEATURE_RANGES[feature_key]

            similarity = calculate_similarity_percentage(
                baby_val, parent_val, ranges['min'], ranges['max']
            )
            feature_similarities[feature_key] = similarity
            
            weighted_sum += similarity * weight
            total_effective_weight += weight
        else:
            feature_similarities[feature_key] = 0 # Placeholder for missing data
            # print(f"  Warning: Feature '{feature_key}' data or range missing for similarity calculation.")
            
    # --- TEMPORARY DEBUG PRINT ---
    print(f"  Debug: weighted_sum = {weighted_sum}, total_effective_weight = {total_effective_weight}")
    # --- END DEBUG PRINT ---

    overall_weighted_similarity = weighted_sum / total_effective_weight if total_effective_weight > 0 else 0

    return {
        "overall_similarity": overall_weighted_similarity,
        "feature_similarities": feature_similarities
    }

# --- Main execution ---

if __name__ == "__main__":
    print("Running face analysis module...")

    if os.path.exists(ANALYSIS_RESULTS_CSV):
        try:
            df_results = pd.read_csv(ANALYSIS_RESULTS_CSV, parse_dates=['timestamp'])
            print(f"Loaded existing results from {ANALYSIS_RESULTS_CSV}.")
        except Exception as e:
            print(f"Error loading {ANALYSIS_RESULTS_CSV}: {e}. Starting with an empty DataFrame.")
            df_results = pd.DataFrame()
    else:
        df_results = pd.DataFrame()
        print(f"'{ANALYSIS_RESULTS_CSV}' not found. A new one will be created.")

    # Get paths for parents' photos (assuming one photo each for mother and father)
    mother_img_path = None
    mother_dir = os.path.join(CAPTURED_FACES_BASE_DIR, "mother")
    if os.path.exists(mother_dir):
        mother_photos = sorted([f for f in os.listdir(mother_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if mother_photos:
            mother_img_path = os.path.join(mother_dir, mother_photos[0])
            print(f"Using mother's photo: {mother_img_path}")
        else:
            print("No mother's photo found in 'captured_faces/mother/'. Skipping mother comparison.")
    else:
        print("Directory 'captured_faces/mother/' not found. Skipping mother comparison.")

    father_img_path = None
    father_dir = os.path.join(CAPTURED_FACES_BASE_DIR, "father")
    if os.path.exists(father_dir):
        father_photos = sorted([f for f in os.listdir(father_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if father_photos:
            father_img_path = os.path.join(father_dir, father_photos[0])
            print(f"Using father's photo: {father_img_path}")
        else:
            print("No father's photo found in 'captured_faces/father/'. Skipping father comparison.")
    else:
        print("Directory 'captured_faces/father/' not found. Skipping father comparison.")

    # Process parent photos
    mother_landmarks, _ = get_face_landmarks(mother_img_path) if mother_img_path else (None, None)
    father_landmarks, _ = get_face_landmarks(father_img_path) if father_img_path else (None, None)

    mother_measurements = get_feature_measurements(mother_landmarks) if mother_landmarks else {}
    father_measurements = get_feature_measurements(father_landmarks) if father_landmarks else {}

    # Get paths for baby photos (all photos in the baby directory)
    baby_dir = os.path.join(CAPTURED_FACES_BASE_DIR, "baby")
    if os.path.exists(baby_dir):
        baby_photos = sorted([f for f in os.listdir(baby_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not baby_photos:
            print("No baby photos found in 'captured_faces/baby/'. No analysis to perform.")
    else:
        print("Directory 'captured_faces/baby/' not found. No analysis to perform.")
        baby_photos = []

    # Analyze each baby photo
    new_results_rows = []
    for baby_filename in baby_photos:
        baby_img_path = os.path.join(baby_dir, baby_filename)
        print(f"\nAnalyzing baby photo: {baby_filename}")

        baby_landmarks, _ = get_face_landmarks(baby_img_path)
        if not baby_landmarks:
            print(f"  Skipping {baby_filename}: Could not detect face or landmarks.")
            continue

        baby_measurements = get_feature_measurements(baby_landmarks)
        
        # Extract timestamp from filename (e.g., baby_YYYYMMDD_HHMMSS.jpg)
        try:
            timestamp_str = baby_filename.split('_')[1] + '_' + baby_filename.split('_')[2].split('.')[0]
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except (IndexError, ValueError):
            print(f"  Warning: Could not parse timestamp from {baby_filename}. Using current time.")
            timestamp = datetime.now()

        # Calculate similarities
        mother_results = calculate_face_similarity(baby_measurements, mother_measurements)
        father_results = calculate_face_similarity(baby_measurements, father_measurements)

        row_data = {
            'timestamp': timestamp,
            'baby_filename': baby_filename,
            'overall_similarity_to_mother_weighted': mother_results['overall_similarity'],
            'overall_similarity_to_father_weighted': father_results['overall_similarity']
        }

        # Add individual feature similarities to the row
        for feature_key, sim_val in mother_results['feature_similarities'].items():
            row_data[f'sim_mother_{feature_key}'] = sim_val
        for feature_key, sim_val in father_results['feature_similarities'].items():
            row_data[f'sim_father_{feature_key}'] = sim_val
        
        new_results_rows.append(row_data)

    if new_results_rows:
        df_new_results = pd.DataFrame(new_results_rows)
        
        # Append new results, avoid duplicates if re-running on same files
        df_results = pd.concat([df_results, df_new_results]).drop_duplicates(subset=['timestamp', 'baby_filename'], keep='last').sort_values('timestamp').reset_index(drop=True)
        
        df_results.to_csv(ANALYSIS_RESULTS_CSV, index=False)
        print(f"\nAnalysis complete. Results saved to {ANALYSIS_RESULTS_CSV}.")
    else:
        print("\nNo new analysis results to save.")

    print("Analysis module finished.")
