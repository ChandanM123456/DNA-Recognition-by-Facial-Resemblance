import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import math

# Define facial feature groups based on dlib's 68-point facial landmarks
# These are the KEYS into the dictionary returned by face_recognition.face_landmarks
FACIAL_FEATURE_NAMES = [
    "left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip",
    "left_eye", "right_eye", "top_lip", "bottom_lip", "chin_jaw"
]

def euclidean_distance(point1, point2):
    """Calculates Euclidean distance between two points (x, y)."""
    if point1 is None or point2 is None:
        return 0 # Handle missing points gracefully
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_feature_distances(landmarks_dict):
    """
    Calculates various distances and ratios for specific facial features
    from the dictionary provided by face_recognition.face_landmarks.
    Returns a dictionary of feature measurements.
    """
    feature_metrics = {}

    # Eye distances
    if 'left_eye' in landmarks_dict and len(landmarks_dict['left_eye']) >= 6:
        le_outer = landmarks_dict['left_eye'][0]
        le_inner = landmarks_dict['left_eye'][3]
        feature_metrics['eye_width_left'] = euclidean_distance(le_outer, le_inner)
    if 'right_eye' in landmarks_dict and len(landmarks_dict['right_eye']) >= 6:
        re_outer = landmarks_dict['right_eye'][0]
        re_inner = landmarks_dict['right_eye'][3]
        feature_metrics['eye_width_right'] = euclidean_distance(re_outer, re_inner)
    
    # Inter-eye distance (assuming both eyes are detected)
    if 'left_eye' in landmarks_dict and 'right_eye' in landmarks_dict and \
       len(landmarks_dict['left_eye']) >= 6 and len(landmarks_dict['right_eye']) >= 6:
        feature_metrics['inter_eye_distance'] = euclidean_distance(landmarks_dict['left_eye'][3], landmarks_dict['right_eye'][0])


    # Nose distances
    if 'nose_tip' in landmarks_dict and len(landmarks_dict['nose_tip']) >= 5:
        nose_bottom = landmarks_dict['nose_tip'][3]
        if 'nose_bridge' in landmarks_dict and len(landmarks_dict['nose_bridge']) >= 4:
            nose_top_bridge = landmarks_dict['nose_bridge'][0]
            feature_metrics['nose_length'] = euclidean_distance(nose_top_bridge, nose_bottom)
        feature_metrics['nose_width'] = euclidean_distance(landmarks_dict['nose_tip'][0], landmarks_dict['nose_tip'][4])

    # Mouth distances
    if 'top_lip' in landmarks_dict and 'bottom_lip' in landmarks_dict and \
       len(landmarks_dict['top_lip']) >= 12 and len(landmarks_dict['bottom_lip']) >= 12: # Ensure enough points
        mouth_left = landmarks_dict['top_lip'][0]
        mouth_right = landmarks_dict['top_lip'][6]
        feature_metrics['mouth_width'] = euclidean_distance(mouth_left, mouth_right)
        
        # Simplified thickness: top of upper lip to bottom of upper lip
        feature_metrics['lip_thickness_upper'] = euclidean_distance(landmarks_dict['top_lip'][3], landmarks_dict['top_lip'][9])
        feature_metrics['lip_thickness_lower'] = euclidean_distance(landmarks_dict['bottom_lip'][3], landmarks_dict['bottom_lip'][9])


    # Jawline / Chin (simplified - usually involves curvature analysis)
    if 'chin_jaw' in landmarks_dict and len(landmarks_dict['chin_jaw']) >= 17:
        jaw_left_end = landmarks_dict['chin_jaw'][0]
        jaw_right_end = landmarks_dict['chin_jaw'][16]
        chin_tip = landmarks_dict['chin_jaw'][8]
        feature_metrics['jaw_width'] = euclidean_distance(jaw_left_end, jaw_right_end)
        feature_metrics['chin_height'] = euclidean_distance(landmarks_dict['nose_tip'][3], chin_tip) # Nose tip to chin tip as a proxy

    return feature_metrics

def compare_feature_metrics(baby_metrics, parent_metrics):
    """
    Compares baby's feature metrics to parent's average metrics.
    Returns a dictionary of similarity percentages for each feature.
    """
    feature_similarities = {}
    for feature_key, baby_val in baby_metrics.items():
        if feature_key in parent_metrics and parent_metrics[feature_key] is not None:
            parent_val = parent_metrics[feature_key]
            if parent_val == 0: # Avoid division by zero if parent metric is zero
                feature_similarities[feature_key] = 0.0
                continue
            
            # Use a robust similarity metric: 1 - |A-B| / ((A+B)/2) - Percentage Difference
            # Scale to 100%
            if (baby_val + parent_val) != 0:
                diff_ratio = abs(baby_val - parent_val) / ((baby_val + parent_val) / 2)
                similarity_percent = max(0, 100 * (1 - diff_ratio))
                feature_similarities[feature_key] = round(similarity_percent, 2)
            else:
                feature_similarities[feature_key] = 0.0 # Both are zero
        else:
            feature_similarities[feature_key] = None # No parent data for this feature

    return feature_similarities


def load_and_process_faces(folder_path):
    """
    Loads images, extracts face encodings and feature landmarks for all valid faces.
    Returns a list of dictionaries: [{'encoding': enc, 'landmarks_dict': lms_dict, 'timestamp': ts, 'filename': fn}]
    """
    processed_faces = []
    print(f"Loading and processing faces from {folder_path}...")
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image)
                
                if not face_locations:
                    print(f"  No face found in {filename}, skipping.")
                    continue

                # Assuming only one prominent face per image for simplicity
                face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                face_landmarks_list_raw = face_recognition.face_landmarks(image, [face_locations[0]])

                if face_landmarks_list_raw:
                    # face_landmarks_list_raw[0] is the dictionary of landmark points
                    landmarks_dict = face_landmarks_list_raw[0] 
                    
                    # Ensure all expected feature keys are present in the returned dictionary,
                    # even if some sub-points are missing (handled by calculate_feature_distances)
                    # We don't need to filter by FACIAL_FEATURE_NAMES here, as landmarks_dict already has them
                    
                    # Extract timestamp from filename (e.g., person_YYYYMMDD_HHMMSS.jpg)
                    timestamp = None
                    try:
                        ts_str = filename.split('_')[-1].split('.')[0]
                        timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    except ValueError:
                        print(f"  Warning: Could not parse full timestamp from {filename}. Using None or partial date.")
                        # Attempt to parse just the date part if time fails
                        try:
                            date_str = filename.split('_')[-1].split('_')[0].split('.')[0] # e.g. 20250728 from baby_20250728_195000.jpg
                            timestamp = datetime.strptime(date_str, "%Y%m%d")
                        except ValueError:
                             pass # Still None if date parsing fails

                    processed_faces.append({
                        'encoding': face_encoding,
                        'landmarks_dict': landmarks_dict, # Use the directly provided dictionary
                        'timestamp': timestamp,
                        'filename': filename
                    })
                else:
                    print(f"  No landmarks detected for face in {filename}, skipping.")

            except Exception as e:
                print(f"  Error processing {filename}: {e}. This might indicate a problem with the image or face detection.")
    return processed_faces

def aggregate_feature_metrics(processed_faces):
    """
    Calculates average feature metrics from a list of processed faces.
    """
    all_metrics_lists = defaultdict(list)
    for face_data in processed_faces:
        metrics = calculate_feature_distances(face_data['landmarks_dict'])
        for feature, value in metrics.items():
            all_metrics_lists[feature].append(value)

    # Calculate mean, but handle cases where a feature might not have any data
    avg_metrics = {}
    for feature, values in all_metrics_lists.items():
        if values: # Ensure there are values to average
            avg_metrics[feature] = np.mean(values)
        else:
            avg_metrics[feature] = None # Indicate no data for this feature

    return avg_metrics

def main_analysis(base_dir="captured_faces"):
    baby_data = load_and_process_faces(os.path.join(base_dir, "baby"))
    mother_data = load_and_process_faces(os.path.join(base_dir, "mother"))
    father_data = load_and_process_faces(os.path.join(base_dir, "father"))

    if not baby_data:
        print("No baby photos found for analysis. Please capture photos first using capture_module.py.")
        return

    if not mother_data:
        print("Warning: No mother photos found. Cannot perform similarity analysis to mother.")
    if not father_data:
        print("Warning: No father photos found. Cannot perform similarity analysis to father.")

    avg_mother_metrics = aggregate_feature_metrics(mother_data) if mother_data else {}
    avg_father_metrics = aggregate_feature_metrics(father_data) if father_data else {}

    results_df_rows = []
    
    # Store aggregated feature similarities for the final average
    # These will be lists of similarities for each specific feature across all baby photos
    all_baby_feature_sims_to_mother = defaultdict(list)
    all_baby_feature_sims_to_father = defaultdict(list)

    print("\n--- Performing Detailed Feature-Level Analysis ---")
    for baby_entry in baby_data:
        baby_metrics = calculate_feature_distances(baby_entry['landmarks_dict'])
        row_data = {
            'timestamp': baby_entry['timestamp'],
            'baby_filename': baby_entry['filename']
        }
        
        # Initialize scores for this specific baby photo
        current_baby_sim_to_mother_features = []
        current_baby_sim_to_father_features = []

        feature_comparison_mother = compare_feature_metrics(baby_metrics, avg_mother_metrics)
        feature_comparison_father = compare_feature_metrics(baby_metrics, avg_father_metrics)

        # Append feature similarities to the main DataFrame row and aggregate for overall average
        for feature_metric_key in baby_metrics.keys(): # Iterate through metrics that were actually calculated for baby
            mother_sim = feature_comparison_mother.get(feature_metric_key, None)
            father_sim = feature_comparison_father.get(feature_metric_key, None)

            row_data[f'sim_mother_{feature_metric_key}'] = mother_sim
            row_data[f'sim_father_{feature_metric_key}'] = father_sim

            if mother_sim is not None:
                current_baby_sim_to_mother_features.append(mother_sim)
                all_baby_feature_sims_to_mother[feature_metric_key].append(mother_sim)
            if father_sim is not None:
                current_baby_sim_to_father_features.append(father_sim)
                all_baby_feature_sims_to_father[feature_metric_key].append(father_sim)

        # Calculate an overall weighted average for this specific baby picture
        # You can assign weights to features here (e.g., eyes more important than jaw)
        if current_baby_sim_to_mother_features:
            avg_sim_mother = np.mean(current_baby_sim_to_mother_features)
            row_data['overall_similarity_to_mother'] = round(avg_sim_mother, 2)
        else:
            row_data['overall_similarity_to_mother'] = None

        if current_baby_sim_to_father_features:
            avg_sim_father = np.mean(current_baby_sim_to_father_features)
            row_data['overall_similarity_to_father'] = round(avg_sim_father, 2)
        else:
            row_data['overall_similarity_to_father'] = None

        results_df_rows.append(row_data)

    df_results = pd.DataFrame(results_df_rows)
    
    # Calculate overall average resemblance across all baby photos for final summary
    final_avg_mother_sim = df_results['overall_similarity_to_mother'].mean() if not df_results['overall_similarity_to_mother'].empty else None
    final_avg_father_sim = df_results['overall_similarity_to_father'].mean() if not df_results['overall_similarity_to_father'].empty else None

    print("\n--- Detailed Feature Similarity Results (for each baby photo) ---")
    print(df_results) # Print full dataframe

    print(f"\nOverall Baby Resemblance (Average across all baby photos):")
    if final_avg_mother_sim is not None:
        print(f"  To Mother: {final_avg_mother_sim:.2f}%")
    if final_avg_father_sim is not None:
        print(f"  To Father: {final_avg_father_sim:.2f}%")

    if final_avg_mother_sim is not None and final_avg_father_sim is not None:
        if final_avg_mother_sim > final_avg_father_sim + 10: # Increased threshold for "more identical"
            print("Conclusion: The baby's face appears overall significantly more identical to the Mother.")
        elif final_avg_father_sim > final_avg_mother_sim + 10: # Increased threshold for "more identical"
            print("Conclusion: The baby's face appears overall significantly more identical to the Father.")
        else:
            print("Conclusion: The baby's face appears to be a relatively balanced mix of both parents, or not strongly leaning towards one.")

    # Identify "Unique Features" based on average similarities
    print("\n--- Analysis of Prominent Feature Resemblances and Uniqueness ---")
    
    # Calculate average feature similarities across all baby photos for consolidated report
    avg_feature_sims_mother = {feat: np.mean(all_baby_feature_sims_to_mother[feat]) 
                               for feat in all_baby_feature_sims_to_mother if all_baby_feature_sims_to_mother[feat]}
    avg_feature_sims_father = {feat: np.mean(all_baby_feature_sims_to_father[feat]) 
                               for feat in all_baby_feature_sims_to_father if all_baby_feature_sims_to_father[feat]}

    # Collect unique insights for the report
    feature_insights = []

    # Iterate through all possible feature keys that we *might* calculate metrics for
    all_possible_feature_metric_keys = set(avg_mother_metrics.keys()).union(set(avg_father_metrics.keys()))

    for feature_metric_key in all_possible_feature_metric_keys:
        mother_sim_avg = avg_feature_sims_mother.get(feature_metric_key, None)
        father_sim_avg = avg_feature_sims_father.get(feature_metric_key, None)

        if mother_sim_avg is None and father_sim_avg is None:
            continue # No data for this feature from either parent or baby

        feature_name_display = feature_metric_key.replace('_', ' ').capitalize()

        if mother_sim_avg is not None and father_sim_avg is not None:
            if mother_sim_avg > 75 and mother_sim_avg > father_sim_avg + 15: # Tunable thresholds
                feature_insights.append(f"- Baby's {feature_name_display} strongly resembles the Mother ({mother_sim_avg:.2f}% similarity).")
            elif father_sim_avg > 75 and father_sim_avg > mother_sim_avg + 15: # Tunable thresholds
                feature_insights.append(f"- Baby's {feature_name_display} strongly resembles the Father ({father_sim_avg:.2f}% similarity).")
            elif mother_sim_avg < 40 and father_sim_avg < 40 and (mother_sim_avg > 0 or father_sim_avg > 0): # Low similarity to both, but not zero
                 feature_insights.append(f"- Baby's {feature_name_display} shows a unique development or blend, with lower direct resemblance to either parent (Mother: {mother_sim_avg:.2f}%, Father: {father_sim_avg:.2f}%).")
            elif mother_sim_avg > 60 and father_sim_avg > 60 and abs(mother_sim_avg - father_sim_avg) < 10:
                 feature_insights.append(f"- Baby's {feature_name_display} appears to be a balanced mix of both parents (Mother: {mother_sim_avg:.2f}%, Father: {father_sim_avg:.2f}%).")
            else:
                 feature_insights.append(f"- Baby's {feature_name_display}: Mixed resemblance (Mother: {mother_sim_avg:.2f}%, Father: {father_sim_avg:.2f}%).")
        elif mother_sim_avg is not None: # Only mother data available
            feature_insights.append(f"- Baby's {feature_name_display}: Only Mother's data available, similarity: {mother_sim_avg:.2f}%.")
        elif father_sim_avg is not None: # Only father data available
            feature_insights.append(f"- Baby's {feature_name_display}: Only Father's data available, similarity: {father_sim_avg:.2f}%.")

    if feature_insights:
        for insight in feature_insights:
            print(insight)
    else:
        print("No specific feature insights could be generated with the available data.")


    # Save results to CSV for reporting module
    df_results.to_csv("analysis_results.csv", index=False)
    print("\nAnalysis results saved to analysis_results.csv")


if __name__ == "__main__":
    main_analysis()
