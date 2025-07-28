import cv2
import os
import datetime
import time

def setup_directories(base_dir="captured_faces"):
    """Creates subdirectories for baby, mother, and father."""
    baby_dir = os.path.join(base_dir, "baby")
    mother_dir = os.path.join(base_dir, "mother")
    father_dir = os.path.join(base_dir, "father")

    os.makedirs(baby_dir, exist_ok=True)
    os.makedirs(mother_dir, exist_ok=True)
    os.makedirs(father_dir, exist_ok=True)
    return baby_dir, mother_dir, father_dir

def capture_session(person_name, output_dir, camera_index=0, num_pics=10, interval_sec=5):
    """
    Captures a specified number of photos for a given person.
    Assumes the correct person is in front of the camera during the session.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index} for {person_name}.")
        return

    print(f"\n--- Starting capture session for: {person_name.upper()} ---")
    print(f"Please ensure {person_name} is clearly visible in the camera frame.")
    print(f"Capturing {num_pics} photos with {interval_sec} second intervals...")
    print("Press 'q' at any time to stop the current session.")

    captured_count = 0
    start_time = time.time()

    while captured_count < num_pics:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame from camera for {person_name}. Retrying...")
            time.sleep(1)
            continue

        cv2.imshow(f"Capturing: {person_name.upper()} - {captured_count + 1}/{num_pics}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Session for {person_name} interrupted by user.")
            break

        if time.time() - start_time >= interval_sec:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"{person_name}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            captured_count += 1
            start_time = time.time() # Reset timer for next capture

    cap.release()
    cv2.destroyAllWindows()
    print(f"--- Capture session for {person_name} finished. Captured {captured_count} photos. ---")


if __name__ == "__main__":
    baby_folder, mother_folder, father_folder = setup_directories()

    # Capture 10 photos for each person with 5-second intervals.
    # You'll need to run these commands sequentially and ensure the correct person
    # is in front of the camera for each session.

    # 1. Capture Baby Photos
    capture_session("baby", baby_folder, num_pics=10, interval_sec=5)
    time.sleep(2) # Small pause between sessions

    # 2. Capture Mother Photos
    capture_session("mother", mother_folder, num_pics=10, interval_sec=5)
    time.sleep(2) # Small pause between sessions

    # 3. Capture Father Photos
    capture_session("father", father_folder, num_pics=10, interval_sec=5)

    print("\nAll capture sessions completed.")
    print("Now proceed to the analysis_module.py to process these images.")
