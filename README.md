# DNA Recognition by Facial Resemblance

A Python-based system that quantifies visual facial resemblance between a baby and its parents using facial landmark analysis to suggest inherited features. This project serves as a proof-of-concept for exploring genetic traits through computer vision.

## 🌟 Features

  * **Facial Landmark Detection:** Accurately identifies 68 facial landmarks using the `face_recognition` library.
  * **Geometric Feature Extraction:** Calculates various facial measurements such as nose length, eye width, mouth width, inter-eye distance, and chin height.
  * **Weighted Similarity Scoring:** Computes a comprehensive resemblance score for the baby to both the mother and the father, based on a customizable set of weighted facial features.
  * **Detailed CSV Report Generation:** Stores all raw measurements, individual feature similarities, and overall weighted resemblance scores in a structured CSV file (`analysis_results.csv`).
  * **Trend Analysis Visualization:** Generates a line graph (`overall_similarity_trend.png`) to visualize how the baby's overall resemblance to each parent evolves over multiple analyses.
  * **Feature-Specific Resemblance Breakdown:** Provides a textual report detailing the baby's resemblance to each parent for individual facial features.
  * **(Placeholder for future enhancement):** Designed to support visual comparison images of cropped facial features (eyes, nose, mouth, chin) if the analysis module is extended to save these.

## 💡 How It Works

1.  **Image Input:** The system takes photographs of the mother, father, and one or more baby photos as input.
2.  **Face Detection & Landmark Identification:** For each image, it detects faces and maps 68 distinct facial landmarks (e.g., corners of eyes, points along the nose, outline of the mouth).
3.  **Feature Measurement:** Based on these landmarks, various geometric measurements (distances, ratios) are calculated for specific facial features.
4.  **Normalization & Similarity:** These measurements are normalized against predefined ranges to ensure consistency and then compared between the baby and each parent. A similarity percentage is derived for each feature.
5.  **Weighted Aggregation:** Individual feature similarities are combined into an overall weighted resemblance score, giving more importance to certain features (e.g., nose, eyes) as defined by `FEATURE_WEIGHTS`.
6.  **Reporting & Visualization:** The results are saved to a CSV file, and a report is generated, including a trend graph and a detailed breakdown of feature resemblances.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

  * Python 3.8+
  * `pip` (Python package installer)
  * `git` (for cloning the repository)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ChandanM123456/DNA-Recognition-by-Facial-Resemblance.git
    cd DNA-Recognition-by-Facial-Resemblance
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Prepare Your Photos

  * Create the following directory structure inside your project folder if it doesn't already exist:

    ```
    .
    ├── captured_faces/
    │   ├── baby/
    │   ├── father/
    │   └── mother/
    ├── analysis_module.py
    ├── report.py
    ├── requirements.txt
    └── README.md
    ```

  * Place **one clear photo** of the mother in `captured_faces/mother/`.

      * Example: `captured_faces/mother/mother_20250728_200624.jpg`

  * Place **one clear photo** of the father in `captured_faces/father/`.

      * Example: `captured_faces/father/father_20250728_200721.jpg`

  * Place **one or more clear photos** of the baby in `captured_faces/baby/`.

      * Example: `captured_faces/baby/baby_20250728_200523.jpg`
      * The system will analyze all photos found in the `baby/` directory.

    **Important Photo Tips:**

      * Use clear, well-lit photos.
      * Faces should be frontal and not obstructed.
      * Avoid extreme angles, strong shadows, or blurry images for best results.
      * For the initial run, using the example filenames shown in the console output might be helpful to mirror the provided log, though any `.jpg`, `.jpeg`, or `.png` files will work.

## 🏃 Running the Project

The project consists of two main scripts: `analysis_module.py` (to perform the facial analysis and save data) and `report.py` (to generate reports and visualizations from the saved data).

1.  **Run the Analysis Module:**
    This script will process the photos and save the resemblance data to `analysis_results.csv`.

    ```bash
    python analysis_module.py
    ```

    You will see debug output in the console indicating the processing of each baby photo and the weighted sum/total effective weight calculation.

2.  **Run the Report Generation Module:**
    This script reads the data from `analysis_results.csv` and generates a text-based report in the console, along with an image file (`overall_similarity_trend.png`) and (if feature image cropping is implemented) individual feature comparison images in the `reports/` directory.

    ```bash
    python report.py
    ```

    The console will display the latest facial resemblance report, and image files will be saved in the `reports/` directory.

## 📂 Project Structure

```
.
├── captured_faces/
│   ├── baby/             # Put baby photos here
│   ├── father/           # Put father's photo here
│   └── mother/           # Put mother's photo here
├── reports/              # Generated reports (charts, comparison images) will be saved here
├── analysis_module.py    # Core logic for facial analysis and data generation
├── report.py             # Generates textual and visual reports from analysis data
├── requirements.txt      # List of Python dependencies
└── README.md             # This file
```
