import os
import pandas as pd
import numpy as np

# Load your raw downloaded dataset
raw_df = pd.read_csv("StudentPerformanceFactors.csv")

# Create output data directory if missing
os.makedirs("data", exist_ok=True)

# Process & Map features to the required app schema
processed_df = pd.DataFrame()

# Direct mapped columns from your dataset
processed_df["study_hours"] = raw_df["Hours_Studied"]
processed_df["attendance_percentage"] = raw_df["Attendance"]
processed_df["previous_exam_score"] = raw_df["Previous_Scores"]
processed_df["sleep_hours"] = raw_df["Sleep_Hours"]
processed_df["final_exam_score"] = raw_df["Exam_Score"]

# Synthesize missing numerical features from existing data for app compatibility
# (Assignments estimated from Attendance & Previous Scores)
processed_df["assignment_score"] = (
    0.6 * raw_df["Previous_Scores"] + 0.4 * raw_df["Attendance"]
).round(1)

# (Participation estimated from Attendance)
processed_df["class_participation"] = (
    raw_df["Attendance"] * 0.95
).clip(0, 100).round(1)

# Reorder exactly to app schema
FEATURE_COLUMNS = [
    "study_hours",
    "attendance_percentage",
    "assignment_score",
    "previous_exam_score",
    "sleep_hours",
    "class_participation",
    "final_exam_score"
]

processed_df = processed_df[FEATURE_COLUMNS]

# Save directly to the required app path
output_path = "data/original_student_data.csv"
processed_df.to_csv(output_path, index=False)

print(f"✅ Successfully converted {len(processed_df)} rows!")
print(f"Saved to: {output_path}")