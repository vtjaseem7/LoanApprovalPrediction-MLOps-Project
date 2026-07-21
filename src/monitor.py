import pandas as pd

LOG_FILE = "logs/predictions.csv"

df = pd.read_csv(LOG_FILE)

print("=" * 50)
print("LOAN APPROVAL MONITORING REPORT")
print("=" * 50)

# Total Predictions

total_predictions = len(df)

print(f"\nTotal Predictions: {total_predictions}")

# Class Distribution

print("\nClass Distribution")

print(df["result"].value_counts())

# Average Confidence

avg_conf = df["probability"].mean()

print(
    f"\nAverage Confidence: "
    f"{avg_conf:.4f}"
)

# Low Confidence Predictions

low_conf = df[
    df["probability"] < 0.60
]

print(
    f"\nLow Confidence Predictions: "
    f"{len(low_conf)}"
)

# Confidence Threshold Analysis

print(
    "\nConfidence Threshold Analysis"
)

thresholds = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

for threshold in thresholds:

    count = len(
        df[
            df["probability"]
            >= threshold
        ]
    )

    print(
        f"Threshold {threshold}: "
        f"{count} predictions"
    )

print("\nMonitoring Complete")