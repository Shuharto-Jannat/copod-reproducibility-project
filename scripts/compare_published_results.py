from pathlib import Path

import pandas as pd


# COPOD values reported in Tables I and II of the paper.
published_results = [
    # MAT datasets
    ["MAT", "arrhythmia", 0.8021, 0.4727],
    ["MAT", "breastw", 0.9936, 0.9877],
    ["MAT", "cardio", 0.8974, 0.5793],
    ["MAT", "ionosphere", 0.8307, 0.7194],
    ["MAT", "lympho", 0.9935, 0.8936],
    ["MAT", "mammography", 0.8942, 0.4290],
    ["MAT", "optdigits", 0.7328, 0.0532],
    ["MAT", "pima", 0.6638, 0.5407],
    ["MAT", "satellite", 0.6612, 0.5854],
    ["MAT", "satimage-2", 0.9852, 0.8599],
    ["MAT", "shuttle", 0.9980, 0.9807],
    ["MAT", "speech", 0.4845, 0.0195],
    ["MAT", "wbc", 0.9747, 0.7827],
    ["MAT", "wine", 0.9490, 0.6082],

    # ARFF datasets
    ["ARFF", "Arrhythmia", 0.7618, 0.7519],
    ["ARFF", "Cardiotocography", 0.6370, 0.3778],
    ["ARFF", "HeartDisease", 0.6728, 0.6400],
    ["ARFF", "Hepatitis", 0.8432, 0.5849],
    ["ARFF", "InternetAds", 0.6763, 0.5102],
    ["ARFF", "Ionosphere", 0.8219, 0.5302],
    ["ARFF", "KDDCup99", 0.9972, 0.5667],
    ["ARFF", "Lymphography", 0.9985, 0.4637],
    ["ARFF", "Pima", 0.6466, 0.7031],
    ["ARFF", "Shuttle", 0.8795, 0.2389],
    ["ARFF", "SpamBase", 0.7007, 0.9817],
    ["ARFF", "Stamps", 0.9388, 0.0854],
    ["ARFF", "Waveform", 0.7865, 0.0785],
    ["ARFF", "WBC", 0.9904, 0.8381],
    ["ARFF", "WDBC", 0.9826, 0.8407],
    ["ARFF", "WPBC", 0.5465, 0.2438],
]

published_df = pd.DataFrame(
    published_results,
    columns=["Format", "Data", "Paper_ROC", "Paper_AP"]
)

mat_roc = pd.read_csv(
    "results/reproduction_mat_14_10_trials/roc.csv"
)
mat_ap = pd.read_csv(
    "results/reproduction_mat_14_10_trials/ap.csv"
)
arff_roc = pd.read_csv(
    "results/reproduction_arff_16_10_trials/roc.csv"
)
arff_ap = pd.read_csv(
    "results/reproduction_arff_16_10_trials/ap.csv"
)

reproduced_rows = []

for _, row in published_df.iterrows():
    data_format = row["Format"]
    dataset = row["Data"]

    if data_format == "MAT":
        roc_table = mat_roc
        ap_table = mat_ap
    else:
        roc_table = arff_roc
        ap_table = arff_ap

    reproduced_roc = roc_table.loc[
        roc_table["Data"] == dataset, "COD"
    ].iloc[0]

    reproduced_ap = ap_table.loc[
        ap_table["Data"] == dataset, "COD"
    ].iloc[0]

    reproduced_rows.append(
        [
            data_format,
            dataset,
            row["Paper_ROC"],
            reproduced_roc,
            reproduced_roc - row["Paper_ROC"],
            abs(reproduced_roc - row["Paper_ROC"]),
            row["Paper_AP"],
            reproduced_ap,
            reproduced_ap - row["Paper_AP"],
            abs(reproduced_ap - row["Paper_AP"]),
        ]
    )

comparison_df = pd.DataFrame(
    reproduced_rows,
    columns=[
        "Format",
        "Data",
        "Paper_ROC",
        "Reproduced_ROC",
        "ROC_Difference",
        "ROC_Absolute_Difference",
        "Paper_AP",
        "Reproduced_AP",
        "AP_Difference",
        "AP_Absolute_Difference",
    ]
)

output_directory = Path("results/comparison")
output_directory.mkdir(parents=True, exist_ok=True)

comparison_df.to_csv(
    output_directory / "copod_published_vs_reproduced.csv",
    index=False
)

summary_df = pd.DataFrame(
    {
        "Metric": [
            "Mean ROC-AUC",
            "Mean Average Precision",
            "Mean absolute ROC-AUC difference",
            "Mean absolute Average Precision difference",
            "Maximum absolute ROC-AUC difference",
            "Maximum absolute Average Precision difference",
        ],
        "Paper": [
            comparison_df["Paper_ROC"].mean(),
            comparison_df["Paper_AP"].mean(),
            None,
            None,
            None,
            None,
        ],
        "Reproduced": [
            comparison_df["Reproduced_ROC"].mean(),
            comparison_df["Reproduced_AP"].mean(),
            None,
            None,
            None,
            None,
        ],
        "Value": [
            None,
            None,
            comparison_df["ROC_Absolute_Difference"].mean(),
            comparison_df["AP_Absolute_Difference"].mean(),
            comparison_df["ROC_Absolute_Difference"].max(),
            comparison_df["AP_Absolute_Difference"].max(),
        ],
    }
)

summary_df.to_csv(
    output_directory / "copod_comparison_summary.csv",
    index=False
)

largest_roc_differences = comparison_df.nlargest(
    5, "ROC_Absolute_Difference"
)

largest_ap_differences = comparison_df.nlargest(
    5, "AP_Absolute_Difference"
)

print("Comparison completed.")
print()
print("Paper mean ROC-AUC:", round(comparison_df["Paper_ROC"].mean(), 4))
print(
    "Reproduced mean ROC-AUC:",
    round(comparison_df["Reproduced_ROC"].mean(), 4)
)
print("Paper mean AP:", round(comparison_df["Paper_AP"].mean(), 4))
print(
    "Reproduced mean AP:",
    round(comparison_df["Reproduced_AP"].mean(), 4)
)
print(
    "Mean absolute ROC-AUC difference:",
    round(comparison_df["ROC_Absolute_Difference"].mean(), 4)
)
print(
    "Mean absolute AP difference:",
    round(comparison_df["AP_Absolute_Difference"].mean(), 4)
)

print()
print("Five largest ROC-AUC differences:")
print(
    largest_roc_differences[
        [
            "Format",
            "Data",
            "Paper_ROC",
            "Reproduced_ROC",
            "ROC_Difference",
        ]
    ].to_string(index=False)
)

print()
print("Five largest Average Precision differences:")
print(
    largest_ap_differences[
        [
            "Format",
            "Data",
            "Paper_AP",
            "Reproduced_AP",
            "AP_Difference",
        ]
    ].to_string(index=False)
)