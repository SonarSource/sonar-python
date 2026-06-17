from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_experiment_heatmap(base_dir: str = "experiment_results"):
    base_path = Path(base_dir)
    data = []

    # 1. Traverse directory to collect findings.json lengths
    for json_file in base_path.glob("*/*/findings.json"):
        prompt_type = json_file.parent.name
        model = json_file.parent.parent.name

        try:
            with open(json_file, "r") as f:
                findings_list = json.load(f)

            if isinstance(findings_list, list):
                data.append({
                    "Model": model,
                    "Prompt Type": prompt_type,
                    "Length": len(findings_list)
                })
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping broken file {json_file}: {e}")

    if not data:
        print("No valid data found to plot.")
        return

    # 2. Convert to DataFrame
    df = pd.DataFrame(data)

    # 3. Pivot table
    pivot_df = df.pivot(index="Model", columns="Prompt Type", values="Length")

    # Swap columns so 'short' comes before 'long'
    desired_order = ["short", "long"]
    existing_order = [c for c in desired_order if c in pivot_df.columns] + [c for c in pivot_df.columns if c not in desired_order]
    pivot_df = pivot_df.reindex(columns=existing_order)

    # 4. Generate the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt="g", cbar_kws={'label': 'Number of Findings'})

    plt.title("Experiment Findings Count Heatmap", fontsize=14, pad=15)
    plt.ylabel("Models", fontsize=12)
    plt.xlabel("Prompt Types", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # 5. Save the plot
    plt.savefig("experiment_heatmap.png", dpi=300)

plot_experiment_heatmap()