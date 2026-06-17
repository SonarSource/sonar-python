from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

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

def plot_feasibility_histogram(file_path: str = "current-analyzer-rules.md"):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    data = []

    # Regex captures: Rule Name, Status Column, and Feasibility Score
    row_pattern = re.compile(
        r'\|\s*\*\*([^*]+)\*\*\s*\|[^|]+\|[^|]+\|\s*([^|]+?)\s*\|\s*(\d+)/10\s*\|'
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = row_pattern.search(line)
            if match:
                status_str = match.group(2).strip()
                score = int(match.group(3))

                # Tag as Implemented if it contains the green check indicator
                status = "Already Implemented" if "✅" in status_str or "exists" in status_str else "Not Implemented"
                data.append({"Feasibility": score, "Status": status})

    if not data:
        print("No validation rules data extracted to plot.")
        return

    df_feasibility = pd.DataFrame(data)

    # Setup the plot figure
    fig, ax1 = plt.subplots(figsize=(13, 6.5))

    # Force zorder context rendering priority
    ax1.set_zorder(1)

    # Prepare data matrix matching discrete buckets 0-10 for stacked bar visualization
    all_scores = np.arange(11)
    implemented_counts = []
    not_implemented_counts = []

    for x in all_scores:
        subset = df_feasibility[df_feasibility["Feasibility"] == x]
        implemented_counts.append(np.sum(subset["Status"] == "Already Implemented"))
        not_implemented_counts.append(np.sum(subset["Status"] == "Not Implemented"))

    # 1. Generate Split Stacked Histograms on primary y-axis (ax1)
    # Plot 'Not Implemented' as the baseline layer
    ax1.bar(all_scores, not_implemented_counts, label="Not Implemented", color="#bdc3c7", edgecolor="black", width=0.8, zorder=2)

    # Plot 'Already Implemented' stacked directly on top of the 'Not Implemented' counts
    ax1.bar(all_scores, implemented_counts, bottom=not_implemented_counts, label="Already Implemented", color="#2ecc71", edgecolor="black", width=0.8, zorder=2)

    # Dynamically set the left axis limit so tall stacked bars never escape the view
    total_stacked_heights = np.array(not_implemented_counts) + np.array(implemented_counts)
    max_height = np.max(total_stacked_heights)
    ax1.set_ylim(0, max_height * 1.15)

    ax1.set_title("Distribution of Rule Quickfix Feasibility Scores & Status Context", fontsize=14, pad=15)
    ax1.set_xlabel("Estimated Feasibility (0 to 10)", fontsize=12)
    ax1.set_ylabel("Count of Rules (Stacked Bars)", fontsize=12)
    ax1.set_xticks(all_scores)
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)

    # 2. Calculate "Greater than or Equal to X" percentages
    total_rules = len(df_feasibility)
    gte_percentages = []
    for x in all_scores:
        count_gte = np.sum(df_feasibility["Feasibility"] >= x)
        percentage = (count_gte / total_rules) * 100
        gte_percentages.append(percentage)

    # 3. Plot the curve on a secondary y-axis (ax2)
    ax2 = ax1.twinx()
    ax2.set_zorder(2)

    # FIX: Added 'r' prefix to treat LaTeX strings as raw text and get rid of SyntaxWarnings
    ax2.plot(all_scores, gte_percentages, color="crimson", marker="o", linewidth=2.5, label=r"$\geq$ Feasibility X", zorder=3)
    ax2.set_ylabel(r"Percentage of Rules $\geq$ X (Line)", fontsize=12, color="crimson")
    ax2.tick_params(axis='y', labelcolor="crimson")
    ax2.set_ylim(-5, 115)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    # Loop to annotate a percentage label at each line coordinate point
    for x, y in zip(all_scores, gte_percentages):
        ax2.annotate(
            f"{int(round(y))}%",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="crimson",
            weight="semibold"
        )

    # FIX: Combine handles/labels and set the zorder on the generated legend object instead of via argument
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=1.0)
    leg.set_zorder(5) # This moves the legend instance safely on top of all visual lines

    plt.tight_layout()

    # Save the plot
    plt.savefig("feasibility_histogram.png", dpi=300)
    plt.close()

# Execute both plot utilities
plot_experiment_heatmap()
plot_feasibility_histogram()