import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns

# Create main figure with 4 rows × 2 cols
fig, axes = plt.subplots(4, 2, figsize=(14, 22))
axes = axes.reshape(4, 2)
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14, 18))
gs = GridSpec(4, 2, width_ratios=[1, 1.5])  # Shrink scatter plots (left col)

# Create the axes array to store subplots
axes = np.empty((4, 2), dtype=object)
for i in range(4):  # rows
    for j in range(2):  # columns
        axes[i, j] = fig.add_subplot(gs[i, j])

# Row labels + full titles
row_titles = [
    "A. Cortical surface area",
    "B. Cortical and subcortical gray matter volume",
    "C. Cortical thickness",
    "E. Functional connectivity (Schaefer)"
]

title_ys = [0.97, 0.73, 0.5, 0.25]


# Which plots are scatter (True) or PNG (False)
is_plot = [
    [True,  False],
    [True,  False],
    [True,  False],
    [True,  False]
]

# PNG file paths
png_paths = [
    [None, "../area_neu.png"],
    [None, "../volume_neu.png"],
    [None, "../thickness_neu.png"],
    [None, "../schaefer_neu.png"]
]

# Scatter data files
scatter_data_paths = [
    ["../table_for_figures_fullnetmats_schaefer_only/neu/df_area.csv", None],
    ["../table_for_figures_fullnetmats_schaefer_only/neu/df_volume.csv", None],
    ["../table_for_figures_fullnetmats_schaefer_only/neu/df_thickness.csv", None],
["../table_for_figures_fullnetmats_schaefer_only/neu/df_schaefer.csv", None]
]

# Loop to create subplots
for i in range(4):  # rows
    for j in range(2):  # cols
        ax=axes[i, j]

        # Add a title above the row, centered across both subplots
        if j == 0:
            fig.text(
                0.01, title_ys[i], row_titles[i],
                ha='left', va='bottom', fontsize=20, fontweight='bold'
            )

        if is_plot[i][j]:
            path = scatter_data_paths[i][j]

            try:
                df = pd.read_csv(path)
                #phen_df = df[df['phen_group'] == "Neuroticism"]

                sns.scatterplot(
                    data=df, x='estimate_no_covar', y='p_fdr', style='Significance', palette='tab10',
                    s=50, ax=ax
                )
                ax.set_xlabel('Estimate')
                ax.set_ylabel('FDR Corrected p-value')
                ax.set_yscale('log')
                ax.set_xlim(-0.1, 0.1)
                ax.set_ylim(10 ** -5, 1)
                ax.axhline(y=0.05, color='r', linestyle='--')
                ax.axvline(x=0, color='r', linestyle='--')

                # Adjust legend
                if j == 1:
                    ax.legend(title='Yeo Network', loc='upper left', bbox_to_anchor=(1.01, 1))
                else:
                    ax.legend_.remove()

            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading {path}\n{e}", ha='center', va='center')
                ax.axis('off')

        else:
            path = png_paths[i][j]
            if path:
                try:
                    img = Image.open(path)
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, "", ha='center', va='center')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, "No Image", ha='center', va='center')
                ax.axis('off')

# Final layout adjustments
plt.tight_layout(rect=[0.05, 0.03, 1, 0.97])  # leave room for titles
plt.subplots_adjust(hspace=0.4)
plt.savefig("../combined_figure_v8_neu.png", dpi=600)
plt.show()
