import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap

# --- Apply the style used in the notebook ---
try:
    plt.style.use('data/styles/ri_right_edge_plot_style.mplstyle')
    print("Applied ri_right_edge_plot_style.mplstyle")
except FileNotFoundError:
    print("Warning: ri_right_edge_plot_style.mplstyle not found. Using default style.")
except OSError as e:
    print(f"Warning: Could not apply style. Error: {e}")

# --- Dummy Data Generation ---
N_ROWS = 15
N_COLS = 8
row_labels = [f"Qualification {i+1}" for i in range(N_ROWS)]
col_labels = [f"{2018+i}" for i in range(N_COLS)]

heatmap_data = pd.DataFrame(
    np.random.rand(N_ROWS, N_COLS) * 100, 
    index=row_labels, 
    columns=col_labels
)
top_data = pd.Series(np.random.randint(10000, 50000, size=N_COLS), index=col_labels)
right_data = pd.Series(np.random.randint(1000, 15000, size=N_ROWS), index=row_labels)

# --- Plotting Logic (based on create_heatmap_with_marginals) ---
figsize=(15, 8.4375) # ~16:9 aspect ratio
cmap='Blues'
heatmap_fmt=".1f" # Use .1f for dummy data
line_color="#7dc35a" # Example color
bar_palette='Blues'
wrap_width=25
heatmap_annot=True
title = "Test: Markkinaosuus vs Markkinakoko"
top_title="Koko markkina (dummy)"
right_title="Tutkinnon markkinakoko (dummy)"

# Ensure index/column alignment 
heatmap_data = heatmap_data.sort_index(ascending=True)
right_data = right_data.reindex(heatmap_data.index) 
top_data = top_data.reindex(heatmap_data.columns) 

fig = plt.figure(figsize=figsize)

# Define GridSpec - Matching notebook
gs = fig.add_gridspec(
    2, 2, width_ratios=[15, 2], height_ratios=[4, 25], 
    wspace=0.0, hspace=0.0 # Match notebook
)

# Create axes
ax_top = fig.add_subplot(gs[0, 0])
ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top) 
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main) 

# --- Plot Top Line Plot (Total Volume) ---
ax_top.plot(top_data.index.astype(str), top_data.values, color=line_color, marker='o')
ax_top.set_title(top_title, fontsize=10, color='gray', loc='left')
ax_top.tick_params(axis='x', bottom=False, labelbottom=False) # Hide x labels/ticks on top
ax_top.tick_params(axis='y', length=0)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['bottom'].set_visible(False) # Keep bottom invisible for cleaner look
ax_top.spines['left'].set_color('#cccccc')
ax_top.grid(axis='y', linestyle='-', alpha=0.5)
ax_top.set_ylabel("Koko markkina", fontsize=9, color='gray')
ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ','))) 

# --- Plot Main Heatmap (Market Share %) ---
sns.heatmap(heatmap_data, ax=ax_main, cmap=cmap, annot=heatmap_annot, fmt=heatmap_fmt,
            linewidths=0.5, linecolor='white', cbar=False)
# Styling - Simplified based on notebook intent
ax_main.tick_params(axis='x', bottom=True, labelbottom=True) # Show labels below heatmap
ax_main.tick_params(axis='y', left=False, length=0, rotation=0) # Keep y ticks minimal
ax_main.set_xlabel(None)
ax_main.set_ylabel(None)
# Wrap y-axis labels
def _wrap_text(text, width):
    return '\n'.join(textwrap.wrap(text, width=width))
ax_main.set_yticklabels([_wrap_text(label.get_text(), wrap_width) for label in ax_main.get_yticklabels()],
                        fontsize=10, va='center')
# Let subplots_adjust handle spine visibility/position initially
# for spine in ax_main.spines.values():
#     spine.set_visible(False)

# --- Plot Right Bar Plot (Latest Year Volume) ---
if right_data is not None and not right_data.empty:
     sns.barplot(x=right_data.values, y=right_data.index, ax=ax_right, orient='h', palette=bar_palette, hue=right_data.index, legend=False)
     # Styling
     ax_right.tick_params(axis='y', left=False, labelleft=False, length=0) # Hide y ticks/labels
     ax_right.tick_params(axis='x', length=0)
     # Spines - keep consistent with notebook implicit style (only bottom visible?)
     ax_right.spines['top'].set_visible(False)
     ax_right.spines['right'].set_visible(False)
     ax_right.spines['bottom'].set_color('#cccccc') # Keep bottom visible and styled
     ax_right.spines['left'].set_visible(False)
     ax_right.set_xlabel(right_title, fontsize=9, color='gray')
     ax_right.set_ylabel(None) 
     ax_right.grid(axis='x', linestyle='-', alpha=0.5)
     # Set specific ticks: 0 and max value
     max_val = right_data.values.max()
     if max_val > 0:
         ax_right.set_xticks([0, max_val])
         ax_right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
         ax_right.set_xlim(left=0, right=max_val * 1.05)
     else: 
         ax_right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
         
     # Explicitly set ylim to match heatmap after plotting - REMOVED
     # ax_right.set_ylim(ax_main.get_ylim())

else:
     ax_right.set_visible(False)
     

# --- Figure Level Formatting ---
fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98, ha='left', x=0.05)

# --- Layout Adjustment (Experiment Here) ---
# Option 1: No adjustment 
# pass 

# Option 2: subplots_adjust - Matching notebook
# fig.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.9) 

# Option 3: tight_layout
# fig.tight_layout(rect=[0.15, 0.07, 0.95, 0.95]) # Example rect

plt.show() 