import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  
file_path = filedialog.askopenfilename(title='Select Excel file', filetypes=[("Excel files", "*.xlsx *.xls")])

data = pd.read_excel(file_path)

columns_to_plot = ['Activity (Light)', 'Activity (Dark)', 'Close contact (Light)', 'Close contact (Dark)']

group_colors = {
    'Male4': '#0000FF',
    'Male8': '#0F80FF',
    'Male15': '#21FFFF',
    'Female4': '#FB02FF',
    'Female8': '#FB0280',
    'Female16': '#FC6666'
}

def hist_stacked(ax, series, groupby, colors, **kwargs):
    bottom = np.zeros(10)
    bins = np.histogram(series.dropna(), bins=10)[1]
    for name, color in colors.items():
        counts, _ = np.histogram(series[groupby == name].dropna(), bins=bins)
        ax.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottom,
               color=color, edgecolor='black', align='edge', alpha=0.7)
        bottom += counts

g = sns.PairGrid(data=data, vars=columns_to_plot, diag_sharey=False)

g.map_diag(lambda x, **kwargs: hist_stacked(plt.gca(), x, data['Group'], group_colors, **kwargs))

def scatter_with_regline(x, y, **kwargs):
    ax = plt.gca()
    sns.scatterplot(x=x, y=y, ax=ax, hue=data['Group'], palette=group_colors, legend=False, marker='o')
    sns.regplot(x=x, y=y, ax=ax, scatter=False, lowess=True, color='red', ci=None)

g.map_lower(scatter_with_regline)

def corrfunc(x, y, **kwargs):
    r, p = stats.pearsonr(x, y)
    ax = plt.gca()
    font_size = abs(r) * 80 + 5
    ax.annotate(f'{r:.2f}', xy=(.5, .5), xycoords=ax.transAxes,
                ha='center', va='center', fontsize=font_size, color='black')
    if p < 0.001:
        ax.annotate('***', xy=(.8, .8), xycoords=ax.transAxes, color='red', fontsize=20, ha='center', va='center')
    elif p < 0.01:
        ax.annotate('**', xy=(.8, .8), xycoords=ax.transAxes, color='red', fontsize=20, ha='center', va='center')
    elif p < 0.05:
        ax.annotate('*', xy=(.8, .8), xycoords=ax.transAxes, color='red', fontsize=20, ha='center', va='center')

g.map_upper(corrfunc)

plt.subplots_adjust(top=0.95)
g.fig.suptitle('')
plt.show()
