import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.colors import to_hex
import matplotlib.cm as cm
from pandas.api.types import CategoricalDtype


def cells_along_trj(
    adata: AnnData,
    cell_color_dict: dict = None,
    anno_col: str = "cell_type",
    pseudotime_key: str = "dpt_pseudotime",
    figsize: tuple = (12, 0.5),
    dot_size: int = 15
) -> plt.Figure:
    """
    Plots cells along a pseudotime trajectory colored by cell type.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with `obs` containing pseudotime and annotation.
    cell_color_dict : dict, optional
        Dictionary mapping cell types to colors. If None, a default palette tab10 will be used.
    anno_col : str, default "cell_type"
        Column in `adata.obs` containing cell type annotations.
    pseudotime_key : str, default "dpt_pseudotime"
        Column in `adata.obs` containing pseudotime values.
    figsize : tuple, default (12, 0.5)
        Size of the figure.
    dot_size : int, default 15
        Size of the scatter plot dots.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"Pseudotime key '{pseudotime_key}' not found in adata.obs.")
    if anno_col not in adata.obs.columns:
        raise ValueError(f"Annotation column '{anno_col}' not found in adata.obs.")

    df = pd.DataFrame()
    df["Pseudotime"] = adata.obs[pseudotime_key]

    # Clean up categories if necessary
    if isinstance(adata.obs[anno_col].dtype, CategoricalDtype):
        adata.obs[anno_col] = adata.obs[anno_col].cat.remove_unused_categories()

    df["Cell type"] = adata.obs[anno_col]

    # Set color palette if not provided
    if cell_color_dict is None:
        unique_cell_types = df["Cell type"].unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_cell_types))
        cell_color_dict = dict(zip(unique_cell_types, map(to_hex, palette)))

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x="Pseudotime",
        y=np.random.uniform(low=0, high=1, size=len(df)),
        hue="Cell type",
        data=df,
        palette=cell_color_dict,
        s=dot_size,
        hue_order=cell_color_dict.keys(),
        ax=ax
    )

    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.set_xticks(np.round(np.linspace(0, 1, 11), 2))
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.legend(
        title="Cell type",
        markerscale=2,
        bbox_to_anchor=(1, 1),
        frameon=False,
        fontsize=12
    )
    
    fig.subplots_adjust(top=0.9, bottom=0.2) # Use subplots_adjust to fix layout issue for very short height
    
    return fig
