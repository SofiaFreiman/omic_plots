import pandas
import numpy
import matplotlib

def pseudotime_gene_expression(adata, gene_list, pseudotime_col='dpt_pseudotime', pseudotime_bins=None, smoothing=None, vmax_ls=None, **kwargs):
    """
    Plots gene expression heatmaps along pseudotime.

    Parameters:
    - adata: AnnData object containing gene expression data.
    - gene_list: List of genes to visualize.
    - pseudotime_col: Column in adata.obs where pseudotime is stored.
    - pseudotime_bins: Number of bins for pseudotime discretization. If None, defaults to len(adata) // 20.
    - smoothing: Rolling window size for smoothing. If None, raw gene expression per bin is plotted.
    - vmax_ls: Optional list of vmax values for each gene plot. If None, calculated as the 95th percentile of expression.
    - kwargs: Additional keyword arguments for customizing internal axes properties.
    """

    matplotlib.pyplot.rcParams['axes.edgecolor'] = 'silver'

    # Create DataFrame with pseudotime values
    df = pandas.DataFrame({'pseudotime': adata.obs[pseudotime_col]})
    
    # Set default pseudotime bin count if not provided
    if pseudotime_bins is None:
        pseudotime_bins = max(1, len(df) // 20)  # Adaptive binning based on dataset size
    
    # Discretize pseudotime into bins
    df['time_bin'] = pandas.cut(
        df['pseudotime'], pseudotime_bins, 
        labels=numpy.linspace(0, df['pseudotime'].max(), pseudotime_bins)
    )
    
    # Extract and process gene expression data
    for gene in gene_list:
        df[gene] = adata[:, gene].X.A[:, 0]  # Extract expression values
    
    # Compute median expression within each pseudotime bin
    df_grouped = df.groupby('time_bin', observed = False).median()
    
    # Apply smoothing if specified
    if smoothing is not None:
        for gene in gene_list:
            df_grouped[gene] = df_grouped[gene].interpolate(method='linear', limit_area='inside')
            df_grouped[gene] = df_grouped[gene].rolling(smoothing, center=True).mean().bfill().ffill()
    
    # Set up the figure and axes
    ngenes = len(gene_list)
    fig, axs = matplotlib.pyplot.subplots(ncols=1, nrows=ngenes, figsize=(12, 0.34 * ngenes), sharex=True)
    fig.subplots_adjust(hspace=0)
    
    # Ensure axs is iterable even if only one gene is plotted
    if ngenes == 1:
        axs = [axs]
    
    # Plot heatmaps for each gene
    for n, gene in enumerate(gene_list):
        axs[n].set_yticks([])
        axs[n].set_ylabel(gene, rotation=True, fontsize=11)
        axs[n].yaxis.set_label_coords(-0.04, 0.05)
        axs[n].grid(False)
        
        # Determine vmax for color scaling
        vmax = vmax_ls[n] if vmax_ls else df_grouped[gene].quantile(0.95)
        vmin = 0
        
        # Plot heatmap
        axs[n].imshow(
            numpy.array([df_grouped[gene]]), aspect='auto', vmin=vmin, vmax=vmax, **kwargs
        )
    
    # Configure x-axis ticks
    xtick_pos = numpy.linspace(0, pseudotime_bins / df['pseudotime'].max(), 11)
    xtick_labels = numpy.round(numpy.linspace(0, 1, 11), 2)
    matplotlib.pyplot.xticks(xtick_pos, xtick_labels, fontsize=11)
    matplotlib.pyplot.xlabel("Pseudotime", fontsize=14)
    
    return fig
