import scanpy as sc
import anndata
import pandas as pd
import scipy
import numpy as np

import doubletdetection
import glob
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

########## SEQC functions ##########
def read_in_dense(path, sample_name):
    adata = sc.read_csv(path, first_column_names=True)
    idy = adata.var_names != 'CLUSTER'
    adata = adata[:, idy]
    
    # combine duplicate genes
    cells = adata.obs_names
    sparse_df = scipy.sparse.csc_matrix(adata.X)
    genes = pd.Series(adata.var_names)
    counts, genes = remove_duplicate_genes(sparse_df, genes)
    adata = sc.AnnData(X=counts, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    adata.obs['Sample'] = sample_name

    return adata
    
def remove_duplicate_genes(
    counts: scipy.sparse.csc_matrix,
    genes: pd.Series,
):
    # Isolate duplicate genes; column 0 is gene name
    dup_genes = pd.DataFrame(genes[genes.duplicated(keep = False)]).groupby(0)

    if len(dup_genes) > 0:

        # Aggregate duplicate genes; Uses Column sparse matrix ops
        del_idx = np.array([])
        new_cols = []
        new_genes = genes.values

        for gene, group in dup_genes:
            idx = group.index.values
            assert len(idx) > 1
            # Get indices to remove duplicate genes and gene columns
            del_idx = np.append(del_idx, idx)
            # Add gene name to end of list
            new_genes = np.append(new_genes, gene)
            # Get aggregated columns to add to end of counts matrix
            new_col = counts[:, idx[0]]
            for i in idx[1:]:
                # Note: csc sum method converts to dense matrix; iterating with '+' is ~10x faster
                new_col = new_col + counts[:, i]
            new_cols.append(new_col)

        # Remove duplicate gene names
        new_genes = pd.Series(np.delete(new_genes, del_idx.astype(int)))

        # Concatenate single gene columns with aggregated duplicates
        keep_idx = list(set(genes.index.values) - set(del_idx))
        new_counts = scipy.sparse.hstack([counts[:, keep_idx]] + new_cols).tocsc()

        # Readout
        no_cols_removed = len(del_idx) - len(new_cols)
        print(str(no_cols_removed) + ' gene duplicate columns were removed.')

        return new_counts, new_genes

    else:
        return counts, genes

########## plotting functions ##########
def pca_plot(adata):
    fig, axes = plt.subplots(1, 2, dpi=300, figsize=(10, 4))
    knee = adata.obsm['X_pca'].shape[1]
    
    x = range(len(adata.uns['pca']['variance_ratio']))
    y = adata.uns['pca']['variance_ratio']
    sns.scatterplot(x=x, y=y, s=4, ax=axes[0])
    axes[0].set(xlabel='PC', ylabel='Fraction of variance explained',
           title='Fraction of variance explained per PC')
    axes[0].axvline(knee, color = 'r')
    
    #Plot cumulative variance
    cml_var_explained = np.cumsum(adata.uns['pca']['variance_ratio'])
    x = range(len(adata.uns['pca']['variance_ratio']))
    y = cml_var_explained
    sns.scatterplot(x=x, y=y, s=4, ax=axes[1])
    axes[1].set(xlabel='PC', ylabel='Cumulative fraction of variance explained',
                title='Cumulative fraction of variance explained by PCs')
    axes[1].axvline(knee, color = 'r')
    
    fig.tight_layout()
    
    return cml_var_explained

def plot_hist(adata, x, ax, args=None):

    plot_params = {
        'data':      adata.obs,
        'x':         x,
        'kde':       True,
        'stat':      'count',
        'bins':      100,
        'alpha':     0.8,
        'line_kws':   dict(linewidth=3)
    }
    
    if args is not None:
        for arg in args:
            plot_params[arg] = args[arg]
            if arg == 'color':
                del plot_params['alpha']

    sns.histplot(**plot_params, ax=ax)
    return ax

def plot_hist_sample(adata, x, ax, palette, args=None):
    
    sample_names = list(palette.keys())
    sample_names.sort()

    
    plot_params = {
        'data':      adata.obs,
        'x':         x,
        'kde':       True,
        'stat':      'count',
        'bins':      100,
        'line_kws':   dict(linewidth=3),
        'hue':       'Sample',
        'hue_order': sample_names,
        'palette':   palette,
    }
    
    if args is not None:
        for arg in args:
            plot_params[arg] = args[arg]

    sns.histplot(**plot_params, ax=ax)
    return ax

def plot_scatter(adata, x, y, ax, args=None):
    plot_params = {
        'data':      adata.obs,
        'x':         x,
        'y':         y,
        'alpha':     0.8
    }

    if args is not None:
        for arg in args:
            plot_params[arg] = args[arg]
            if arg == 'color':
                del plot_params['alpha']

    sns.scatterplot(**plot_params, ax=ax)
    return ax

def plot_scatter_sample(adata, x, y, palette, ax, args=None):
    sample_names = list(palette.keys())
    sample_names.sort()
    
    plot_params = {
        'data':      adata.obs,
        'x':         x,
        'y':         y,
        'hue':       'Sample',
        'hue_order': sample_names,
        'palette':   palette,
    }
    
    if args is not None:
        for arg in args:
            plot_params[arg] = args[arg]
            if arg == 'color':
                del plot_params['alpha']

    sns.scatterplot(**plot_params, ax=ax)
    return ax


def plot_num_cells(adata, obs_col, ax, palette=None):  
    
    if not palette:
        palette = gen_palette(adata, obs_col)

    df = adata.obs[obs_col].value_counts().to_frame().reset_index()
    df.columns = [obs_col, 'Number of Cells']

    sns.barplot(data=df, x=obs_col, y='Number of Cells',
               edgecolor='black', palette=palette)

    ax.set(xlabel=obs_col, 
           ylabel='Number of Cells', 
           title='Number of Cells per Sample')

    # Prints the value above the bar chart
    for container in ax.containers:
        ax.bar_label(container)

    return ax

def gen_palette(adata, obs_col):
    
    values = adata.obs[obs_col].unique()
    
    palette = sns.color_palette("husl", len(values))
    palette_d = {}
    for i, val in enumerate(values):
        palette_d[val] = palette[i]

    return palette_d


def plot_hvgs(adata, ax):

    adata.uns['id_hvg'] = np.where(adata.var['highly_variable'])[0]

    # Scatterplot values
    x = np.log2(adata.var['means'] + 1)
    y = np.log2(adata.var['variances_norm'] + 1)

    s = 30

    # All genes
    sns.scatterplot(x=x, y=y, ax=ax, label='All genes', color='b', s=s)

    # Highly variable genes
    sns.scatterplot(x=x[adata.uns['id_hvg']], y=y[adata.uns['id_hvg']], 
                    ax=ax, label='HVGs', color='r', s=s)

    ax.set(xlabel='log2(Mean Expression)', ylabel='log2(Normalized Variance)', 
           title='Mean Expression vs. Normalized Variance of Genes')
    return ax

def plot_umap_subset(adata, obs_col, axes, palette=None):
    
    if not palette:
        palette = gen_palette(adata, obs_col)
        
    values = adata.obs[obs_col].unique().tolist()
    values.sort()
    
    lightgray = sns.color_palette("pastel")[7]

    for i, val in enumerate(values):
        c = palette[val]

        idx = adata.obs[obs_col] == val
        subset = adata[idx, :]

        sns.scatterplot(x=adata.obsm["X_umap"][:, 0], 
                        y=adata.obsm["X_umap"][:, 1], 
                        color=lightgray, s=2, ax=axes[i])
        sns.scatterplot(x=subset.obsm["X_umap"][:, 0], 
                        y=subset.obsm["X_umap"][:, 1], 
                        color=c, s=2, ax=axes[i])

        sns.despine(ax=axes[i], left=True, bottom=True) 
        axes[i].tick_params(left=False, bottom=False)
        axes[i].set(xticklabels=[], yticklabels=[])

        axes[i].set(title=val)
        
def plot_prop_cells_per_sample(adata, obs_col, ax, palette):

    sample_names = list(palette.keys())
    sample_names.sort()

    df = adata.obs[[obs_col, 'Sample']]
    values = list(set(df[obs_col]))
    values.sort()

    values_sample_d = {}
    for val in values:
        values_sample_d[val] = df.loc[df[obs_col] == val, 'Sample'].value_counts(dropna=False).to_dict()

    # Calculate proportion of #cells of sample / #cells in cluster
    df = pd.DataFrame.from_dict(values_sample_d).T
    df = df.div(df.sum(axis=1), axis=0)
    df = df.round(2)

    df.plot(kind='bar', stacked=True, edgecolor='black', ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    ax.set(title=f'{obs_col} per Sample', xlabel=obs_col, ylabel='Percentage of cells')
    
    return ax


def plot_boxplot_cluster(adata, groupby, obs_col, palette, ax):
    df = adata.obs[[obs_col, groupby]]
    sns.boxplot(data=df, x=groupby, y=obs_col, ax=ax, palette=palette)
    ax.set(title=f'{obs_col} per {groupby}')
    return ax


def plot_mito_combined_one_color(adata, combined_sample_color):
    ########## Combined
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Histogram
    plot_labels = {
        'title':     'MT-content',
        'xlabel':    'Mitochondrial Fraction',
    }
    plot_hist(
        adata=adata,
        x='pct_counts_mito',
        ax=axes[0],
        args=dict(color=combined_sample_color)
    )
    axes[0].set(**plot_labels)

    # Scatterplot
    plot_labels = {
        'title':     'MT-content vs Library Size',
        'xlabel':    'Mitochondrial Fraction',
        'ylabel':    'log10(Total Counts)'
    }
    plot_scatter(
        adata=adata,
        x='pct_counts_mito', 
        y='log10_total_counts', 
        ax=axes[1], 
        args=dict(color=combined_sample_color)
    )
    axes[1].set(**plot_labels)


    fig.suptitle('Combined Samples')
    fig.tight_layout()
    
    
def plot_mito_combined(adata, sample_palette_d, sample_names):
        ########## Colored by sample
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Histogram
    plot_labels = {
        'title':     'MT-content',
        'xlabel':    'Mitochondrial Fraction',
    }
    plot_hist_sample(
        adata=adata,
        x='pct_counts_mito',
        palette = sample_palette_d,
        ax=axes[0]
    )
    axes[0].set(**plot_labels)

    # Scatterplot
    plot_labels = {
        'title':     'MT-content vs Library Size',
        'xlabel':    'Mitochondrial Fraction',
        'ylabel':    'log10(Total Counts)'
    }
    plot_scatter_sample(
        adata=adata,
        x='pct_counts_mito', 
        y='log10_total_counts', 
        palette = sample_palette_d,
        ax=axes[1]
    )
    axes[1].set(**plot_labels)

    fig.suptitle('Mitochondrial Fraction')
    fig.tight_layout()
    
def plot_mito_separated(adata, sample_palette_d, sample_names):
    ########## Individual samples
    num_samples = len(sample_names)
    fig, axes = plt.subplots(num_samples, 2, 
                         figsize=(12, 5 * num_samples), 
                         dpi=300,
                         sharex='col', sharey='col')

    for i, sample in enumerate(sample_names):
        sample_subset = adata.obs[adata.obs['Sample'] == sample]

        # Histogram
        plot_labels = {
            'title':     sample + '\nMT-content',
            'xlabel':    'Mitochondrial Fraction',
        }
        plot_hist(
            adata=adata,
            x='pct_counts_mito', 
            ax=axes[i, 0], 
            args=dict(
                data=sample_subset,
                color=sample_palette_d[sample]
            )
        )
        axes[i, 0].set(**plot_labels)


        # Scatterplot
        plot_labels = {
            'title':     sample + '\nMT-content vs Library Size',
            'xlabel':    'Mitochondrial Fraction',
            'ylabel':    'log10(Total Counts)'
        }
        plot_scatter(
            adata=adata,
            x='pct_counts_mito', 
            y='log10_total_counts',
            ax=axes[i, 1], 
            args=dict(
                data=sample_subset, 
                color=sample_palette_d[sample]
            )
        )
        axes[i, 1].set(**plot_labels)

        axes[i, 0].xaxis.set_tick_params(labelbottom=True)
        axes[i, 1].xaxis.set_tick_params(labelbottom=True)

    fig.suptitle('Mitochondrial Fraction')
    fig.tight_layout()


def plot_ribo_combined_one_color(adata, combined_sample_color):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Histogram
    plot_labels = {
        'title':     'Ribo-content',
        'xlabel':    'Ribosomal Fraction',
    }
    plot_hist(
        adata=adata,
        x='pct_counts_ribo',
        ax=axes[0],
        args=dict(color=combined_sample_color)
    )
    axes[0].set(**plot_labels)

    # Scatterplot
    plot_labels = {
        'title':     'Ribo-content vs Library Size',
        'xlabel':    'Ribosomal Fraction',
        'ylabel':    'log10(Total Counts)'
    }
    plot_scatter(
        adata=adata,
        x='pct_counts_ribo', 
        y='log10_total_counts', 
        ax=axes[1], 
        args=dict(color=combined_sample_color)
    )
    axes[1].set(**plot_labels)


    fig.suptitle('Combined Samples')
    fig.tight_layout()
    
    
def plot_ribo_combined(adata, sample_palette_d, sample_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Histogram
    plot_labels = {
        'title':     'Ribo-content',
        'xlabel':    'Ribosomal Fraction',
    }
    plot_hist_sample(
        adata=adata,
        x='pct_counts_ribo',
        palette = sample_palette_d,
        ax=axes[0]
    )
    axes[0].set(**plot_labels)

    # Scatterplot
    plot_labels = {
        'title':     'Ribo-content vs Library Size',
        'xlabel':    'Ribosomal Fraction',
        'ylabel':    'log10(Total Counts)'
    }
    plot_scatter_sample(
        adata=adata,
        x='pct_counts_ribo', 
        y='log10_total_counts', 
        palette = sample_palette_d,
        ax=axes[1]
    )
    axes[1].set(**plot_labels)

    fig.suptitle('Ribosomal Fraction')
    fig.tight_layout()
    
    
def plot_ribo_separated(adata, sample_palette_d, sample_names):
    num_samples = len(sample_names)
    fig, axes = plt.subplots(num_samples, 2, 
                         figsize=(12, 5 * num_samples), 
                         dpi=300,
                         sharex='col', sharey='col')

    for i, sample in enumerate(sample_names):
        sample_subset = adata.obs[adata.obs['Sample'] == sample]

        # Histogram
        plot_labels = {
            'title':     sample + '\nRibo-content',
            'xlabel':    'Ribosomal Fraction',
        }
        plot_hist(
            adata=adata,
            x='pct_counts_ribo', 
            ax=axes[i, 0], 
            args=dict(
                data=sample_subset,
                color=sample_palette_d[sample]
            )
        )
        axes[i, 0].set(**plot_labels)


        # Scatterplot
        plot_labels = {
            'title':     sample + '\nRibo-content vs Library Size',
            'xlabel':    'Ribosomal Fraction',
            'ylabel':    'log10(Total Counts)'
        }
        plot_scatter(
            adata=adata,
            x='pct_counts_ribo', 
            y='log10_total_counts',
            ax=axes[i, 1], 
            args=dict(
                data=sample_subset, 
                color=sample_palette_d[sample]
            )
        )
        axes[i, 1].set(**plot_labels)

        axes[i, 0].xaxis.set_tick_params(labelbottom=True)
        axes[i, 1].xaxis.set_tick_params(labelbottom=True)

        fig.suptitle('Ribosomal Fraction')
        fig.tight_layout()
    
def plot_ngenes_combined_one_color(adata, combined_sample_color):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Histogram
    plot_labels = {
        'title':     'Number of Cells a Gene is Expressed in',
        'xlabel':    'Number of Cells',
    }
    plot_hist(
        adata=adata,
        x='n_cells_by_counts', 
        ax=axes[0], 
        args=dict(
            data=adata.var, 
            color=combined_sample_color
        )
    )
    axes[0].set(**plot_labels)

    # Log2 histogram
    df = pd.DataFrame(adata.var['n_cells_by_counts'])
    df['n_cells_by_counts'] = np.log2(df['n_cells_by_counts'] + 1)
    plot_labels = {
        'title':     'log2(Number of Cells a Gene is Expressed in)',
        'xlabel':    'log2(Number of Cells)',
    }
    plot_hist(
        adata=adata,
        x='n_cells_by_counts', 
        ax=axes[1], 
        args=dict(
            data=df, 
            color=combined_sample_color
        )
    )
    axes[1].set(**plot_labels)

    fig.suptitle('Combined Samples')
    fig.tight_layout()
    
def plot_ngenes_combined(adata, sample_palette_d, sample_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Within each sample
    # For each gene, how many cells express that gene
    for sample in sample_names:

        # Calculate the counts for each sample
        sample_subset = adata.obs[adata.obs['Sample'] == sample].index
        c = sample_palette_d[sample]
        gene_count = np.apply_along_axis(np.count_nonzero, 
                                         axis=0, arr=adata[sample_subset, :].X.todense())

        # Histogram n cells by counts
        df = pd.DataFrame(gene_count, columns=['n_cells_by_counts'])
        plot_labels = {
            'title':     'Number of Cells a Gene is Expressed in',
            'xlabel':    'Number of Cells',
        }
        plot_hist(
            adata=adata,
            x='n_cells_by_counts', 
            ax=axes[0], 
            args=dict(
                data=df, 
                color=c, 
                label=sample, 
                legend=True
            )
        )
        axes[0].set(**plot_labels)

        # Histogram log2 n cells by counts
        gene_count = np.log2(gene_count + 1)
        df = pd.DataFrame(gene_count, columns=['n_cells_by_counts'])
        plot_labels = {
            'title':     'log2(Number of Cells a Gene is Expressed in)',
            'xlabel':    'log2(Number of Cells)',
        }
        plot_hist(
            adata=adata,
            x='n_cells_by_counts', 
            ax=axes[1], 
            args=dict(data=df, color=c)
        )
        axes[1].set(**plot_labels)

    fig.legend(loc=9)
    fig.tight_layout()
    
def plot_ngenes_separated(adata, sample_palette_d, sample_names):
    num_samples = len(sample_names)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples), dpi=300,
                             sharex='col', sharey='col')

    # Within each sample
    # For each gene, how many cells express that gene
    for i, sample in enumerate(sample_names):

        # Calculate counts for each sample
        sample_subset = adata.obs[adata.obs['Sample'] == sample].index
        c = sample_palette_d[sample]
        gene_count = np.apply_along_axis(np.count_nonzero, axis=0, arr=adata[sample_subset].X.todense())

        # Histogram n cells by counts
        df = pd.DataFrame(gene_count, columns=['n_cells_by_counts'])
        plot_labels = {
            'title':     sample + '\nNumber of Cells a Gene is Expressed in',
            'xlabel':    'Number of Cells',
        }
        plot_hist(
            adata=adata,
            x='n_cells_by_counts', 
            ax=axes[i, 0], 
            args=dict(data=df, color=c)
        )
        axes[i, 0].set(**plot_labels)

        # Histogram log2 n cells by counts
        gene_count = np.log2(gene_count + 1)
        df = pd.DataFrame(gene_count, columns=['n_cells_by_counts'])

        plot_labels = {
            'title':     sample + '\nlog2(Number of Cells a Gene is Expressed in',
            'xlabel':    'log2(Number of Cells)',
        }
        plot_hist(
            adata=adata,
            x='n_cells_by_counts', 
            ax=axes[i, 1], 
            args=dict(data=df, color=c))
        axes[i, 1].set(**plot_labels)

        # Make sure x-axis has tick labels
        axes[i, 0].xaxis.set_tick_params(labelbottom=True)
        axes[i, 1].xaxis.set_tick_params(labelbottom=True)

    fig.tight_layout()
    
def plot_lib_combined_one_color(adata, combined_sample_color):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Histogram
    plot_labels = {
        'title':     'Library Size Distribution',
        'xlabel':    'Total Counts',
    }
    plot_hist(adata=adata, x='total_counts', ax=axes[0], 
              args=dict(color=combined_sample_color))
    axes[0].set(**plot_labels)

    # Histogram Log10
    plot_labels = {
        'title':     'Log10(Library Size) Distribution',
        'xlabel':    'Log10(Total Counts)',
    }
    plot_hist(adata=adata, x='log10_total_counts', ax=axes[1], 
              args=dict(color=combined_sample_color))
    axes[1].set(**plot_labels)

    fig.suptitle('Combined Samples')
    fig.tight_layout()
    
def plot_lib_combined(adata, sample_palette_d, sample_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Histogram
    plot_labels = {
        'title':     'Library Size Distribution',
        'xlabel':    'Total Counts',
    }
    plot_hist_sample(adata=adata, x='total_counts', ax=axes[0], palette=sample_palette_d,)
    axes[0].set(**plot_labels)

    # Histogram Log10
    plot_labels = {
        'title':     'Log10(Library Size) Distribution',
        'xlabel':    'Log10(Total Counts)',
    }
    plot_hist_sample(adata=adata, x='log10_total_counts', ax=axes[1], palette=sample_palette_d,)
    axes[1].set(**plot_labels)

    fig.suptitle('Library Size')
    fig.tight_layout()
    
def plot_lib_separated(adata, sample_palette_d, sample_names):
    num_samples = len(sample_names)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples), dpi=300,
                             sharex='col', sharey='col')

    for i, sample in enumerate(sample_names):
        sample_subset = adata.obs[adata.obs['Sample'] == sample]
        c = sample_palette_d[sample]

        plot_labels = {
            'title':     sample + '\nLibrary Size Distribution',
            'xlabel':    'Total Counts',
        }
        plot_hist(adata=adata, x='total_counts', ax=axes[i, 0], 
                  args=dict(data=sample_subset, color=c))
        axes[i, 0].set(**plot_labels)

        # Log10
        plot_labels = {
            'title':     sample + '\nLog10(Library Size) Distribution',
            'xlabel':    'Log10(Total Counts)',
        }
        plot_hist(adata=adata, x='log10_total_counts', ax=axes[i, 1], 
                  args=dict(data=sample_subset, color=c))
        axes[i, 1].set(**plot_labels)

        # Make sure x-axis has tick labels
        axes[i, 0].xaxis.set_tick_params(labelbottom=True)
        axes[i, 1].xaxis.set_tick_params(labelbottom=True)

    fig.suptitle('Library Size')
    fig.tight_layout()


    
########## QC functions ##########

def remove_genes(adata, gene_list): 
    return adata[:,[x not in gene_list for x in adata.var.index]]

def preprocessing(adata):
    # Normalize
    adata.layers['raw_counts'] = adata.X
    adata.layers['median'] = adata.layers['raw_counts'].copy()
    sc.pp.normalize_total(adata, layer='median')
    adata.layers['log'] = adata.layers['median'].copy()
    sc.pp.log1p(adata, layer='log')
    
    # HVG
    sc.pp.highly_variable_genes(adata, layer='raw_counts', 
                                flavor='seurat_v3', max_mean=np.inf, 
                                n_top_genes=4500, batch_key='Sample')
    
    # PCA
    n_pcs = 300
    adata.X = adata.layers['log']
    sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True)
    adata.obsm['X_pca_max'] = adata.obsm['X_pca'].copy()
    adata.obsm['X_pca'] = adata.obsm['X_pca_max'][:, :45]
    
    # UMAP
    sc.pp.neighbors(adata, method='umap', n_neighbors = 30, use_rep='X_pca')
    sc.tl.umap(adata, min_dist = 0.1, random_state=5)
    
    # Phenograph
    k = 30
    communities, _, _ = sc.external.tl.phenograph(
        pd.DataFrame(adata.obsm['X_pca']), k=k,
        nn_method = 'brute', njobs = -1,
    )
    adata.obs['PhenoGraph_clusters'] = pd.Categorical(communities)
    return adata

def qc_metrics(adata, path_RB='utils/RB_genes_human'):
    mito_genes = adata.var_names.str.startswith('MT-')
    adata.var['mito'] = mito_genes
    mito_genes = np.array(adata.var.index)[mito_genes]
    
    with open(path_RB,'r') as file:
        lines = file.readlines()
    RB_genes = [x.rstrip('\n') for x in lines]
    data_genes = list(adata.var.index)
    RB_genes_in_data = set(data_genes).intersection(RB_genes)
    RB_genes_in_data = list(RB_genes_in_data)
    
    adata.var['ribo'] = False
    adata.var.loc[RB_genes_in_data, 'ribo'] = True
    
    sc.pp.calculate_qc_metrics(adata, qc_vars=('mito', 'ribo'), inplace=True, layer='raw_counts')
    
    return adata, mito_genes, RB_genes_in_data
