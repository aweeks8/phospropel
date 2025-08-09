#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:49:10 2025

@author: Amy
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Rectangle


def readMod(file, modification):

    """
    Loads a Proteome Discoverer-exported Peptide Groups csv file into a Pandas dataframe and makes a separate dataframe that has only peptides with a user-defined modification. The function also strips the leading/trailing tryptic cleavage annotations (e.g., "[K]." at the start or ".[R]" at the end) from the "Annotated Sequence" column.

    Parameters
    ----------
    file : str or path-like
        Path to a CSV file containing peptide data. Must contain at least the columns:
        - "Annotated Sequence"
        - "Modifications"
    modification : str
        The modification of interest as it appears in the "Modifications" column
        (e.g., "Phospho" or "Dehydrated").

    Returns
    -------
    peptides : pandas.DataFrame
        DataFrame of all peptides from the CSV file
    peptides_mod : pandas.DataFrame
        Subset of `peptides` containing only rows where the "Modifications"
        column includes the specified `modification`.
    """

    peptides = pd.read_csv(file)
    peptides["Annotated Sequence"] = peptides["Annotated Sequence"].str.replace(
    r"(^\[[^\]]+\]\.)|(\.\[[^\]]+\]$)",
    "", regex=True)

    peptides_mod = peptides[peptides["Modifications"].str.contains(modification, na=False)]

    return peptides, peptides_mod


def lowercaseModSites(df, modification):

    """
    Convert one-letter amino acid code to lowercase in the "Annotated Sequence" column of a peptide DataFrame at positions carrying a specified modification.

    The function identifies modified residues by parsing the "Modifications" column,
    finds their positions, and converts those residues to lowercase in the
    "Annotated Sequence". Other residues remain unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing peptide data. Must have:
        - "Annotated Sequence": peptide sequence string
        - "Modifications": modification annotations with residue positions in square brackets (e.g. [S10] or [S10, S11])
    modification : str
        Name of the modification to search for in the "Modifications" column
        (e.g., "Phospho" or "Dehydrated").

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with the "Annotated Sequence" column modified so that
        residues at the specified modification sites are lowercase.

    Notes
    -----
    - Only modifications on S, T, or Y (matching the regex `[STY](\d{1,2})`) are affected.
    """

    offset = len(modification) + 1
    df = df.copy()

    def process_row(row):
        seq = row["Annotated Sequence"]
        mods = row["Modifications"]
        input_peptide = seq
        lowered = input_peptide
        if modification in mods:
            positionstart = mods.index(modification) + offset
            positionend = mods[positionstart:].index("]") + positionstart + 1
            localizationinfo = mods[positionstart:positionend]
            input_peptide = row["Annotated Sequence"]
            position = re.findall(r"[STY](\d{1,2})", localizationinfo)
            char_list = list(input_peptide)
            for i in position:
                pos_index = int(i) - 1
                char_list[pos_index] = char_list[pos_index].lower()
            lowered = ''.join(char_list)

        return lowered

    df["Annotated Sequence"] = df.apply(process_row, axis=1)
    return df


def alignLowerModSitesSTY(
    df,
    modification,
    central_residues="STY",
    require_proline_at_plus1=False,
    exclude_proline_at_plus1=False
):

    """
    Extracts 9-amino-acid sequence windows centered on modified S/T/Y residues.

    For each peptide in `df` that contains the specified `modification` in the
    "Modifications" column, this function:
      - Finds positions of modified residues matching `central_residues`.
      - Pads the sequence with underscores ('_') on both sides so that sites
        near the termini still produce 9-residue windows.
      - Returns each 9-mer window centered on the modified residue
        (4 residues before, the modified residue, and 4 residues after).

    Optional filters can require or exclude proline ('P') at the +1 position.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing peptide data. Must include:
        - "Annotated Sequence": peptide sequence string
        - "Modifications": modification annotations with residue positions
          in square brackets.
        Typically, modified sites in "Annotated Sequence" are lowercase as the input dataframe is typically generated using lowercaseModSites.
    modification : str
        Modification name to match in the "Modifications" column
        (e.g., "Phospho" or "Dehydrated").
    central_residues : str, optional
        One or more amino acid letters to treat as valid central residues for
        alignment (default "STY").
    require_proline_at_plus1 : bool, optional
        If True, only keeps sites with 'P' at position +1 (default False).
    exclude_proline_at_plus1 : bool, optional
        If True, removes sites with 'P' at position +1 (default False).

    Returns
    -------
    list of str
        List of 9-mer sequence windows centered on the modified residue.
        Underscores ('_') indicate padding for sites near sequence termini.
    """
    offset = len(modification) + 1
    pattern = rf"[{re.escape(central_residues)}](\d{{1,2}})"

    modsites = []

    for _, row in df.iterrows():
        mods = row.get("Modifications")
        seq  = row.get("Annotated Sequence")

        positionstart = mods.index(modification) + offset
        positionend = mods[positionstart:].index("]") + positionstart + 1
        localizationinfo = mods[positionstart:positionend]
        padded = "____" + seq + "____"
        positions = re.findall(pattern, localizationinfo)

        for site_str in positions:
            site_index = int(site_str)  # 1-based
            center = site_index + 4   # account for left padding
            nine_mer = padded[center-5:center+4]
            plus1_residue = nine_mer[5]
            if require_proline_at_plus1 and plus1_residue != 'P':
                continue  # require P at +1, skip if not
            if exclude_proline_at_plus1 and plus1_residue == 'P':
                continue  # exclude if P at +1

            modsites.append(nine_mer)

    return modsites


def modCountAAOccurrence(inputlist):

    """
    Counts the occurrence of each amino acid at each position in a set of aligned sequences.

    Given a list of 9-residue sequence windows centered on a modified site,
    this function counts how many times each of 24 possible characters occurs at
    each position. The 24 characters include the 20 standard amino acids, lowercase
    's', 't', and 'y' to indicate modified serine, threonine, and tyrosine, and '_'
    as a padding character.

    Parameters
    ----------
    inputlist : list of str
        List of 9-character sequences (with possible '_' padding) representing
        aligned sequence windows around modification sites.

    Returns
    -------
    numpy.ndarray
        A 24 × 9 array where rows correspond to amino acids in the order:
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
         's', 't', 'y', '_'],
        and columns correspond to positions -4 through +4 relative to the central residue.
        Each element contains the count of that amino acid at that position.
    """
    aminoacids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 's', 't', 'y', '_']
    val = 0
    aminoDict=dict()
    for AA in aminoacids:
    	aminoDict[AA] = val
    	val += 1

    modcount = np.zeros((24,9))

    for site in inputlist:
        for i in range(0,9):
            try:
                modcount[aminoDict[site[i]]][i] +=1
            except:
                pass
    return modcount

def proteomeAbundanceControlCount(directory_path, dimension1=9, dimension2=24):

    """
    Builds a control counts matrix for the global abundance of each amino acid, as well as phosphoserine, phosphothreonine, and phosphotyrosine, among the analyzed phosphopeptides.

    Reads all CSV files in `directory_path`, extracts peptides with a
    'Phospho' modification, converts modified S/T/Y residues to lowercase,
    and counts the total occurrences of each amino acid across all sequences. The counts are then expanded into a (23,9) matrix. A row of NaNs is add to make the output dimensions (24,9) as '_' does not occur in these sequences.

    Parameters
    ----------
    directory_path : str
        Path to a directory containing Proteome Discoverer-exported CSV files.
        Files must include "Annotated Sequence" and "Modifications" columns.
    dimension1 : int, optional
        Number of positions (columns) in the output matrix (default 9).
    dimension2 : int, optional
        Number of amino acid rows in the output matrix, including the final
        NaN row (default 24).

    Returns
    -------
    numpy.ndarray
        Array of shape (dimension2, dimension1) containing amino acid counts
        for the proteome control dataset, with the last row filled with NaNs.
        Row order is:
            ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
             's', 't', 'y', '_']
        where the final '_' row is NaNs.
    """
    peptidecount = np.ones((dimension1, dimension2-1))

    total_letter_counts = Counter()

    ordered_letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                       's', 't', 'y']

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)

            allpeptides, phosphopeptides = readMod(file_path, 'Phospho')
            lowersitesall = lowercaseModSites(phosphopeptides, 'Phospho')

            for sequence in lowersitesall["Annotated Sequence"]:
                total_letter_counts.update(sequence)

    counts = [total_letter_counts[letter] for letter in ordered_letters]
    counts_array = np.array(counts)

    control_counts_matrix = (peptidecount * counts_array).T  # shape (23, 9)

    # Add an extra row of NaNs at the bottom
    control_counts_matrix_with_nan = np.vstack([
        control_counts_matrix,
        np.full((1, dimension1), np.nan)
    ])  # shape (24, 9)

    return control_counts_matrix_with_nan

def calculatePhosphoFrequency(countsarray):
    """
    Calculates the frequency of each amino acid at the central position (Pos0)
    from a positional counts matrix.

    Takes a (24, 9) array of amino acid counts, where rows correspond to amino
    acids in the fixed order:
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
         's', 't', 'y', '_']
    and columns correspond to positional offsets from the modified residue:
        Pos-4, Pos-3, Pos-2, Pos-1, Pos0, Pos1, Pos2, Pos3, Pos4.

    The function normalizes counts in each column to frequencies, then returns
    the frequency distribution at the central position (column index 4, Pos0).

    Parameters
    ----------
    countsarray : numpy.ndarray
        2D array of shape (24, 9) containing amino acid counts.

    Returns
    -------
    numpy.ndarray
        1D array of length 24 containing the frequency of each amino acid at
        the central position (Pos0).
    """
    countstotal = np.sum(countsarray, axis = 0)
    countsfreq = countsarray/countstotal
    phospho_freq = countsfreq[:, 4]

    return phospho_freq

def alignedCountDirectory(
    directory_path,
    modification,
    central_residues = "STY",
    require_proline_at_plus1=False,
    exclude_proline_at_plus1=False,
    two_mod_mode=False,
    modification2=None
    ):

    """
    Processes all CSV files in a directory to compute amino acid occurrence counts
    for aligned sequence windows centered on a given modification.

    For each CSV file in `directory_path`, the function:
        1. Reads peptide data and filters for those containing `modification`.
        2. Converts modified sites in sequences to lowercase.
        3. Aligns each modified site within a fixed-length window (±4 residues).
        4. Counts the occurrence of each amino acid (including lowercase
           modified residues and padding) at each position in the window.
        5. Accumulates counts across all files.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing CSV files to process.
    modification : str
        Modification of interest as it appears in the "Modifications" column
        (e.g., "Phospho" or "Dehydrated").
    central_residues : str, optional
        One or more single-letter amino acid codes considered central residues
        for alignment (default "STY").
    require_proline_at_plus1 : bool, optional
        If True, only include sites where the residue at +1 is proline ('P').
        Default is False.
    exclude_proline_at_plus1 : bool, optional
        If True, exclude sites where the residue at +1 is proline ('P').
        Default is False.
    two_mod_mode : bool, optional
        If True, will lowercase sites of two different modifications. The second modification must also be specified. Used mainly for analysis of phosphosites in positions flanking b-elimination sites. Default is False.
    modification2 : str
        Second modification of interest as it appears in the "Modifications" column
        (e.g., "Phospho" or "Dehydrated").

    Returns
    -------
    numpy.ndarray
        A 24 × 9 array of counts where rows correspond to amino acids in the order:
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
         's', 't', 'y', '_'],
        and columns correspond to positions -4 through +4 relative to the
        central residue. Each element contains the total count aggregated
        across all processed files.
    """
    combined_counts = None

    if two_mod_mode:
        modification2 = modification2
    else:
        modification2 = modification

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)

            allpeptides, phosphopeptides = readMod(file_path, modification)
            #first lowercase mod of interest
            lowersites_mod = lowercaseModSites(phosphopeptides, modification)
            #then lowercase other modification
            lowersitesall = lowercaseModSites(lowersites_mod, modification2)
            alignedlowersitessitesall = alignLowerModSitesSTY(lowersitesall, modification, central_residues, require_proline_at_plus1, exclude_proline_at_plus1)
            siteAAoccurenceall = modCountAAOccurrence(alignedlowersitessitesall)

            if combined_counts is None:
                combined_counts = siteAAoccurenceall
            else:
                combined_counts += siteAAoccurenceall

    return combined_counts

def zScore(
    sample,
    control,
    use_control_variance=True,
    output_file=None,
    row_labels=None,
    col_labels=None
):

    """
    Calculates Z-scores for differences in proportions between two datasets.

    For each position (column) in the input arrays, the function computes the
    frequency of events in `sample` and `control`, calculates the standard error
    for each proportion, combines them according to the specified variance model,
    and then computes a Z-score:

        Z = (p_sample - p_control) / SE

    Parameters
    ----------
    sample : numpy.ndarray
        2D array of counts for the sample condition. Columns correspond to
        positions, rows to amino acids.
    control : numpy.ndarray
        2D array of counts for the control condition. Columns correspond to
        positions, rows to amino acids. Must have the same shape as `sample`.
    use_control_variance : bool, optional
        If True (default), uses the full two-proportion standard error that
        includes both sample and control variance:

            SE_combined = sqrt( (p_sample * (1 - p_sample) / n_sample)
                                + (p_control * (1 - p_control) / n_control) )

        If False, uses only the control proportion to estimate variance,
        scaled by the sample size:

            SE = sqrt( p_control * (1 - p_control) / n_sample )

        The latter assumes that the control proportion is a fixed value based on a very large number of measurments (single-proportion Wald Z-test).
    output_file : str or None, optional
        Path to save the resulting Z-score array as a CSV file. If None (default),
        no file is saved.
    row_labels : list of str or None, optional
        Labels for rows. Can be used to provide custom row labels. If None, defaults to:
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
         's', 't', 'y', '_'].
    col_labels : list of str or None, optional
        Labels for columns. Can be used to provide custom column labels. If None, defaults to:
        ['Pos-4', 'Pos-3', 'Pos-2', 'Pos-1', 'Pos0', 'Pos1', 'Pos2', 'Pos3', 'Pos4'].

    Returns
    -------
    numpy.ndarray
        Array of Z-scores with the same shape as the input arrays.
        Positive values indicate enrichment in `sample`, negative values
        indicate enrichment in `control`.

    Notes
    -----
    - This function assumes counts are large enough for the normal approximation
      to hold when converting differences in proportions to Z-scores.
    """

    sampletotal = np.nansum(sample, axis=0) #axis=0 sums the columns, axis=1 sums over rows
    print(sampletotal)
    samplefreq = sample/sampletotal
    controltotal = np.nansum(control, axis = 0)
    print(controltotal)
    controlfreq = control/controltotal
    sample_se = np.sqrt(samplefreq*(1-samplefreq)*((1/sampletotal)))

    if use_control_variance:
        control_se = np.sqrt(controlfreq * (1 - controlfreq) / controltotal)
        combined_se = np.sqrt(sample_se**2 + control_se**2)
    else:
        combined_se = np.sqrt(controlfreq * (1 - controlfreq) / sampletotal)
    zscore = ((samplefreq-controlfreq)/combined_se)

    if row_labels is None:
        row_labels = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
            's', 't', 'y', '_'
        ]
    if col_labels is None:
        col_labels = [f"Pos{i}" for i in range(-4, 5)]

    if output_file is not None:
        df = pd.DataFrame(zscore, index=row_labels, columns=col_labels)
        df.to_csv(output_file)

    return zscore


def plotZscoreHeatmap(
    inputarray,
    pdfname,
    alpha=0.05,
    highlim=None,
    lowlim=None,
    save_pvals_file=None,
    fdr_method="fdr_bh",
    row_labels=None,
    col_labels=None,
    draw_boxes=True
):

    """
    Plot a z-score heatmap and (optionally) outline cells that are significant after FDR correction.

    NaN and infinite values in `inputarray` are replaced with 0.0 for plotting and
    p-value derivation, but their positions are tracked so the returned p-values
    and rejected mask contain NaNs in those locations.

    Parameters
    ----------
    inputarray : ndarray
        2D array of z-scores (e.g., 24 × 9).
    pdfname : str
        Output path for the heatmap PDF.
    alpha : float, optional
        FDR threshold for significance (default 0.05).
    highlim, lowlim : float or None, optional
        Color limits; if None, set symmetrically around 0 based on data.
    save_pvals_file : str or None, optional
        If set, specifies the output path for saving the FDR-adjusted p-values as a labeled CSV.
    fdr_method : str, optional
        Method for `multipletests` (default "fdr_bh").
    row_labels : list of str or None, optional
        Labels for rows. Can be used to provide custom row labels. If None, defaults to:
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
         's', 't', 'y', '_'].
    col_labels : list of str or None, optional
        Labels for columns. Can be used to provide custom column labels. If None, defaults to:
        ['Pos-4', 'Pos-3', 'Pos-2', 'Pos-1', 'Pos0', 'Pos1', 'Pos2', 'Pos3', 'Pos4'].
    draw_boxes : bool, optional
        If True (default), draws a black rectangle around cells where
        FDR-adjusted p < alpha.

    Returns
    -------
    rejected_matrix : ndarray of bool
        True where adjusted p < alpha.
    pvals_adj : ndarray of float
        FDR-adjusted p-values (NaN where input was non-finite).
    """

    aminoacids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                  's', 't', 'y', '_']

# Keep track invalid values
    invalid_mask = ~np.isfinite(inputarray)

    # Replace NaN/inf with 0 for plotting/pvals
    safe_array = np.nan_to_num(inputarray, nan=0.0, posinf=0.0, neginf=0.0)

    if highlim is None or lowlim is None:
        maxvalue = np.nanmax(safe_array)
        minvalue = np.nanmin(safe_array)

    # Set symmetric color limits around zero
        if abs(maxvalue) >= abs(minvalue):
            highlim = abs(maxvalue)
            lowlim = -highlim
        else:
            highlim = abs(minvalue)
            lowlim = -highlim

    #Calculate pvals
    pvals = 2 * (1 - norm.cdf(np.abs(safe_array)))

    finite_mask = np.isfinite(pvals)

    pvals_adj = np.full_like(pvals, np.nan, dtype=float)
    rejected_matrix = np.zeros_like(pvals, dtype=bool)

    if np.any(finite_mask):
        p_flat = pvals[finite_mask].ravel()
        rejected_flat, p_adj_flat, _, _ = multipletests(p_flat, alpha=alpha, method=fdr_method)
        rejected_matrix[finite_mask] = rejected_flat
        pvals_adj[finite_mask] = p_adj_flat

    # Restore NaN positions from invalid entries
    pvals_adj[invalid_mask] = np.nan
    rejected_matrix[invalid_mask] = False

    # Plot the heatmap
    nan_mask = ~np.isfinite(inputarray)

    # Set up colormap with NaN color
    cmap = plt.get_cmap('RdBu')
    cmap.set_bad(color="#CCCCCC")  # light gray for NaN

    fig, ax = plt.subplots()
    sns.heatmap(
        safe_array,
        mask=nan_mask,
        vmin=lowlim, vmax=highlim,
        yticklabels=aminoacids,
        square=True,
        cmap='RdBu',
        ax=ax,
        cbar_kws={'label': 'Z-score'}
        )

    # Rotate y-axis labels
    plt.yticks(rotation=0)

    # Adjust y-limits so the top and bottom boxes aren’t cut off
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 1, top - 1)

    # Draw boxes where BH-adjusted p-value < alpha
    if draw_boxes:
        num_rows, num_cols = inputarray.shape
        for i in range(num_rows):     # y-axis (residues)
            for j in range(num_cols): # x-axis (positions)
                if rejected_matrix[i, j]:
                    rect = Rectangle((j, i), 1, 1, linewidth=0.7,
                                     edgecolor='black', facecolor='none')
                    ax.add_patch(rect)

    # Save to file
    fig.savefig(pdfname, bbox_inches='tight')
    plt.close(fig)

    # Save adjusted p-values if needed
# Defaults
    if row_labels is None:
        row_labels = [
            'A','C','D','E','F','G','H','I','K','L',
            'M','N','P','Q','R','S','T','V','W','Y',
            's','t','y','_'
        ]
    if col_labels is None:
        col_labels = [f"Pos{i}" for i in range(-4, 5)]

    if save_pvals_file is not None:
        df = pd.DataFrame(pvals_adj, index=row_labels, columns=col_labels)
        df.to_csv(save_pvals_file)

    return rejected_matrix, pvals_adj

def plotZscoreHeatmapBubble(zscorearray, phospho_freq, pdfname):
    aminoacids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','s','t','y','_']  # ← add '_' to make 24
    masked_array_all = np.nan_to_num(zscorearray, nan=0.0, posinf=0.0, neginf=0.0)
    excluded_array = np.delete(masked_array_all, 4, axis=1)

    maxvalue = np.nanmax(excluded_array)
    minvalue = np.nanmin(excluded_array)
    highlim = max(abs(maxvalue), abs(minvalue))
    lowlim  = -highlim

    mask = np.zeros_like(masked_array_all, dtype=bool)
    mask[:, 4] = True

    fig, ax = plt.subplots()
    sns.heatmap(masked_array_all, cmap="RdBu", center=0, mask=mask,
                vmin=lowlim, vmax=highlim, yticklabels=aminoacids,
                square=True, ax=ax, cbar_kws={'label': 'Z-score'})

    max_bubble_size = 200
    phospho_freq = np.asarray(phospho_freq, dtype=float)
    if zscorearray.shape[0] == 24 and phospho_freq.shape[0] == 24:
        phospho_freq = phospho_freq[:23]  # drop '_' row

    denom = np.nanmax(phospho_freq)
    bubble_sizes = np.zeros_like(phospho_freq) if denom == 0 else (phospho_freq / denom) * max_bubble_size

    y_positions = np.arange(len(phospho_freq))
    x_position = np.full(len(phospho_freq), 4)
    ax.scatter(x_position + 0.5, y_positions + 0.5, s=bubble_sizes,
               color='gray', alpha=0.7, edgecolors="black")

    legend_sizes = [0.01, 0.1, 0.5, 1.0]
    scale = (max_bubble_size / denom) if denom != 0 else 0
    legend_bubbles = [plt.scatter([], [], s=s * scale, color='grey', alpha=0.6, edgecolors='black') for s in legend_sizes]
    fig.legend(legend_bubbles, [f"{s}" for s in legend_sizes],
               title="Phosphosite Frequency", bbox_to_anchor=(1.1, 1.1),
               bbox_transform=fig.transFigure, loc='upper right')

    plt.yticks(rotation=0)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    fig.savefig(pdfname)


def score_sequence(sequence, df, sd_cutoff=3):
    """
    Identifies rows in a DataFrame whose summed PSSM scores for a given 9-mer sequence
    exceed mean + (sd_cutoff × standard deviation).

    Parameters
    ----------
    sequence : str
        9-character sequence like "____s____" (Pos-4..Pos+4). '_' is a padding character.
        Index 4 is the central residue and is ignored here.
    df : pandas.DataFrame
        Rows are entities to score (e.g., kinases). Columns are named like
        "-4A", "-3S", "-2T", "-1Y", "1P", "2D", "3E", "4K".
    sd_cutoff : float
        Threshold multiplier: keep rows with Row_Sums > mean + sd_cutoff*std.

    Returns
    -------
    list
        Index labels of rows whose Row_Sums exceed the threshold.
    """
    columns_to_extract = []
    if sequence[0] != '_':
        columns_to_extract.append(f'-4{sequence[0]}')
    if sequence[1] != '_':
        columns_to_extract.append(f'-3{sequence[1]}')
    if sequence[2] != '_':
        columns_to_extract.append(f'-2{sequence[2]}')
    if sequence[3] != '_':
        columns_to_extract.append(f'-1{sequence[3]}')
    if sequence[5] != '_':  # Assuming columns are in sequence from -1 to +4
        columns_to_extract.append(f'1{sequence[5]}')
    if sequence[6] != '_':
        columns_to_extract.append(f'2{sequence[6]}')
    if sequence[7] != '_':
        columns_to_extract.append(f'3{sequence[7]}')
    if sequence[8] != '_':
        columns_to_extract.append(f'4{sequence[8]}')

    valid_columns = [col for col in columns_to_extract if col in df.columns]
    new_df = df[valid_columns]

    new_df = new_df.assign(Row_Sums=new_df.sum(axis=1, min_count=1))

    mean_row_sums = new_df['Row_Sums'].mean()
    std_row_sums = new_df['Row_Sums'].std()

    # Calculate the threshold
    threshold = mean_row_sums + sd_cutoff * std_row_sums

    # Find the row names where Row Sums are greater than the threshold
    rows_above_threshold = new_df[new_df['Row_Sums'] > threshold]

    # Get the row names (index) of those rows
    row_names_above_threshold = rows_above_threshold.index.tolist()

    return row_names_above_threshold

def score_and_threshold(input_dir, kinase_type, sd_cutoff=3):
    """
    Score aligned phosphosite sequences against a position-specific scoring matrix (PSSM) and report sequences with scores above a standard deviation threshold.

    This function processes all `.csv` peptide identification files in a directory, aligns phosphorylated sites to a 9-amino-acid window, scores them using a kinase-specific PSSM, and writes out the sequences and any PSSM rows that exceed the mean + (sd_cutoff × standard deviation) threshold.

    Workflow:
    1. Select the appropriate PSSM file and target residue set based on `kinase_type`:
       - 'st': serine/threonine kinases (PSSM from `kin_pssm.csv`, sites = {'s', 't'})
       - 'y' : tyrosine kinases (PSSM from `tk_pssm.csv`, sites = {'y'})
    2. For each `.csv` file in `input_dir`:
       a. Parse phosphopeptides and convert modified residues to lowercase.
       b. Align each site to a 9-aa window using `alignLowerModSitesSTY`.
       c. Save the aligned sequences to an `_aligned.txt` file in the output directory.
       d. For each aligned sequence, run `score_sequence` to find PSSM rows with scores above threshold.
       e. Save the sequence and its above-threshold PSSM row names to a `_results.txt` file.

    Parameters
    ----------
    input_dir : str
        Path to directory containing `.csv` peptide identification files.
    kinase_type : {'st', 'y'}
        Type of kinase motif to score:
        - 'st': serine/threonine kinases (uses kin_pssm.csv)
        - 'y' : tyrosine kinases (uses tk_pssm.csv)
    sd_cutoff : float, optional
        Multiplier for the standard deviation used to set the score threshold.
        Default is 3.

    Outputs
    -------
    Creates a new directory named:
        "<input_dir>_pssm_scores_<kinase_type>_<sd_cutoff>sd"
    containing:
        - `_aligned.txt` files: aligned 9-aa windows centered on phosphosites
        - `_results.txt` files: sequences and their above-threshold PSSM row names

    Notes
    -----
    - Requires `kin_pssm.csv` and/or `tk_pssm.csv` in the current working directory.
    - Assumes columns in the PSSM are named with positional offsets and amino acid codes
      (e.g., '-4S', '2P') as expected by `score_sequence`.
    - Only sequences of length 9 with the correct central residue are scored.
    """

    input_dir = input_dir.rstrip("/")
    output_dir=input_dir + '_pssm_scores' + '_'+ kinase_type + '_'+ str(sd_cutoff) + 'sd'
    os.makedirs(output_dir, exist_ok=True)

    path_to=os.path.dirname(__file__)
    if kinase_type == 'st':
        df = pd.read_csv(f'{path_to}/pssms/kin_pssm.csv', index_col=0)
        sites = ['s','t']
    elif kinase_type == 'y':
        df = pd.read_csv(f'{path_to}/pssms/tk_pssm.csv', index_col=0)
        sites = ['y']

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            input_file_path = os.path.join(input_dir, file_name)

            allpeptides, phosphopeptides = readMod(input_file_path, 'Phospho')
            lowersitesall = lowercaseModSites(phosphopeptides, 'Phospho')
            alignedlowersitessitesall = alignLowerModSitesSTY(lowersitesall, 'Phospho')

            aligned_file_name = file_name.replace('.csv', '_aligned.txt')
            aligned_file_path = os.path.join(output_dir, aligned_file_name)
            with open(aligned_file_path, 'w') as aligned_file:
                for item in alignedlowersitessitesall:
                    aligned_file.write(f"{item}\n")

            with open(aligned_file_path, 'r') as file:
                sequences = [line.strip() for line in file if line.strip()]

            results = {}

            for seq in sequences:
                if len(seq) == 9 and seq[4] in sites:
                    results[seq] = score_sequence(seq, df, sd_cutoff)

            output_file_name = file_name.replace('.csv', '_results.txt')
            output_file_path = os.path.join(output_dir, output_file_name)

            with open(output_file_path, 'w') as f:
                for seq, row_names in results.items():
                    f.write(f'Sequence: {seq}\nRow Names Above Threshold: {row_names}\n\n')

            print(f'Processed {file_name}. Results written to {output_file_name}.')


def kinase_count(results_file, output_dir, mapping_file=None):
    """
    Aggregate and reformat kinase PSSM hit counts from a scored results file.

    This function reads the output from `score_and_threshold`, counts how often
    each PSSM row name appears across all sequences, optionally remaps the names
    using a `mapping_file`, and writes the aggregated counts to a tab-delimited file.

    Workflow:
    1. Load the mapping file (`kinase_mapping.csv` by default) with columns:
       - 'current_format' : original PSSM row names from the scoring step
       - 'new_format'     : reformatted names for output
    2. Parse the `results_file` to extract all PSSM row names listed under
       "Row Names Above Threshold" for each sequence.
    3. Count occurrences of each row name across all sequences.
    4. Replace original names with `new_format` names if a mapping is found. This was implemented for compatibility with the CORAL tool from the Phanstiel lab.
    5. Write the mapped counts to `<sequence_prefix>_converted_counts.txt` in `output_dir`.

    Parameters
    ----------
    results_file : str
        Path to the `_results.txt` file produced by `score_and_threshold`.
    output_dir : str
        Path to directory where the output counts file will be saved.
    mapping_file : str, optional
        CSV file with mapping between original and reformatted PSSM row names.
        Must contain columns 'current_format' and 'new_format'.
        Defaults to 'kinase_mapping.csv'.

    Outputs
    -------
    Creates a file `<basename>_converted_counts.txt` in `output_dir` containing:
        - Reformatted PSSM row names (or original names if no mapping found)
        - Their total occurrence counts across all sequences

    Notes
    -----
    - The `results_file` must follow the exact text format generated by `score_and_threshold`, where sequences start with "Sequence:" and hits start with "Row Names Above Threshold:".
    - Row name mapping is optional; unmapped names are kept as-is.
    """
    if mapping_file==None:
        path_to=os.path.dirname(__file__)
        mapping_file=f"{path_to}/pssms/kinase_mapping.csv"

    mapping_df = pd.read_csv(mapping_file)
    row_title_mapping = pd.Series(mapping_df['new_format'].values, index=mapping_df['current_format']).to_dict()

    results_dict = {}
    with open(results_file, 'r') as file:
        lines = file.readlines()
        current_sequence = None
        for line in lines:
            if line.startswith('Sequence:'):
                current_sequence = line.strip().split('Sequence: ')[1]
                results_dict[current_sequence] = []
            elif line.startswith('Row Names Above Threshold:'):
                row_names = line.strip().split('Row Names Above Threshold: ')[1]
                if current_sequence:
                    results_dict[current_sequence].extend(row_names.strip("[]").replace("'", "").split(", "))

    all_row_titles = []
    for row_titles in results_dict.values():
        all_row_titles.extend(row_titles)
    row_title_counts = Counter(all_row_titles)

    converted_row_title_counts = {}
    for row_title, count in row_title_counts.items():
        if row_title in row_title_mapping:
            new_title = row_title_mapping[row_title]
        else:
            new_title = row_title  # Keep the same if no mapping found
        converted_row_title_counts[new_title] = count

    output_file_name = os.path.basename(results_file).replace('_results.txt', '_converted_counts.txt')
    output_file_path = os.path.join(output_dir, output_file_name)
    with open(output_file_path, 'w') as f:
        for row_title, count in converted_row_title_counts.items():
            f.write(f"{row_title}\t{count}\n")

    print(f"Processed {results_file}. Results written to {output_file_path}")


def kinase_count_directory(input_dir, mapping_file=None):
    """
    Batch-process multiple kinase PSSM scoring results files to aggregate hit counts.

    Iterates over all `_results.txt` files in the specified directory (produced by
    `score_and_threshold`), runs `kinase_count` on each file, and saves the converted
    hit counts in a new output directory.

    The output directory is created automatically and named:
        <input_dir>_kinase_count

    Parameters
    ----------
    input_dir : str
        Path to the directory containing `_results.txt` files from `score_and_threshold`.
    mapping_file : str, optional
        CSV file with mapping between original and reformatted PSSM row names.
        Must contain columns 'current_format' and 'new_format'.
        If None -> Defaults to 'kinase_mapping.csv'.

    Outputs
    -------
    For each `_results.txt` file in `input_dir`, creates a corresponding
    `<basename>_converted_counts.txt` in `<input_dir>_kinase_count`.

    Notes
    -----
    - This function is for batch processing with `kinase_count` and assumes the input
      directory only contains results files with the expected naming convention.
    - The row name mapping step is optional; unmapped names are kept as-is.
    """

    if mapping_file==None:
        path_to=os.path.dirname(__file__)
        mapping_file=f"{path_to}/pssms/kinase_mapping.csv"

    input_dir = input_dir.rstrip("/")
    output_dir=input_dir + '_kinase_count'
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith('_results.txt'):
            results_file_path = os.path.join(input_dir, file_name)
            kinase_count(results_file_path, output_dir, mapping_file)

def average_replicate_counts(input_dir):
    """
    Calculates average kinase hit counts across multiple replicate results files.

    Reads all `_converted_counts.txt` files in the specified directory (produced by
    `kinase_count_directory`), combines them, and computes the mean count for each
    kinase across all files. Saves the results as a tab-delimited text file.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing `_converted_counts.txt` files. These files
        should each have two tab-separated columns: kinase name and count.

    Outputs
    -------
    average_counts_across_files.txt
        Tab-delimited file with two columns:
        - 'Kinase' : kinase name (str)
        - 'Average_Count' : mean count across all input files (float)

    Notes
    -----
    - Any kinases missing from some files will still be included; counts are averaged
      over the files in which they appear.
    """
    all_data = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('_converted_counts.txt'):
            file_path = os.path.join(input_dir, file_name)

            df = pd.read_csv(file_path, sep='\t', header=None, names=['Kinase', 'Count'])

            df['Source_File'] = file_name

            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    averaged_df = combined_df.groupby('Kinase', as_index=False)['Count'].mean()

    output_file = os.path.join(input_dir, 'average_counts_across_files.txt')
    averaged_df.to_csv(output_file, sep='\t', index=False, header=['Kinase', 'Average_Count'])

    print(f"Averaged counts written to {output_file}")

def compile_replicate_counts(input_dir):
    """
    Combine kinase count results from multiple replicates into a single table.

    Reads all `_converted_counts.txt` files in the specified directory
    (produced by `kinase_count_directory`), merges them on the 'Kinase' column,
    and fills in missing values with zeros. The resulting table contains one
    column per input file.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing `_converted_counts.txt` files. These files
        should each have two tab-separated columns: kinase name and count.

    Outputs
    -------
    compiled_counts.txt
        Tab-delimited file with:
        - 'Kinase' : kinase name (str)
        - One column per replicate file, containing integer counts.

    Notes
    -----
    - Missing kinases in any file are assigned a count of 0 for that replicate.
    - Column headers (after 'Kinase') are the original filenames of the input files.
    """

    compiled_df = pd.DataFrame()

    for file_name in os.listdir(input_dir):
        if file_name.endswith("_converted_counts.txt"):
            file_path = os.path.join(input_dir, file_name)

            df = pd.read_csv(file_path, sep="\t", header=None, names=["Kinase", file_name])

            if compiled_df.empty:
                compiled_df = df
            else:
                compiled_df = pd.merge(compiled_df, df, on="Kinase", how="outer")

    compiled_df.fillna(0, inplace=True)
    for col in compiled_df.columns[1:]:
        compiled_df[col] = compiled_df[col].astype(int)

    output_file_path = os.path.join(input_dir, "compiled_counts.txt")
    compiled_df.to_csv(output_file_path, sep="\t", index=False)

    print(f"Compiled counts written to {output_file_path}")
