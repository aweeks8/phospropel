#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:28:47 2025

@author: Amy
"""

from phospropel import proteomeAbundanceControlCount, calculatePhosphoFrequency, alignedCountDirectory, zScore, plotZscoreHeatmapBubble

control_frequencies = proteomeAbundanceControlCount("data/perlib_ptmrs")
combined_counts = alignedCountDirectory("data/perlib_ptmrs", 'Phospho')

phospho_v_proteomeall = zScore(combined_counts, control_frequencies, use_control_variance=False)
pdfname = "./perlib_heatmap_bubble.pdf"

phospho_freq = calculatePhosphoFrequency(combined_counts)
phospho_freq1 = phospho_freq[:23]
phospho_v_proteome_plot_all = plotZscoreHeatmapBubble(phospho_v_proteomeall, phospho_freq1, pdfname)
