#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:17:26 2025

@author: Amy
"""

from phospropel import alignedCountDirectory, zScore, plotZscoreHeatmap

control_counts = alignedCountDirectory("data/spvc_perlib_ptmrs", 'Phospho', central_residues = "ST", two_mod_mode=False)

#lowercase and count dehydrated sites
dehydrated_only_count = alignedCountDirectory("data/spvc_perlib_ptmrs", 'Dehydrated', central_residues = "ST", two_mod_mode=False)

#hide central column for downstream analysis
dehydrated_only_count[:, 4] = 0

#analyze flanking phospho + dehydrated sites
experiment_counts = alignedCountDirectory("data/spvc_perlib_ptmrs", 'Dehydrated', central_residues = "ST", two_mod_mode=True, modification2='Phospho')

#subtract out flanking dehydrated sites (can't know what state these were in when enzyme acted)
experiment_counts[20] = experiment_counts[20] - dehydrated_only_count[20]
experiment_counts[21] = experiment_counts[21] - dehydrated_only_count[21]

experiment_vs_control = zScore(experiment_counts, control_counts)
experiment_vs_control = -experiment_vs_control

pdf = "spvc_perlib_heatmap_Dehydrated_internal.pdf"

plotZscoreHeatmap(experiment_vs_control, pdf, alpha=0.0001)
