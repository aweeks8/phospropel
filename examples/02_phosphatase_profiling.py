#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:45:15 2025

@author: Amy
"""

from phospropel import alignedCountDirectory, zScore, plotZscoreHeatmap

background_counts = alignedCountDirectory("data/lambda_ptmrs_0min", "Phospho")
experiment_counts = alignedCountDirectory("data/lambda_ptmrs_5min", "Phospho")

z = zScore(experiment_counts, background_counts, use_control_variance=True)

pdf = "./lambdapp_5min_heatmap.pdf"
plotZscoreHeatmap(z, str(pdf), alpha=1e-4, highlim=25, lowlim=-25)
