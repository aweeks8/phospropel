#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 17:58:39 2025

@author: Amy
"""

from phospropel import (
    score_and_threshold,
    kinase_count_directory,
    average_replicate_counts,
    compile_replicate_counts,
)


score_and_threshold("data/perlib_ptmrs", kinase_type = 'st', sd_cutoff=3)
kinase_count_directory('data/perlib_ptmrs_pssm_scores_st_3sd')
average_replicate_counts('data/perlib_ptmrs_pssm_scores_st_3sd_kinase_count')
compile_replicate_counts('data/perlib_ptmrs_pssm_scores_st_3sd_kinase_count')
