# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 14:13:54 2025

@author: smcoyle
"""

# Optional convenience re-exports
from .utils import (	readMod, 
						lowercaseModSites,
						alignLowerModSitesSTY, 
						modCountAAOccurrence, 
						proteomeAbundanceControlCount,
						zScore,
						plotZscoreHeatmap,
						plotZscoreHeatmapBubble,
						score_sequence,
						score_and_threshold,
						kinase_count_directory,
						average_replicate_counts,
						compile_replicate_counts,
                        calculatePhosphoFrequency,
                        alignedCountDirectory
					)

__version__ = "0.1.0"

__all__ = [
			"readMod", "lowercaseModSites", "alignLowerModSitesSTY",
			"modCountAAOccurrence", "proteomeAbundanceControlCount",
			"zScore","plotZscoreHeatmap","plotZscoreHeatmapBubble",
			"score_sequence","score_and_threshold","kinase_count_directory",
			"average_replicate_counts","compile_replicate_counts",
            "calculatePhosphoFrequency", "alignedCountDirectory"
			]