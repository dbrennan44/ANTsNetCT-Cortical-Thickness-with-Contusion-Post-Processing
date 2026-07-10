<p align="center">
  <img src="./Picture1.png" alt="Process_figure" width="100%">
</p>

Visual depiction of each lesion masking strategy. A) Focal lesion evident on structural MRI. Contusion core (Red Arrow) and edema (Green Arrow) are identified and masked in subsequent masking strategies. B) Cortical thickness map with lesion included in the estimation (No Masking). Note, the location and relative intensity of the contusion edema has caused misattribution of these voxels as gray matter and estimation of cortical thickness in this area (Green Arrow). C) Desikan-Killiany-Tourville parcellation with lesion mask removed, which excludes data from within the lesion at the level of data summarization (Atlas Masking). (D) Tissue segmentation, cortical thickness and cortical parcellation maps with lesion fully excluded (Full Masking). In this strategy, the lesion mask is removed from tissue segmentation.


# ANTsNetCT-Cortical-Thickness-with-Contusion-Post-Processing
Post-processing script that utilizes contusion mask and tissue priors generated with ANTsNetCT. Automatically finds contusion mask in BIDS dir and applies contusion masking prior to cortical thickness estimation with ANTsPyNet/ANTsNetCT tools.

paper: Brennan D, Schneider ALC, Shinohara RT, Diaz-Arrastia R, Cook PA, Gee JC, Gugger JJ. Contusions bias cortical thickness estimates after traumatic brain injury: A TRACK-TBI study. Neuroimage Clin. 2026;50:104003. doi: 10.1016/j.nicl.2026.104003. Epub 2026 May 6. PMID: 42114207; PMCID: PMC13191617.
