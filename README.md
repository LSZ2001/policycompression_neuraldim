# policycompression_neuraldim
 Code for the manuscript "Neural and behavioral signatures of policy compression in cognitive control". The authors are Shuze Liu, Atsushi Kikumoto, David Badre, and Samuel J. Gershman. 

- **Bold** words are .m filenames.
- *Italicized* words are .mat or .tex filenames or directory folder names.

The code runs on Matlab R2023a.


## Main folder
- **manuscript_figures.m** creates all figures for the manuscript, based on the saved behavioral and EEG dimensionality files.
- **compute_policycost.m** computes policy cost for each participant in each block separately. It outputs **T_policycost_Paforeachblock.txt**.
- **compute_neural_dimensionality.m** computes decoding accuracies, based on raw decoding accuracy files at https://neurodata.riken.jp/id/20240831-001. It outputs **A304_DIM_READOUT_RL_processed.txt** and similar files for each individual participant.

### *BEH* subfolder
- **HAL2017_READOUT_TEST_BehP.txt** is the raw behavior file for the sampled SOA phase analyzed. It is from https://neurodata.riken.jp/id/20240831-001.
- **T_policycost_Paforeachblock.txt** additionally include policy costs for each trial.

### *EEG_paper_analysis* subfolder
#### *Dimension* subsubfolder
- It contains raw decoding files **A304_DIM_READOUT_RL_B_pred_DIM.h5** and similarly named files for other participants (304 is one of the participant's ID). These files are too large to be kept in this Github repository, and could be found at https://neurodata.riken.jp/id/20240831-001. 
#### *new_dimensionality_files* subsubfolder
- It contains processed files **A304_DIM_READOUT_RL_processed.txt** and similarly named files for other participants, to be used by **manuscript_figures.m**.

### *figs* subfolder
- It contains saved figures.
