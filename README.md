# ncER_datasets

Supplementary_Table_S1_FeatureDescription_20181009.xlx contains the description and accession links of all input features used in the model.
The file was created on October 10th 2018.

ncER_10bpBins_percentile_version1.txt.gz ([https://ncer.telentilab.com](https://ncer.telentilab.com)) contains the genome-wide percentiles (10bp bins) of the ncER scores created on August 13th 2018. The coordinates are mapped to hg19.

the ncER_version2 folder contains two subfolders :
Bin_1bp subfolder contains 1 file per chromosome, with the 1bp resolution genome-wide percentile of the ncER scores created on March 6th 2019. The coordinates are mapped to hg19.
Bin_10bp subfolder contains 1 file with the 10bp resolution genome-wide percentile of the ncER scores created on March 6th 2019. The coordinates are mapped to hg19.

All ncER files have the following structure :
Column 1 is the chromosome (autosomes and chromosome X; ordered alpha-numerically).
Column 2 is the start position of the bin (where the first base in a chromosome is numbered 0; the start base is included)
Column 3 is the end position of the bin (where the first base in a chromosome is numbered 0; the end base is not included)
Column 4 is genome-wide ncER percentile. The higher the percentile, the more likely essential (in terms of regulation) the region is.
