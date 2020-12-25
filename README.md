# ncER_datasets

-   **Supplementary_Table_S1_FeatureDescription_20181009.xlx** contains the description and accession links of all input features used in the model. The file was created on October 10th 2018.
-   **xgboost_github_v1.py** contains the python script used to train the XGBoost model (March 6th 2019).
-   **ncER version 1** ([ncER_10bpBins_percentile_version1.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v1/ncER_10bpBins_percentile_version1.txt.gz)) contains the genome-wide percentiles (10bp bins) of the ncER scores created on August 13th 2018. The coordinates are mapped to hg19.
-   **ncER version2** are available in two formats:
    -   **Bin_1bp**: 1 file per chromosome, with the 1bp resolution genome-wide percentile of the ncER scores created on March 6th 2019. The coordinates are mapped to hg19.
    -   **Bin_10bp**: 1 file ([ncER_10bpBins_allChr_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_10bp/ncER_10bpBins_allChr_coordSorted.txt.gz)) with the 10bp resolution genome-wide percentile of the ncER scores created on March 6th 2019. The coordinates are mapped to hg19.

## Download Links

| Version | Link                                                                                                                                            | Description                              |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| v1      | [ncER_10bpBins_percentile_version1.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v1/ncER_10bpBins_percentile_version1.txt.gz)        | (historical) v1 - 10bp - all chromosomes |
| v2      | [ncER_10bpBins_allChr_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_10bp/ncER_10bpBins_allChr_coordSorted.txt.gz) | v2 - 10bp - all chromosomes              |
| v2      | [ncER_perc_chr1_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr1_coordSorted.txt.gz)              | v2 - 1bp - chromosome 1                  |
| v2      | [ncER_perc_chr2_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr2_coordSorted.txt.gz)              | v2 - 1bp - chromosome 2                  |
| v2      | [ncER_perc_chr3_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr3_coordSorted.txt.gz)              | v2 - 1bp - chromosome 3                  |
| v2      | [ncER_perc_chr4_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr4_coordSorted.txt.gz)              | v2 - 1bp - chromosome 4                  |
| v2      | [ncER_perc_chr5_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr5_coordSorted.txt.gz)              | v2 - 1bp - chromosome 5                  |
| v2      | [ncER_perc_chr6_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr6_coordSorted.txt.gz)              | v2 - 1bp - chromosome 6                  |
| v2      | [ncER_perc_chr7_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr7_coordSorted.txt.gz)              | v2 - 1bp - chromosome 7                  |
| v2      | [ncER_perc_chr8_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr8_coordSorted.txt.gz)              | v2 - 1bp - chromosome 8                  |
| v2      | [ncER_perc_chr9_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr9_coordSorted.txt.gz)              | v2 - 1bp - chromosome 9                  |
| v2      | [ncER_perc_chr10_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr10_coordSorted.txt.gz)            | v2 - 1bp - chromosome 10                 |
| v2      | [ncER_perc_chr11_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr11_coordSorted.txt.gz)            | v2 - 1bp - chromosome 11                 |
| v2      | [ncER_perc_chr12_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr12_coordSorted.txt.gz)            | v2 - 1bp - chromosome 12                 |
| v2      | [ncER_perc_chr13_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr13_coordSorted.txt.gz)            | v2 - 1bp - chromosome 13                 |
| v2      | [ncER_perc_chr14_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr14_coordSorted.txt.gz)            | v2 - 1bp - chromosome 14                 |
| v2      | [ncER_perc_chr15_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr15_coordSorted.txt.gz)            | v2 - 1bp - chromosome 15                 |
| v2      | [ncER_perc_chr16_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr16_coordSorted.txt.gz)            | v2 - 1bp - chromosome 16                 |
| v2      | [ncER_perc_chr17_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr17_coordSorted.txt.gz)            | v2 - 1bp - chromosome 17                 |
| v2      | [ncER_perc_chr18_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr18_coordSorted.txt.gz)            | v2 - 1bp - chromosome 18                 |
| v2      | [ncER_perc_chr19_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr19_coordSorted.txt.gz)            | v2 - 1bp - chromosome 19                 |
| v2      | [ncER_perc_chr20_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr20_coordSorted.txt.gz)            | v2 - 1bp - chromosome 20                 |
| v2      | [ncER_perc_chr21_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr21_coordSorted.txt.gz)            | v2 - 1bp - chromosome 21                 |
| v2      | [ncER_perc_chr22_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chr22_coordSorted.txt.gz)            | v2 - 1bp - chromosome 22                 |
| v2      | [ncER_perc_chrX_coordSorted.txt.gz](https://telentilab-dataset.s3.amazonaws.com/ncER/v2/Bin_1bp/ncER_perc_chrX_coordSorted.txt.gz)              | v2 - 1bp - chromosome X                  |

All ncER files have the following structure :

-   Column 1 is the chromosome (autosomes and chromosome X; ordered alpha-numerically).
-   Column 2 is the start position of the bin (where the first base in a chromosome is numbered 0; the start base is included)
-   Column 3 is the end position of the bin (where the first base in a chromosome is numbered 0; the end base is not included)
-   Column 4 is genome-wide ncER percentile. The higher the percentile, the more likely essential (in terms of regulation) the region is.
