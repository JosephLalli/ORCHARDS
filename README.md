## ORCHARDS

This is the repository for data and software used in the analyses for "Antigenic changes to influenza preferentially accumulate during transmission" {citation}

# Overview
--------

    ORCHARDS
    |- README          # the top level description of content
    |
    |- data            # raw and primary data, are not changed once created
    |  |- references/  # reference fasta files and gtf files to be used for sequence alignment and annotations
    |  |- sample_metadata/  # anonymized subject/sample metadata as recieved by RedCap.
    |  |- fastqs/      # raw data, will not be altered. In practice this is where the raw fastq files go. They will need to be downloaded from the SRA. 
    |  |- secondary_analyses/     # secondary analyses and intermediate files generated by the Sniffles2 pipeline. This includes per-sample and per-season consensus sequences, vcf files, and SNPGenie results.
    |
    |- scripts/           # code used to process data
    |  |- primary_analysis/    # Sniffles 2 config file, 
    |  |- secondary_analysis/ # Functions applied to vcf files/SNPGenie results to produce figures
    |  |  |- data_cleaning_scripts # code used to clean SNP calls and generate dataframes
    |  |  |- pre_processing_scripts # code used to perform analyses used in figure generation (eg, )
    |  |  |- figures # code used to generate figures in paper. Includes both notebooks for each figure, and tools created to create graphics/statisical analyses (eg error bars).
    |
    |- results         # all output from workflows and analyses
    |  |- figures/     # manuscript figures
    |  |- dataframes/  # tidy dataframes of cleaned data

    
  --------
# Dependencies

-vcfClass
-Sniffles2
-chartAnnotator
-SNPgenie2

The consensus sequence analysis and antigenic analysis relies on [muscle](http://www.drive5.com/muscle/downloads.htm). It expects the exicutable to be named "muscle". Please change the path to this exicutable in the Makefile.

Also the R analysis relies heavily on the package HIVEr which contains functions that are commonly used in the analysis. That can be found [here](https://github.com/jtmccr1/HIVEr) and can be installed using devtools. 

# Reproducing the analysis

To reproduce the analysis reported in the paper, please do the following:

## Downloading raw data


## Processing the raw data


## Secondary analysis
