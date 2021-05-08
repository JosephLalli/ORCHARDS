import os
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import json
import ete3
import subprocess
import re
import multiprocessing as mp
from operator import itemgetter
from tqdm import tqdm
from Bio import SeqIO

import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, rgb2hex, to_hex
from matplotlib.collections import PathCollection

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import math

from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import statsmodels.api as sm
from sklearn import linear_model
from statsmodels.formula.api import ols
import scipy.stats as stats



SNP_frequency_cutoff = 0.01
calcBottlenecks = False

#Set Constants
potentialmixed = ['18VR001531', '19VR004455', '19VR003675', '19VR003920', '19VR003675']
#Samples which, based off visual inspection of mutations, seem to be mixed infections.
#This is determined by looking at 10 samples with the most mutations and seeing if those
#mutations, when close together, tend to be linked on the same read.


installDir = '/mnt/d/orchards/h1n1/'

metadatafile = installDir+'metadata.csv'
completemetadatakey= installDir+'completemetadata_key.csv'
expandedMetadata = installDir+'completemetadata.csv'
figures = os.path.join(installDir, 'figures')

commonGlobalfreqsFilelocation = '/mnt/d/orchards'
GlobalH3N2AAfreqsFile = commonGlobalfreqsFilelocation + '/H3N2_AA.json'
GlobalH1N1AAfreqsFile = commonGlobalfreqsFilelocation + '/H1N1_AA.json'
GlobalFluBAAfreqsFile = commonGlobalfreqsFilelocation + '/FluB_AA.json'
GlobalH3N2DNAfreqsFile = commonGlobalfreqsFilelocation + '/H3N2_DNA.json'
GlobalH1N1DNAfreqsFile = commonGlobalfreqsFilelocation + '/H1N1_DNA.json'
GlobalFluBDNAfreqsFile = commonGlobalfreqsFilelocation + '/FluB_DNA.json'

mainSampleFolders = [installDir+'orchards_run19H3N2/A_Singapore_INFIMH-16-0019_2016', 
                     installDir+'orchards_run19H3N2/A_Hong_Kong_4801_2014_EPI834581', 
                     installDir+'orchards_run19/A_Michigan_45_2015_H1N1_18', 
                     installDir+'orchards_run19/A_Michigan_45_2015_H1N1_19',
                     installDir+'orchards_runB/B_Phuket_3073_2013_17',
                     installDir+'orchards_runB/B_Phuket_3073_2013_18']

vcfdirs = [x + '/snp_calls/filtered_snpcalls' for x in mainSampleFolders]
vcffiles = [x+'/all_snps.vcf' for x in vcfdirs]
allSNPsVCFfiles = [x+'/all_snps_with_depths.vcf' for x in vcfdirs]
references = ['A_Singapore_INFIMH-16-0019_2016',
              'A_Hong_Kong_4801_2014_EPI834581',
              'A_Michigan_45_2015_H1N1_18',
              'A_Michigan_45_2015_H1N1_19',
              'B_Phuket_3073_2013_17',
              'B_Phuket_3073_2013_18']
consensusReferences = [mainSampleFolder + '/consensus/' + reference + '_consensus_noambig.fasta' for mainSampleFolder, reference in zip(mainSampleFolders, references)]
distancefiles = [f + '/snp_calls/sequenceDistances.tsv' for f in mainSampleFolders]
gtfFiles = ['/mnt/d/orchards/h1n1/' + reference + '_antigenic.gtf' for reference in references]
allnucVCFfiles = [f+'/rerun/all_snps.vcf' for f in mainSampleFolders]
SnpGenieSegFolders = []
for f in mainSampleFolders:
    SnpGenieSegFolders.extend(glob.glob(f+'/SNP_Genie_filtered/*'))

treebase = '/mnt/d/orchards/usacladework/augurWD/'
treefiles = ['FluB/all/FluBYamaDNA_HA_all_refined.tree','/allH3N2/H3N2DNA_HA_refined.tree','/iqtreeTake2/H1N1DNA_HA_all_aligned.fasta.treefile']
clade_references = '/mnt/d/orchards/recombination/clade_references.txt'

hongkongContigs = {'NP':'A_Hong_Kong_4801_2014_834574_NP', 'NS':'A_Hong_Kong_4801_2014_834575_NS', 
                   'MP':'A_Hong_Kong_4801_2014_834576_MP', 'PA':'A_Hong_Kong_4801_2014_834577_PA',
                   'PB2':'A_Hong_Kong_4801_2014_834578_PB2', 'PB1':'A_Hong_Kong_4801_2014_834579_PB1',
                  'NA':'A_Hong_Kong_4801_2014_834580_NA','HA':'A_Hong_Kong_4801_2014_834581_HA'}

#location of all statistics tsvs:
dataFolder = installDir + 'figures'


subtypesToAnalyze = ['H1N1', 'H3N2', 'Influenza B'] #options: 'H1N1', 'H3N2', 'Influenza B', 'Mixed'
naValues = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A','N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan','','*']
read_tsv_args = {'sep':'\t', 'keep_default_na':False, 'na_values':naValues}
read_csv_args = {'keep_default_na':False, 'na_values':naValues}

referenceDict = {"A_Singapore_INFIMH-16-0019_2016":"H3N2","A_Hong_Kong_4801_2014_EPI834581":"H3N2", "A_Michigan_45_2015_H1N1":"H1N1","A_Michigan_45_2015_H1N1_18":"H1N1","A_Michigan_45_2015_H1N1_19":"H1N1", "B_Brisbane_60_2008":"Influenza B", "B_Phuket_3073_2013_17":"Influenza B","B_Phuket_3073_2013_18":"Influenza B","Influenza A H3N2, Influenza B (Yamagata)":"Mixed"}
referenceSeasonDict = {"A_Singapore_INFIMH-16-0019_2016":"2018-2019 H3N2","A_Hong_Kong_4801_2014_EPI834581":"2017-2018 H3N2","A_Michigan_45_2015_H1N1_18":"2017-2018 H1N1","A_Michigan_45_2015_H1N1_19":"2018-2019 H1N1", "B_Phuket_3073_2013_17":"2017-2018 Influenza B","B_Phuket_3073_2013_18":"2018-2019 Influenza B","B_Phuket_3073_2013_16":"2016-2017 Influenza B",}

sampleFolderDict = {'H3N2':{'17-18':installDir+'/ORCHARDS_run19H3N2/A_Hong_Kong_4801_2014_EPI834581/', '18-19':installDir+'/ORCHARDS_run19H3N2/A_Singapore_INFIMH-16-0019_2016/'},'H1N1':{'17-18':installDir+'Orchards_run19/A_Michigan_45_2015_H1N1_18/','18-19':installDir+'Orchards_run19/A_Michigan_45_2015_H1N1_19/'},'H1N1':{'17-18':installDir+'Orchards_run19/A_Michigan_45_2015_H1N1_18/','18-19':installDir+'Orchards_run19/A_Michigan_45_2015_H1N1_19/'},'Influenza B':{'16-17':installDir+'Orchards_runB/B_Phuket_3073_2013_17/','17-18':installDir+'Orchards_runB/B_Phuket_3073_2013_18/'}}

refFileDict={'H3N2':{'17-18':installDir+'A_Hong_Kong_4801_2014_EPI834581.fasta', '18-19':installDir+'A_Singapore_INFIMH-16-0019_2016.fasta'},'H1N1':{'17-18':installDir+'A_Michigan_45_2015_H1N1_18.fasta','18-19':installDir+'A_Michigan_45_2015_H1N1_19.fasta'},'Influenza B':{'16-17':installDir+'B_Phuket_3073_2013_17.fasta','17-18':installDir+'B_Phuket_3073_2013_18.fasta'}}
gtfFileDict={'H3N2':{'17-18':installDir+'A_Hong_Kong_4801_2014_EPI834581_antigenic.gtf', '18-19':installDir+'A_Singapore_INFIMH-16-0019_2016_antigenic.gtf'},'H1N1':{'17-18':installDir+'A_Michigan_45_2015_H1N1_18_antigenic.gtf','18-19':installDir+'A_Michigan_45_2015_H1N1_19_antigenic.gtf'},'H1N1':{'17-18':installDir+'A_Michigan_45_2015_H1N1_18_antigenic.gtf','18-19':installDir+'A_Michigan_45_2015_H1N1_19_antigenic.gtf'},'Influenza B':{'16-17':installDir+'B_Phuket_3073_2013_17_antigenic.gtf','17-18':installDir+'B_Phuket_3073_2013_18_antigenic.gtf'}}
subtypeDict = {'Influenza A H3N2':'H3N2','Flu A (H3)':'H3N2', 
	'Flu A (Unable to Subtype)':'H3N2', 'Flu B (Yamagata)':'Influenza B', 'Flu A 09H1':'H1N1', 
	'Influenza A H1N1':'H1N1', 'Influenza A, Influenza B': 'Mixed', 'Influenza B':'Influenza B',
	'Influenza A H3N2, Influenza B (Yamagata)':'Mixed', 'Influenza A H3N2, Influenza A H1N1':'Mixed',
	'Influenza A, Influenza B (Yamagata)':'Mixed', 'Influenza B (Yamagata)':'Influenza B', 'Influenza A':'H3N2',
	'Influenza B (Victoria)':'Influenza B', 'H3N2':'H3N2','H1N1':'H1N1','Influenza B':'Influenza B'}

snpGenieDict = {'H3N2':{'18-19':installDir+'ORCHARDS_run19H3N2/A_Singapore_INFIMH-16-0019_2016/SNP_Genie_filtered/A_Singapore_INFIMH-16-0019_2016_',
                        '17-18':installDir+'ORCHARDS_run19H3N2/A_Hong_Kong_4801_2014_EPI834581/SNP_Genie_filtered/A_Hong_Kong_4801_2014_834574_'},
                'H1N1':{'18-19':installDir+'ORCHARDS_run19/A_Michigan_45_2015_H1N1_19/SNP_Genie_filtered/A_Michigan_45_2015_',
                          '17-18':installDir+'ORCHARDS_run19/A_Michigan_45_2015_H1N1_19/SNP_Genie_filtered/A_Michigan_45_2015_'},
               'Influenza B':{'17-18':installDir+'ORCHARDS_runB/B_Phuket_3073_2013_18/SNP_Genie_filtered/B_Phuket_3073_2013_',
                              '16-17':installDir+'ORCHARDS_runB/B_Phuket_3073_2013_18/SNP_Genie_filtered/B_Phuket_3073_2013_'}}

H1N1_antigenic_sites = [87,88,90,91,92, 132,
          141,142,143,171,172,174,177,180,
          170,173,202,206,210,211,212,
          151,154,156,157,158,159,200,238,
          147]
H1N1_antigenic_sites = [site-1 for site in H1N1_antigenic_sites] #convert to zero-index

antigenic_sites = {59, 60, 61, 62, 63, 65, 66, 68, 69, 72, 74, 77, 78, 82, 90, 93, 95, 96, 97, 98, 101, 102, 103, 106, 107, 109, 111, 117, 118, 124, 132, 136, 137, 139, 141, 143, 144, 145, 146, 147, 148, 150, 152, 153, 155, 157, 158, 159, 160, 161, 165, 167, 170, 171, 172, 173, 174, 178, 180, 182, 183, 185, 186, 187, 188, 189, 190, 191, 192, 194, 197, 201, 202, 203, 204, 205, 207, 208, 209, 211, 212, 213, 216, 218, 222, 223, 224, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 253, 255, 257, 259, 261, 262, 263, 275, 276, 277, 280, 288, 290, 291, 293, 294, 295, 309, 312, 314, 315, 319, 320, 322, 323, 324, 325, 326, 327}
def convertListofClassicH3N2SitestoZeroIndexedMStart(listOfSites):
    return [site+15 for site in listOfSites]
glycosylation_sites = set(convertListofClassicH3N2SitestoZeroIndexedMStart([8,22,38,45,63,81,133,126,159,160,165,246,285]))

antigenic_sites = antigenic_sites.union(glycosylation_sites)

displayContext = 'poster'

palettes = dict()
snsblue, snsorange, snsgreen, snsred, snspurple, snsbrown, snspink, snsgrey, snsyellow, snssky = sns.color_palette('muted')
palettes['kind'] = sns.color_palette(('#eedc5b','#d3494e'), 2)
palettes['subtype'] = sns.color_palette('deep')
palettes['AAtype'] = sns.color_palette((snsblue, snsorange, snsgreen),3)
palettes['synon'] = sns.color_palette((snsblue, snsorange),2)
palettes['vax'] = sns.color_palette('Reds',2)[::-1]
palettes['age_category'] = sns.color_palette('Paired')
palettes['age_category_only'] = sns.color_palette('tab20')[8:10]

genelengths={'H3N2': {'NEP': 366,
  'HA': 1701,
  'HA_antigenic':len(antigenic_sites)*3,
  'HA_nonantigenic': 1701-len(antigenic_sites)*3,
  'M1': 759,
  'M2': 294,
  'NA': 1410,
  'NP': 1497,
  'NS1': 693,
  'PA': 2151,
  'PA-X': 759,
  'PB1': 2274,
  'PB1-F2': 273,
  'PB2': 2280},
 'H1N1': {'HA_antigenic':len(H1N1_antigenic_sites)*3,
  'HA_nonantigenic':1701-len(H1N1_antigenic_sites)*3,
  'HA': 1701,
  'M1': 759,
  'M2': 294,
  'NA': 1410,
  'NP': 1497,
  'NEP': 366,
  'NS1': 660,
  'PA': 2151,
  'PA-X': 699,
  'PB1': 2274,
  'PB1-F2': 273,
  'PB2': 2280},
 'Influenza B': {'HA': 1755,
  'M1': 747,
  'NA': 1401,
  'NP': 1683,
  'NEP': 369,
  'NS1': 846,
  'PA': 2181,
  'PB1': 2259,
  'PB2': 2313,
  'BM2': 330,
  'NB': 303}}

geneOrder = ["PB2", "PB1", 'PB1-F2', "PA", 'PA-X', "HA", "NP", "NA",  "M1", "M2", "NS1", "NEP"]
antigenicGeneOrder = ["PB2", "PB1", 'PB1-F2', "PA", 'PA-X', "HA", 'HA_antigenic','HA_nonantigenic',"NP", "NA",  "M1", "M2", "NS1", "NEP"]

segOrder = ['PB2','PB1','NP','HA','NA','PA','MP','NS']
subtypeOrder = ['H3N2','H1N1','Influenza B']
vaxOrder = [0,1]
named_vaxOrder = ['Unvaccinated','Vaccinated']
ageOrder = ['18 or Under','Over 18']
NS_order = ['Nonsynon','Synon']

antigenicGeneNames = ["PB2", "PB1", "PA", "HA", 'Anti.\nHA','Nonanti.\nHA',"NP", "NA",  "M1", "M2", "NS1", "NEP"]
antigenicGeneNames_withMinor = ["PB2", "PB1", 'PB1-F2', "PA", 'PA-X', "HA", 'Anti.\nHA','Nonanti.\nHA',"NP", "NA",  "M1", "M2", "NS1", "NEP"]
errorBarArgs = {"capsize":.1, "errwidth":2}

#Define palette options for all charts
palettes = dict()
palettes['kind'] = sns.color_palette(('#eedc5b','#d3494e'), 2)
palettes['subtype'] = sns.color_palette('deep')
palettes['AAtype'] = sns.color_palette((snsblue, snsorange, snsgreen),3)
palettes['synon'] = sns.color_palette((snsblue, snsorange),2)
palettes['vax'] = sns.color_palette('Reds',2)[::-1]
palettes['age_category'] = sns.color_palette('Paired')
palettes['age_category_only'] = sns.color_palette('tab20')[8:10]

#Load data
print ('loading subjects...')
subjects = pd.read_csv(dataFolder+'/subjects.tsv', **read_tsv_args)
print ('loading samples...')
samples = pd.read_csv(dataFolder+'/samples.tsv', **read_tsv_args)

#For downstream analysis, it can be nice to have a few figure-specific variables
samples['age_category'] = '18 or Under'
samples.loc[samples.age>18, 'age_category'] = 'Over 18'

meltedPiSamples = samples.melt(id_vars=['sampleID','subtype', 'recieved_flu_vaccine','age_category','symptom_severity'], value_vars=['piN_sample','piS_sample']).rename(columns={'variable':'PiN_PiS','value':'Pi'})

print ('loading segments...')
segments = pd.read_csv(dataFolder+'/segments.tsv', **read_tsv_args)

#I'll go ahead and make a melted version of all dataframes with piN/piS measurements
meltedPiSegments = segments.melt(id_vars=['sampleID','subtype', 'segment','recieved_flu_vaccine','symptom_severity'], value_vars=['piN_segment','piS_segment']).rename(columns={'variable':'PiN_PiS','value':'Pi'})

print ('loading genes...')
genes = pd.read_csv(dataFolder+'/genes.tsv', **read_tsv_args)
try:
    meltedPiGenes = genes.melt(id_vars=['sampleID','subtype', 'segment','product','age_category','recieved_flu_vaccine','symptom_severity'], value_vars=['piN_gene','piS_gene']).rename(columns={'variable':'PiN_PiS','value':'Pi'})
except:
    print (genes.columns)
    raise

print ('loading SNPs...')
allSNPs = pd.read_csv(dataFolder+'/SNPs_lenient_filter.gz', **read_tsv_args)
SNPs = allSNPs#.loc[allSNPs.transformed_frequency>SNP_frequency_cutoff]

print ('loading transmission pairs...')
transmissionPairs = pd.read_csv(dataFolder+'/transmissionPairs.tsv', **read_tsv_args)
print ('loading transmission segments...')
transmissionSegments = pd.read_csv(dataFolder+'/transmissionSegments.tsv', **read_tsv_args)
print ('loading transmission SNPs...')
transmissionSNPs = pd.read_csv(dataFolder+'/transmissionSNPs_lenient_filter.gz', **read_tsv_args)

# HA_add_on = transmissionSNPs.loc[transmissionSNPs.antigenic_product.isin(['HA_antigenic','HA_nonantigenic'])]
# HA_add_on['antigenic_product'] = 'HA'
# transmissionSNPs = transmissionSNPs.append(HA_add_on)

# plotSNPs = transmissionSNPs.loc[transmissionSNPs.kind=='transmission']
# transmissionSNPs = transmissionSNPs.loc[~((transmissionSNPs.depth_contact < 100)|(transmissionSNPs.depth_index < 100))]

#Adjust SNP frequencies so that I'm always looking at the change that happens to the *minor* allele
# transmissionSNPs['minorAlleleFreq_index']= transmissionSNPs.SNP_frequency_index
# transmissionSNPs['minorAlleleFreq_contact']= transmissionSNPs.SNP_frequency_contact
# transmissionSNPs['minor_alt_nuc']= transmissionSNPs.alt_nuc
# transmissionSNPs['minor_ref_nuc']= transmissionSNPs.ref_nuc
# # print (transmissionSNPs.SNP_frequency_index.max())
# tmpSNPs = transmissionSNPs.copy()

# majorityMinoritySNPs=tmpSNPs.SNP_frequency_index > 0.5
# alt_nucs = tmpSNPs.loc[majorityMinoritySNPs,'alt_nuc']
# tmpSNPs.loc[majorityMinoritySNPs,'minor_alt_nuc'] = tmpSNPs.loc[majorityMinoritySNPs,'ref_nuc']
# tmpSNPs.loc[majorityMinoritySNPs,'minor_ref_nuc'] = alt_nucs
# print (transmissionSNPs.SNP_frequency_index.max())
# tmpSNPs.loc[majorityMinoritySNPs, 'minorAlleleFreq_index'] = np.abs(1-tmpSNPs.loc[majorityMinoritySNPs, 'SNP_frequency_index'].values)
# tmpSNPs.loc[majorityMinoritySNPs, 'minorAlleleFreq_contact'] = np.abs(1-tmpSNPs.loc[majorityMinoritySNPs, 'SNP_frequency_contact'].values)
# print (transmissionSNPs.SNP_frequency_index.max())
# tmpSNPs['SNP_frequency_directional_change'] = tmpSNPs.SNP_frequency_contact - tmpSNPs.SNP_frequency_index
# tmpSNPs['abs_SNP_frequency_difference'] = np.abs(tmpSNPs.SNP_frequency_directional_change)
# # tmpSNPs[majorityMinoritySNPs].columns, 'SNP_frequency_index'
# adjustedTransmissionSNPs = tmpSNPs
# print (transmissionSNPs.SNP_frequency_index.max())
# #make all vs all distance DF for distance comparisons

distanceDF = pd.DataFrame()
for distancefile in distancefiles:
    tmp = pd.read_csv(distancefile, **read_tsv_args).set_index('Unnamed: 0')
    distanceDF = distanceDF.append(tmp, sort=True)

distanceDF = distanceDF.sort_index()

mask = np.triu(np.ones(distanceDF.shape)).astype(np.bool)
distanceDF = distanceDF.mask(mask,np.nan)
# allvsall = distanceDF.stack()
# allvsall = allvsall.reset_index().rename(columns={'Unnamed: 0':'index','level_1':'contact',0:'distance'}).dropna()
allvsall = pd.read_csv('/mnt/d/orchards/H1N1/figures/allvsall.tsv', **read_tsv_args)
allvsall = allvsall.merge(samples, left_on='index',right_on='sampleID',how='left')
allvsall = allvsall.merge(samples, left_on='contact',right_on='sampleID',how='left', suffixes=('_index','_contact'))

#limit comparisons to those where contact infected after index, and onset of symptoms are separated by less than one week
allvsall = allvsall.loc[(pd.to_datetime(allvsall['time_of_symptom_onset_contact']) - pd.to_datetime(allvsall['time_of_symptom_onset_index'])) >= pd.Timedelta(0)]
allvsall = allvsall.loc[pd.to_datetime(allvsall['time_of_symptom_onset_contact']) - pd.to_datetime(allvsall['time_of_symptom_onset_index']) <= pd.Timedelta('10 days')]
allvsall = allvsall.loc[allvsall.subtype_index == allvsall.subtype_contact]

allvsall['school_match'] = 'Does not attend'
allvsall.loc[allvsall.school_index == allvsall.school_contact, 'school_match'] = 'Within school'
allvsall.loc[allvsall.school_index != allvsall.school_contact, 'school_match'] = 'Between schools'

allvsall['household_match'] = 'Other'
allvsall.loc[allvsall.household_index != allvsall.household_contact, 'household_match'] = 'No'
allvsall.loc[allvsall.household_index == allvsall.household_contact, 'household_match'] = 'Yes'

allvsall = allvsall.reset_index(drop=True) #tidy up the index
allvsall['Relatedness'] = 'Random'
allvsall.loc[allvsall.clade_index == allvsall.clade_contact, 'Relatedness'] = 'Same Clade'
allvsall.loc[allvsall.subclade_index == allvsall.subclade_contact, 'Relatedness'] = 'Same Subclade'
allvsall.loc[allvsall.household_index == allvsall.household_contact, 'Relatedness'] = 'Same Household'
allvsall.loc[allvsall.school_index == allvsall.school_contact, 'Relatedness'] = 'Same School'

id_columns = ['sampleID', 'subtype','season','age','age_category','recieved_flu_vaccine','clade','subclade']
sample_N_stats = ['nonsynon_snps_per_day_samp','Xue_nonsynon_divergence','num_of_nonsynon_muts','nonsynon_mutation_rate_samp','Xue_nonsynon_divergence_per_day','nonsynon_divergence_rate']
sample_S_stats = ['synon_snps_per_day_samp','Xue_synon_divergence','num_of_synon_muts','synon_mutation_rate_samp','Xue_synon_divergence_per_day','synon_divergence_rate']

segment_N_stats = ['nonsynon_snps_per_day_seg', 'Xue_nonsynon_divergence_segment', 'num_of_nonsynon_muts_segment', 'nonsynon_mutation_rate_seg', 'nonsynon_divergence_per_day_seg', 'nonsynon_divergence_rate_seg']
segment_S_stats = [col.replace('nonsynon_','synon_') for col in segment_N_stats]

gene_N_stats = [col.replace('_segment','').replace('_seg','')+'_gene' for col in segment_N_stats]
gene_S_stats = [col.replace('_segment','').replace('_seg','')+'_gene' for col in segment_S_stats]

sample_N_stats.append('piN_sample')
sample_S_stats.append('piS_sample')
segment_N_stats.append('piN_segment')
segment_S_stats.append('piS_segment')
gene_N_stats.append('piN_gene')
gene_S_stats.append('piS_gene')

N_sample_renameDict = {col: col.replace('nonsynon_', '').replace('piN','pi') for col in sample_N_stats}
S_sample_renameDict = {col: col.replace('synon_', '').replace('piS','pi') for col in sample_S_stats}

N_segment_renameDict = {col: col.replace('nonsynon_', '').replace('piN','pi') for col in segment_N_stats}
S_segment_renameDict = {col: col.replace('synon_', '').replace('piS','pi') for col in segment_S_stats}

N_gene_renameDict = {col: col.replace('nonsynon_', '').replace('piN','pi') for col in gene_N_stats}
S_gene_renameDict = {col: col.replace('synon_', '').replace('piS','pi') for col in gene_S_stats}

N_samples = samples[id_columns+sample_N_stats].rename(columns=N_sample_renameDict)
S_samples = samples[id_columns+sample_S_stats].rename(columns=S_sample_renameDict)

N_segments = segments[['segment'] + id_columns+segment_N_stats].rename(columns=N_segment_renameDict)
S_segments = segments[['segment'] +id_columns+segment_S_stats].rename(columns=S_segment_renameDict)

N_genes = genes[['segment', 'product'] +id_columns+gene_N_stats].rename(columns=N_gene_renameDict)
S_genes = genes[['segment', 'product'] +id_columns+gene_S_stats].rename(columns=S_gene_renameDict)

N_samples['Synon_Nonsynon'] = N_segments['Synon_Nonsynon'] = N_genes['Synon_Nonsynon'] = 'Nonsynon'
S_samples['Synon_Nonsynon'] = S_segments['Synon_Nonsynon'] = S_genes['Synon_Nonsynon'] = 'Synon'

NS_samples = N_samples.append(S_samples)
NS_segments = N_segments.append(S_segments)
NS_genes = N_genes.append(S_genes)


samples['recieved_flu_vaccine'] = samples['recieved_flu_vaccine'].map({0:'Unvaccinated', 1:'Vaccinated', np.nan:np.nan})
NS_samples['recieved_flu_vaccine'] = NS_samples['recieved_flu_vaccine'].map({0:'Unvaccinated', 1:'Vaccinated', np.nan:np.nan})
genes['recieved_flu_vaccine'] = genes['recieved_flu_vaccine'].map({0:'Unvaccinated', 1:'Vaccinated', np.nan:np.nan})