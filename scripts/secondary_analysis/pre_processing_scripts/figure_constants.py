import os
import numpy as np
import pandas as pd
import seaborn as sns
import glob


SNP_frequency_cutoff = 0.01
min_coverage = 100
calcBottlenecks = False

#Set Constants
potentialmixed = ['18VR001531', '19VR004455', '19VR003675', '19VR003920', '19VR003675']
#Samples which, based off visual inspection of mutations, seem to be mixed infections.
#This is determined by looking at 10 samples with the most mutations and seeing if those
#mutations, when close together, tend to be linked on the same read.


installDir = '/'.join(os.getcwd().split('/')[:-3])

metadataDir = installDir + '/data/sample_metadata/'
metadatafile = metadataDir + 'sample_metadata.csv'
completemetadatakey = metadataDir + 'subject_metadata_key.csv'
expandedMetadata = metadataDir + 'subject_metadata.csv'

figures = os.path.join(installDir, 'results', 'figures')
bottleneck_output = os.path.join(installDir, 'results', 'bottleneck_output')
secondaryDataFolders = [installDir + '/data/secondary_analysis/H3N2/18-19',
                        installDir + '/data/secondary_analysis/H3N2/17-18',
                        installDir + '/data/secondary_analysis/H1N1/18-19',
                        installDir + '/data/secondary_analysis/H1N1/17-18',
                        installDir + '/data/secondary_analysis/FluB/16-17',
                        installDir + '/data/secondary_analysis/FluB/17-18']

referenceDir = installDir + '/references'

vcfdirs = secondaryDataFolders
vcffiles = [f + '/all_snps_filtered.vcf' for f in vcfdirs]

references = ['A_Singapore_INFIMH-16-0019_2016',
              'A_Hong_Kong_4801_2014_EPI834581',
              'A_Michigan_45_2015_H1N1_18',
              'A_Michigan_45_2015_H1N1_19',
              'B_Phuket_3073_2013_17',
              'B_Phuket_3073_2013_18']

consensusReferences = [mainSampleFolder + '/consensus/' + reference + '_consensus_noambig.fasta' for mainSampleFolder, reference in zip(secondaryDataFolders, references)]
gtfFiles = [referenceDir + '/' + reference + '_antigenic.gtf' for reference in references]

SnpGenieSegFolders = []
for f in secondaryDataFolders:
    SnpGenieSegFolders.extend(glob.glob(f + '/SNPGenie_output/*'))


treefiles = [installDir + '/data/secondary_analysis/FluB/FluB.tree', 
             installDir + '/data/secondary_analysis/H3N2/H3N2.tree', 
             installDir + '/data/secondary_analysis/H1N1/H1N1.tree']
clade_references = installDir + '/data/references/subclade_definitions/Clade_reference_sequence_names.txt'

hongkongContigs = {'NP': 'A_Hong_Kong_4801_2014_834574_NP', 'NS': 'A_Hong_Kong_4801_2014_834575_NS',
                   'MP': 'A_Hong_Kong_4801_2014_834576_MP', 'PA': 'A_Hong_Kong_4801_2014_834577_PA',
                   'PB2': 'A_Hong_Kong_4801_2014_834578_PB2', 'PB1': 'A_Hong_Kong_4801_2014_834579_PB1',
                   'NA': 'A_Hong_Kong_4801_2014_834580_NA', 'HA': 'A_Hong_Kong_4801_2014_834581_HA'}

# location of all statistics tsvs:
dataFolder = installDir + '/results/dataframes'

subtypesToAnalyze = ['H1N1', 'H3N2', 'Influenza B']

# exclude 'NA' as a reserved term for nan when importing pandas dataframes
naValues = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A', 'N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', '', '*']
read_tsv_args = {'sep': '\t', 'keep_default_na': False, 'na_values': naValues}
read_csv_args = {'keep_default_na': False, 'na_values': naValues}
gene_to_seg_dict = {'HA': 'HA', 'NA': 'NA', 'PB1': 'PB1', 'PB2': 'PB2', 'PA': 'PA', 'NP': 'NP',
                    'NEP': 'NS', 'NS1': 'NS', 'M1': 'MP', 'M2': 'MP', 'PB1-F2': 'PB1', 'PA-X': 'PA',
                    'NB': 'NA', 'BM2': 'MP'}

referenceDict = {"A_Singapore_INFIMH-16-0019_2016": "H3N2",
                 "A_Hong_Kong_4801_2014_EPI834581": "H3N2",
                 "A_Michigan_45_2015_H1N1": "H1N1",
                 "A_Michigan_45_2015_H1N1_18": "H1N1",
                 "A_Michigan_45_2015_H1N1_19": "H1N1",
                 "B_Brisbane_60_2008": "Influenza B",
                 "B_Phuket_3073_2013_17": "Influenza B",
                 "B_Phuket_3073_2013_18": "Influenza B",
                 "Influenza A H3N2, Influenza B (Yamagata)": "Mixed"}

referenceSeasonDict = {"A_Singapore_INFIMH-16-0019_2016": "2018-2019 H3N2",
                       "A_Hong_Kong_4801_2014_EPI834581": "2017-2018 H3N2",
                       "A_Michigan_45_2015_H1N1_18": "2017-2018 H1N1",
                       "A_Michigan_45_2015_H1N1_19": "2018-2019 H1N1",
                       "B_Phuket_3073_2013_17": "2017-2018 Influenza B",
                       "B_Phuket_3073_2013_18": "2018-2019 Influenza B",
                       "B_Phuket_3073_2013_16": "2016-2017 Influenza B"}

sampleFolderDict = {'H3N2': {'17-18': installDir + '/data/secondary_analysis/H3N2/17-18',
                             '18-19': installDir + '/data/secondary_analysis/H3N2/18-19'},
                    'H1N1': {'17-18': installDir + '/data/secondary_analysis/H1N1/17-18',
                             '18-19': installDir + '/data/secondary_analysis/H1N1/18-19'},
                    'H1N1pdm': {'17-18': installDir + '/data/secondary_analysis/H1N1/17-18',
                                '18-19': installDir + '/data/secondary_analysis/H1N1/18-19'},
                    'Influenza B': {'16-17': installDir + '/data/secondary_analysis/FluB/16-17',
                                    '17-18': installDir + '/data/secondary_analysis/FluB/17-18'}}

# Dictionary to convert myriad different subtypes in metadata file into consistent set of four subtypes
subtypeDict = {'Influenza A H3N2': 'H3N2', 'Flu A (H3)': 'H3N2',
               'Flu A (Unable to Subtype)': 'H3N2', 'Flu B (Yamagata)': 'Influenza B', 'Flu A 09H1': 'H1N1',
               'Influenza A H1N1': 'H1N1', 'Influenza A, Influenza B': 'Mixed', 'Influenza B': 'Influenza B',
               'Influenza A H3N2, Influenza B (Yamagata)': 'Mixed', 'Influenza A H3N2, Influenza A H1N1': 'Mixed',
               'Influenza A, Influenza B (Yamagata)': 'Mixed', 'Influenza B (Yamagata)': 'Influenza B', 'Influenza A': 'H3N2',
               'Influenza B (Victoria)': 'Influenza B', 'H3N2': 'H3N2', 'H1N1': 'H1N1', 'Influenza B': 'Influenza B'}

H1N1_antigenic_sites = [87, 88, 90, 91, 92, 132, 141, 142,
                        143, 147, 171, 172, 174, 177, 180,
                        170, 173, 202, 206, 210, 211, 212,
                        151, 154, 156, 157, 158, 159, 200, 238]

H1N1_antigenic_sites = [site - 1 for site in H1N1_antigenic_sites]  # convert to zero-index

antigenic_sites = {59, 60, 61, 62, 63, 65, 66, 68, 69, 72, 74, 77, 78, 82, 90, 93, 95, 96, 97, 98, 101, 102, 103, 106, 107, 109, 111, 117, 118, 124, 132, 136, 137, 139, 141, 143, 144, 145, 146, 147, 148, 150, 152, 153, 155, 157, 158, 159, 160, 161, 165, 167, 170, 171, 172, 173, 174, 178, 180, 182, 183, 185, 186, 187, 188, 189, 190, 191, 192, 194, 197, 201, 202, 203, 204, 205, 207, 208, 209, 211, 212, 213, 216, 218, 222, 223, 224, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 253, 255, 257, 259, 261, 262, 263, 275, 276, 277, 280, 288, 290, 291, 293, 294, 295, 309, 312, 314, 315, 319, 320, 322, 323, 324, 325, 326, 327}

def convertListofClassicH3N2SitestoZeroIndexedMStart(listOfSites):
    return [site + 15 for site in listOfSites]

glycosylation_sites = [8, 22, 38, 45, 63, 81, 133, 126, 159, 160, 165, 246, 285]
glycosylation_sites = set(convertListofClassicH3N2SitestoZeroIndexedMStart(glycosylation_sites))

antigenic_sites = antigenic_sites.union(glycosylation_sites)

genelengths = {'H3N2': {'NEP': 366,
  'HA': 1701,
  'HA_antigenic': len(antigenic_sites) * 3,
  'HA_nonantigenic': 1701 - len(antigenic_sites) * 3,
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
 'H1N1': {'HA_antigenic': len(H1N1_antigenic_sites) * 3,
  'HA_nonantigenic': 1701 - len(H1N1_antigenic_sites) * 3,
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

# Display constants
displayContext = 'poster'

palettes = dict()
snsblue, snsorange, snsgreen, snsred, snspurple, snsbrown, snspink, snsgrey, snsyellow, snssky = sns.color_palette('muted')
palettes['kind'] = sns.color_palette(('#eedc5b', '#d3494e'), 2)
palettes['subtype'] = sns.color_palette('deep')
palettes['AAtype'] = sns.color_palette((snsblue, snsorange, snsgreen), 3)
palettes['synon'] = sns.color_palette((snsblue, snsorange), 2)
palettes['vax'] = sns.color_palette('Reds', 2)[::-1]
palettes['age_category'] = sns.color_palette('Paired')
palettes['age_category_only'] = sns.color_palette('tab20')[8:10]

geneOrder = ["PB2", "PB1", 'PB1-F2', "PA", 'PA-X', "HA", "NP", "NA", "M1", "M2", "NS1", "NEP"]
antigenicGeneOrder = ["PB2", "PB1", 'PB1-F2', "PA", 'PA-X', "HA", 'HA_antigenic', 'HA_nonantigenic',"NP", "NA", "M1", "M2", "NS1", "NEP"]

segOrder = ['PB2', 'PB1', 'NP', 'HA', 'NA', 'PA', 'MP', 'NS']
subtypeOrder = ['H3N2', 'H1N1', 'Influenza B']
vaxOrder = [0, 1]
named_vaxOrder = ['Unvaccinated', 'Vaccinated']
ageOrder = ['18 or Under', 'Over 18']
NS_order = ['Nonsynon', 'Synon']

antigenicGeneNames = ["PB2", "PB1", "PA", "HA", 'Anti.\nHA', 'Nonanti.\nHA', "NP", "NA", "M1", "M2", "NS1", "NEP"]
antigenicGeneNames_withMinor = ["PB2", "PB1", 'PB1-F2', "PA", 'PA-X', "HA", 'Anti.\nHA', 'Nonanti.\nHA', "NP", "NA", "M1", "M2", "NS1", "NEP"]
errorBarArgs = {"capsize": .1, "errwidth": 2}

# Load data
print ('loading subjects...')
subjects = pd.read_csv(dataFolder + '/subjects.tsv', **read_tsv_args)
print ('loading samples...')
samples = pd.read_csv(dataFolder + '/samples.tsv', **read_tsv_args)

# For downstream analysis, it can be nice to have a few figure-specific variables
samples['age_category'] = '18 or Under'
samples.loc[samples.age > 18, 'age_category'] = 'Over 18'

meltedPiSamples = samples.melt(id_vars=['sampleID', 'subtype', 'recieved_flu_vaccine', 'age_category', 'symptom_severity'], value_vars=['piN_sample', 'piS_sample']).rename(columns={'variable': 'PiN_PiS', 'value': 'Pi'})

print ('loading segments...')
segments = pd.read_csv(dataFolder + '/segments.tsv', **read_tsv_args)

# I'll go ahead and make a melted version of all dataframes with piN/piS measurements
meltedPiSegments = segments.melt(id_vars=['sampleID', 'subtype', 'segment', 'recieved_flu_vaccine', 'symptom_severity'], value_vars=['piN_segment', 'piS_segment']).rename(columns={'variable': 'PiN_PiS', 'value': 'Pi'})

print ('loading genes...')
genes = pd.read_csv(dataFolder + '/genes.tsv', **read_tsv_args)
try:
    meltedPiGenes = genes.melt(id_vars=['sampleID', 'subtype', 'segment', 'product', 'age_category', 'recieved_flu_vaccine', 'symptom_severity'], value_vars=['piN_gene', 'piS_gene']).rename(columns={'variable': 'PiN_PiS', 'value': 'Pi'})
except:
    print (genes.columns)
    raise

print ('loading SNPs...')
SNPs = pd.read_csv(dataFolder + '/SNPs_lenient_filter.gz', **read_tsv_args)
SNPs

print ('loading transmission pairs...')
transmissionPairs = pd.read_csv(dataFolder + '/transmissionPairs.tsv', **read_tsv_args)
print ('loading transmission segments...')
transmissionSegments = pd.read_csv(dataFolder + '/transmissionSegments.tsv', **read_tsv_args)
print ('loading transmission SNPs...')
transmissionSNPs = pd.read_csv(dataFolder + '/transmissionSNPs_lenient_filter.gz', **read_tsv_args)

# make all vs all distance DF for distance comparisons
allvsall = pd.read_csv('/mnt/d/orchards/H1N1/figures/allvsall.tsv', **read_tsv_args)
allvsall = allvsall.merge(samples, left_on='index', right_on='sampleID', how='left')
allvsall = allvsall.merge(samples, left_on='contact', right_on='sampleID', how='left', suffixes=('_index', '_contact'))

# limit comparisons to those where contact infected after index, and onset of symptoms are separated by less than one week
allvsall = allvsall.loc[(pd.to_datetime(allvsall['time_of_symptom_onset_contact']) - pd.to_datetime(allvsall['time_of_symptom_onset_index'])) >= pd.Timedelta(0)]
allvsall = allvsall.loc[pd.to_datetime(allvsall['time_of_symptom_onset_contact']) - pd.to_datetime(allvsall['time_of_symptom_onset_index']) <= pd.Timedelta('10 days')]
allvsall = allvsall.loc[allvsall.subtype_index == allvsall.subtype_contact]

allvsall['school_match'] = 'Does not attend'
allvsall.loc[allvsall.school_index == allvsall.school_contact, 'school_match'] = 'Within school'
allvsall.loc[allvsall.school_index != allvsall.school_contact, 'school_match'] = 'Between schools'

allvsall['household_match'] = 'Other'
allvsall.loc[allvsall.household_index != allvsall.household_contact, 'household_match'] = 'No'
allvsall.loc[allvsall.household_index == allvsall.household_contact, 'household_match'] = 'Yes'

allvsall = allvsall.reset_index(drop=True)
allvsall['Relatedness'] = 'Random'
allvsall.loc[allvsall.clade_index == allvsall.clade_contact, 'Relatedness'] = 'Same Clade'
allvsall.loc[allvsall.subclade_index == allvsall.subclade_contact, 'Relatedness'] = 'Same Subclade'
allvsall.loc[allvsall.household_index == allvsall.household_contact, 'Relatedness'] = 'Same Household'
allvsall.loc[allvsall.school_index == allvsall.school_contact, 'Relatedness'] = 'Same School'

id_columns = ['sampleID', 'subtype', 'season', 'age', 'age_category', 'recieved_flu_vaccine', 'clade', 'subclade']
sample_N_stats = ['nonsynon_snps_per_day_samp', 'Xue_nonsynon_divergence', 'num_of_nonsynon_muts', 'nonsynon_mutation_rate_samp', 'Xue_nonsynon_divergence_per_day', 'nonsynon_divergence_rate']
sample_S_stats = ['synon_snps_per_day_samp', 'Xue_synon_divergence', 'num_of_synon_muts', 'synon_mutation_rate_samp', 'Xue_synon_divergence_per_day', 'synon_divergence_rate']

segment_N_stats = ['nonsynon_snps_per_day_seg', 'Xue_nonsynon_divergence_segment', 'num_of_nonsynon_muts_segment', 'nonsynon_mutation_rate_seg', 'nonsynon_divergence_per_day_seg', 'nonsynon_divergence_rate_seg']
segment_S_stats = [col.replace('nonsynon_', 'synon_') for col in segment_N_stats]

gene_N_stats = [col.replace('_segment', '').replace('_seg', '')+'_gene' for col in segment_N_stats]
gene_S_stats = [col.replace('_segment', '').replace('_seg', '')+'_gene' for col in segment_S_stats]

sample_N_stats.append('piN_sample')
sample_S_stats.append('piS_sample')
segment_N_stats.append('piN_segment')
segment_S_stats.append('piS_segment')
gene_N_stats.append('piN_gene')
gene_S_stats.append('piS_gene')

N_sample_renameDict = {col: col.replace('nonsynon_', '').replace('piN', 'pi') for col in sample_N_stats}
S_sample_renameDict = {col: col.replace('synon_', '').replace('piS', 'pi') for col in sample_S_stats}

N_segment_renameDict = {col: col.replace('nonsynon_', '').replace('piN', 'pi') for col in segment_N_stats}
S_segment_renameDict = {col: col.replace('synon_', '').replace('piS', 'pi') for col in segment_S_stats}

N_gene_renameDict = {col: col.replace('nonsynon_', '').replace('piN', 'pi') for col in gene_N_stats}
S_gene_renameDict = {col: col.replace('synon_', '').replace('piS', 'pi') for col in gene_S_stats}

N_samples = samples[id_columns + sample_N_stats].rename(columns=N_sample_renameDict)
S_samples = samples[id_columns + sample_S_stats].rename(columns=S_sample_renameDict)

N_segments = segments[['segment'] + id_columns + segment_N_stats].rename(columns=N_segment_renameDict)
S_segments = segments[['segment'] + id_columns + segment_S_stats].rename(columns=S_segment_renameDict)

N_genes = genes[['segment', 'product'] + id_columns + gene_N_stats].rename(columns=N_gene_renameDict)
S_genes = genes[['segment', 'product'] + id_columns + gene_S_stats].rename(columns=S_gene_renameDict)

N_samples['Synon_Nonsynon'] = N_segments['Synon_Nonsynon'] = N_genes['Synon_Nonsynon'] = 'Nonsynon'
S_samples['Synon_Nonsynon'] = S_segments['Synon_Nonsynon'] = S_genes['Synon_Nonsynon'] = 'Synon'

NS_samples = N_samples.append(S_samples)
NS_segments = N_segments.append(S_segments)
NS_genes = N_genes.append(S_genes)


samples['recieved_flu_vaccine'] = samples['recieved_flu_vaccine'].map({0: 'Unvaccinated', 1: 'Vaccinated', np.nan: np.nan})
NS_samples['recieved_flu_vaccine'] = NS_samples['recieved_flu_vaccine'].map({0: 'Unvaccinated', 1: 'Vaccinated', np.nan: np.nan})
genes['recieved_flu_vaccine'] = genes['recieved_flu_vaccine'].map({0: 'Unvaccinated', 1: 'Vaccinated', np.nan: np.nan})