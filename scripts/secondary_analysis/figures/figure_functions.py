import os
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import re
import pysam

from itertools import product
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

import math

import scipy.stats as stats
from scipy.stats import truncnorm

from figure_constants import palettes, genelengths, hongkongContigs, SNPs, SNP_frequency_cutoff, transmissionSNPs, figures

def set_fig_settings(displayContext, rcParams=dict()):
    sns.set()
    sns.set_style("white")
    assert (displayContext in ['paper','talk','poster'])
    sns.set_context(displayContext)
    if displayContext == 'paper':
        sns.set_context('paper', font_scale=1)
    elif displayContext == 'talk':
        sns.set_context("talk")
    elif displayContext == 'poster':
        sns.set_context("poster")

    rcParams['figure.figsize'] = [8.0, 6.0]
    rcParams['figure.titleweight'] = 'bold'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['font.weight'] = 'bold'
    rcParams['font.size'] = 36
    rcParams['font.stretch'] = 'condensed'

    rcParams['axes.labelsize'] = 26
    rcParams['axes.labelweight'] = 'bold'
    rcParams['xtick.labelsize'] = 18
    rcParams['ytick.labelsize'] = 18

    rcParams['figure.dpi'] = 300

    rcParams['font.family'] = 'Roboto'
    return rcParams

def addStatsLines(ax, x, y, data, 
                  method='mean', hue=None, order=None, hue_order=None, dodge=True, 
                  linewidth=2, meanwidth=0.35, err_ratio=0.5, meancolor='black', color='black'):
    hw = meanwidth
    er = err_ratio

    xticks = [text.get_text() for text in ax.get_xticklabels()]
    sns_box_plotter = sns.categorical._BoxPlotter(x, y, hue, data, order, hue_order, orient=None, width=.8, color=None, palette=None, saturation=.75, dodge=dodge, fliersize=5, linewidth=None)

    if hue:
        hueOffsets = {hue: sns_box_plotter.hue_offsets[sns_box_plotter.hue_names.index(hue)] for hue in sns_box_plotter.hue_names}

        xlocs = {group: xticks.index(str(group[0])) + hueOffsets[group[1]] for group in product(sns_box_plotter.group_names, hueOffsets.keys())}

        groups = [x, hue]
        hw = hw / len(sns_box_plotter.hue_names)

    else:
        groups = [x]
        xlocs = {group: xcat for group, xcat in zip(sns_box_plotter.group_names, ax.xaxis.get_ticklocs())}

    for xcat, df in data.groupby(groups):
        xloc = xlocs[xcat]
        if method == 'median':
            middle = df[y].median()
            uppererror = np.percentile(df[y].dropna(), 75) - middle
            lowererror = middle - np.percentile(df[y].dropna(), 25)
            print (f'{middle}, 95th upper-lower {np.percentile(df[y].dropna(), 95)}, {np.percentile(df[y].dropna(), 5)}, 25th upper-lower {np.percentile(df[y].dropna(), 75)}, {np.percentile(df[y].dropna(), 25)}')
        else:
            middle = df[y].mean()
            uppererror = lowererror = df[y].sem() * 1.96
        ax.hlines(middle, xloc - hw, xloc + hw, zorder=10, linewidth=linewidth, color=meancolor)
        ax.hlines((middle - lowererror, middle + uppererror),
                   xloc - (hw * er),
                   xloc + (hw * er),
                   zorder=10, linewidth=linewidth, color=color)
        ax.vlines(xloc, middle - lowererror, middle + uppererror, zorder=10, linewidth=linewidth, color=color)
    return ax

def addRegStats(ax, x, y, data):
    def form(x):
        if x < 0.01:
            return f'{x:.2e}'
        elif x > 10:
            exp = x//10
            exp += 3
            return f'{x:.3e}'
        else:
            return f'{x:.3}'

    m, b, r_value, p_value, std_err = stats.linregress(data[[x, y]].dropna().to_numpy())

    textstr = f'y = {form(m)}x + {form(b)}\n$r^2$ = {form(r_value**2)}\nci = {form(std_err*1.96)}\np = {form(p_value)}'
    ax.text(0.05, .78, textstr, transform=ax.transAxes, ha="left", fontsize=rcParams['font.size'] * .66)

def errorbar(x, y, low, high, order, color, ax):
    ynum = [order.index(y_i) for y_i in y]
    lowerrbar = [x - low for x, low in zip(x, low)]
    uppererrbar = [high - x for x, high in zip(x, high)]
    return ax.errorbar(ynum, x, yerr=(lowerrbar, uppererrbar), fmt="none", color=color, elinewidth=1, capsize=5)

def calc_ci(array, z=1.96):
    s = np.std(array)  # std of vector
    n = len(array)  # number of obs

    return (z * (s / math.sqrt(n)))

def bootstrap(array, num_of_bootstraps, function, *args, **kwargs):
    x_bar = function(array, *args, **kwargs)
    sampled_results = np.zeros(num_of_bootstraps)
    for i in range(num_of_bootstraps):
        sample = np.random.choice(array, len(array), replace=True)
        sampled_results[i] = function(sample, *args, **kwargs)
    deltastar = sampled_results - x_bar
    ci = np.percentile(deltastar, 2.5)
    return ci

def bootstrap_df(df, num_of_bootstraps, function, *args, **kwargs):
    x_bar = function(df, *args, **kwargs)
    sampled_results = np.zeros(num_of_bootstraps)
    for i in range(num_of_bootstraps):
        sample = df.sample(n=len(df), replace=True)
        sampled_results[i] = function(sample, *args, **kwargs)
    deltastar = sampled_results - x_bar
    ci = np.nanpercentile(deltastar, 2.5)
    if ci == np.nan:
        print (sampled_results)
    return ci

def convertListofClassicH3N2SitestoZeroIndexedMStart(listOfSites):
    return [site + 15 for site in listOfSites]

def parseGTF(gtffile, segmentLocations):
    '''given file location of gtf, and dictionary of starting locations
       of each chrom in a concatenated sequence, return dictionary of
       {gene product : numpy filter for concatenated sequence'''

    with open(gtffile, 'r') as g:
        gtf = g.readlines()

    coding_regions = {}
    for line in gtf:
        line = line.replace("/", "_")
        lineitems = line.split("\t")
        segment_name = lineitems[0]
        annotation_type = lineitems[2]
        start = int(lineitems[3]) - 1  # adding the -1 here for 0 indexing
        stop = int(lineitems[4]) - 1 # adding the -1 here for 0 indexing
        gene_name = lineitems[8]
        gene_name = gene_name.split(";")[0]
        gene_name = gene_name.replace("gene_id ", "")
        gene_name = gene_name.replace("\"", "")

        if annotation_type.lower() == "cds":
            if segment_name not in coding_regions:
                coding_regions[segment_name] = {}
                coding_regions[segment_name][gene_name] = [[start, stop]]
            elif segment_name in coding_regions and gene_name not in coding_regions[segment_name]:
                coding_regions[segment_name][gene_name] = [[start, stop]]
            elif gene_name in coding_regions[segment_name]:
                coding_regions[segment_name][gene_name].append([start, stop])

    return coding_regions

def makeManhattanPlot(ax, y, data, nrows=2, subtype=None, geneorder=None, antigenic=True, hue=None, hue_order=['Nonsynonymous', 'Synonymous'], palette_type='synon', color=None, dotsize=40, linewidth=0, alpha=1, negativeDataSet=False, y_label='Minor Allele\nFrequency', no_zero_for_ha=True):
    mother_ax = ax
    del ax
    mother_ax.get_yaxis().set_visible(False)
    mother_ax.get_xaxis().set_visible(False)
    mother_ax.spines['right'].set_visible(False)
    mother_ax.spines['top'].set_visible(False)
    mother_ax.spines['left'].set_visible(False)
    mother_ax.spines['bottom'].set_visible(False)
    if subtype:
        if subtype == 'H1N1pdm':  # Early on while coding this I called all H1N1 samples "H1N1pdm"; this is incorrect.
            subtype = 'H1N1'
            data = data.replace('H1N1pdm', 'H1N1')
        data = data.loc[data.subtype == subtype]
    else:
        subtype = data.subtype.first()
    if geneorder:
        pass
    elif antigenic:
        geneorder = ["PB2", "PB1", "PA", "HA", "HA_antigenic", "HA_nonantigenic", "NP", "NA", "M1", "M2", "NS1", "NEP", 'PB1-F2', 'PA-X']
    else:
        geneorder = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M1", "M2", "NS1", "NEP", 'PB1-F2', 'PA-X']

    args = {'y': y, 'hue': hue, 'color': color, 'hue_order': hue_order, 'palette': palettes[palette_type], 'alpha': alpha}
    if type(dotsize) == int:
        args['s'] = dotsize
    elif type(dotsize) == str:
        args['size'] = dotsize
        args['s'] = 40
    ymax = data[y].max()
    ymin = data[y].min()

    ordered_gene_lengths = [(gene, length) for gene, length in genelengths[subtype].items() if gene in geneorder]

    ordered_gene_lengths.sort(key=lambda x: geneorder.index(x[0]))

    lengths = [x[1] for x in ordered_gene_lengths]

    current_length = 0
    end_of_row = sum([length for gene, length in ordered_gene_lengths])/nrows


    reset_points = [0]
    for i, (gene, length) in enumerate(ordered_gene_lengths):
        if current_length > end_of_row:
            reset_points.append(i)
            current_length = length
        else:
            current_length += length

    ncolumns = reset_points[1]
    minorrowadjust = int(abs(sum(lengths[0:ncolumns]) - sum(lengths[ncolumns:])) / 2)

    # make gene subaxes
    ax_x_positions = list()
    row_x_positions = list()
    current_x_pos = 0
    max_x_pos = 0
    for i, (gene, length) in enumerate(ordered_gene_lengths):
        if (i not in reset_points) or i == 0:
            row_x_positions.append((current_x_pos, length))
            current_x_pos += length
        else:
            current_x_pos = minorrowadjust
            ax_x_positions.append(row_x_positions)
            row_x_positions = list()
            row_x_positions.append((current_x_pos, length))
            current_x_pos += length
        if current_x_pos > max_x_pos:
            max_x_pos = current_x_pos
    ax_x_positions.append(row_x_positions)

    # convert from data to axis positions
    text_offset = 0.15
    ax_x_positions = [(start / max_x_pos, ((nrows - (i + 1)) / nrows) + text_offset, length / max_x_pos, (1 / nrows) - text_offset)
                       for i, row in enumerate(ax_x_positions) for start, length in row]

    axes = [mother_ax.inset_axes(bounds) for bounds in ax_x_positions]
    properGeneName = {gene: gene for gene in geneorder}
    properGeneName['HA_antigenic'] = 'Antigenic\nHA'
    properGeneName['HA_nonantigenic'] = 'Nonantigenic\nHA'
    properGeneName['PB1-F2'] = 'PB1-\nF2'
    for i, ((gene, length), ax) in enumerate(zip(ordered_gene_lengths, axes)):

        if i in reset_points:
            ax.set_ylabel(y_label, labelpad=12)
        else:
            ax.get_yaxis().set_visible(False)

        ax = sns.scatterplot(x='inGenePos', data=data.loc[(data['product'] == gene)], legend=False, ax=ax, linewidth=linewidth, **args)
        ax.set_xlim(left=0, right=length)
        ax.set_xlabel(properGeneName[gene], labelpad=8)
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(MultipleLocator(250))
        ax.set_ylim((ymin - ymax*0.04), (ymax + ymax*0.04))
        ax.tick_params(reset=True, which='both', axis='x', bottom=True, top=False)

        if no_zero_for_ha and gene == 'HA' and subtype == 'H3N2':
            ax.xaxis.get_major_ticks()[0].draw = lambda *args:None
            ax.xaxis.get_major_ticks()[1].draw = lambda *args:None
    return mother_ax


# Functions related to calculating bottleneck size

def getReadDepth(sample, segment, pos, alt):
    reffile = SNPs.loc[SNPs['sampleID'] == sample, 'referenceFile'].iloc[0]
    ref = reffile.split('/')[5]
    if 'Hong_Kong' in reffile:
        chrom = hongkongContigs[segment]
    elif 'Michigan' in reffile:
        chrom = ref[:-7] + segment
    elif ref[-2:] in ['17', '18', '19']:
        chrom = ref[:-2] + segment
    else:
        chrom = ref + '_' + segment
    bamfile = '/'.join(reffile.split('/')[0:6]) + '/map_to_consensus/' + sample + '.bam'
    pos = int(pos)
    sam = pysam.AlignmentFile(bamfile, "rb")
    pileup = sam.pileup(contig=chrom, start=pos-1, end=pos, truncate=True, stepper="nofilter")
    column = next(pileup)
    column.set_min_base_quality(30)
    try:
        bases = column.get_query_sequences(mark_matches=True)
        altreads = bases.count(alt.lower()) + bases.count(alt.upper())
    except:
        altreads = 0
    frequency = round((altreads / column.get_num_aligned()), 4)
    depth = column.get_num_aligned()
    return frequency, altreads, depth

def makeBottleneckInputFile(pairings, category):
    pairings = list(pairings)
    indexes = [pairing[0] for pairing in pairings]
    contacts = [pairing[1] for pairing in pairings]
    export = transmissionSNPs.loc[(transmissionSNPs['index'].isin(indexes)) & (transmissionSNPs.contact.isin(contacts)), ['index','contact','segment', 'pos', 'ref_nuc','alt_nuc', 'SNP_frequency_index', 'AD_index', 'depth_index','SNP_frequency_contact', 'AD_contact', 'depth_contact']]
    for ix, row in export.iterrows():
        if pd.isna(row.depth_contact):
            export.loc[ix, ['SNP_frequency_contact','AD_contact','depth_contact']] = getReadDepth(row.contact, ix[0], ix[1], row.alt_nuc_contact)
    export.fillna(0)
    filename = figures + '/bottleneck_figures/' + category.replace(' ', '_') + '.txt'
    export.to_csv(filename[:-4] + '.tsv', sep='\t')
    export = export.loc[(0.99 > export.SNP_frequency_index) & (export.SNP_frequency_index > 0.01)]
    export.loc[export['depth_contact'] == 0, ['SNP_frequency_contact', 'depth_contact', 'AD_contact']] = export.loc[export["depth_contact"]==0].apply(lambda x:getReadDepth(x['contact'], x['segment'], x['pos'], x['alt_nuc']), axis=1)
    export = export.loc[~export.duplicated()]
    export = export[['SNP_frequency_index', 'SNP_frequency_contact', 'depth_contact', 'AD_contact']].round(5)
    export.to_csv(filename, sep='\t', header=False, index=False)
    return filename

def koelleBottleneckCategorical(data, category):
    categories = list(data[category].unique())
    if np.nan in categories:
        categories.remove(np.nan)
    returnlist = []
    for group in categories:
        subdata = data.loc[data[category] == group]
        indexes = subdata['index']
        contacts = subdata['contact']
        assert (len(indexes) == len(contacts))

        pairings = (zip(indexes, contacts))

        filename = makeBottleneckInputFile(pairings, group)

        bottleneckregex = r"(?:size\n)(\d*)"
        lowerboundregex = r"(?:left bound\n)(\d*)"
        upperboundregex = r"(?:right bound\n)(\d*)"
        with open(f"{figures}/betabinomialResults_exact.log", 'a+') as outputFile:
            cmd = f'Rscript /d/orchards/betaBinomial/Bottleneck_size_estimation_exact.r --file {filename} --plot_bool TRUE --var_calling_threshold {SNP_frequency_cutoff} --Nb_min 1 --Nb_max 200 --confidence_level .95'
            outputFile.write(f"\n\n--------------------\n\n{group}\n\n")
            print (cmd)
            results = subprocess.run(cmd.split(" "), text=True, stdout=subprocess.PIPE)
            print (results.stdout)
            bottleneck = int(re.search(bottleneckregex, results.stdout).group(1))
            lowerbound = int(re.search(lowerboundregex, results.stdout).group(1))
            upperbound = int(re.search(upperboundregex, results.stdout).group(1))
            outputFile.write(f"{group}: {lowerbound}|--- {bottleneck} ---|{upperbound}")
            print (f"{group}: {lowerbound}|--- {bottleneck} ---|{upperbound}")
            try:
                os.rename('/mnt/d/orchards/betaBinomial/exact_plot.svg', f'{figures}/{group}_bottleneckplot_exact.svg')
            except:
                print (f'{category} doesn\'t have an exact plot svg')
        returnlist.append({'Pairing Category': group, 'Lower CI': lowerbound, 'Avg Bottleneck': bottleneck, 'Upper CI': upperbound})

    return returnlist

# Functions related to adding zeros to a log-scaled chart
def extract_exponent(x):
    return np.floor(np.log10(x))

def round_to_exponent(x, high=False):
    if high:
        return 10**extract_exponent(x) + 1
    else:
        return 10**extract_exponent(x)

def get_truncated_normal_distribution(mean=0, sd=1, low=0, upp=10, n=1):
    dist = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    return dist.rvs(n)

def set_log_ax_ytick_range(ax, r=1):
    t = np.log10(ax.get_yticks())
    new_t = 10**np.arange(t[0], t[-1], step=r)
    ax.set_yticks(new_t)
    return ax

def swarmplot_with_zeros(figargs, fig, spacing=.2, gap_between_data_and_zero=0, dotsize=5, jitter_zeros=False):
    y = figargs['y']
    figargs['data'] = data = figargs['data'].copy()

    log_new_zero = determine_log_zero(figargs, spacing) - gap_between_data_and_zero
    line_placement = 10**(log_new_zero + spacing)
    ax_bottom = 10**(log_new_zero - spacing)

    if not jitter_zeros:
        figargs['data'][y] = data[y].replace(0, 10**log_new_zero)
    else:
        data.loc[data[y] == 0, y] = jitter(data.loc[data[y] == 0, y], log_new_zero, (np.log10(ax_bottom * 1.05), np.log10(line_placement * .95)))
        figargs['data'] = data

    ax = sns.swarmplot(**figargs, size=dotsize)
    ax.axhline(line_placement, color='black', linestyle='--')

    old_ylim = ax.get_ylim()
    fig.canvas.draw_idle()
    add_zero_y_axis(ax, log_new_zero)
    ax.set_ylim(ax_bottom, old_ylim[1])

    return ax

def jitter(points, mid, ranges):
    low, high = ranges
    sd = (high - low) * .34
    return np.power(10, np.array(get_truncated_normal_distribution(mid, sd, low, high, len(points))))

def render_ax(ax):
    renderer = plt.gcf().canvas.renderer
    ax.draw(renderer)

def determine_log_zero(figargs, spacing):
    y = figargs['y']
    figargs['data'] = data = figargs['data'].copy()
    min_nonzero_y = data.loc[data[y] > 0, y].min()
    log_min_y = np.log10(round_to_exponent(min_nonzero_y)) - spacing
    return log_min_y

def add_zero_y_axis(ax, log_new_zero):
    yticks = ax.get_yticks()
    yticks[1] = 10**log_new_zero
    labels = [tick._text for tick in ax.get_yticklabels()]
    labels[1] = '0'
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    return ax
