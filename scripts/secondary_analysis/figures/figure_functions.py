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
from itertools import product
from tqdm import tqdm
from Bio import SeqIO

from figure_constants import *

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
    # sns.set(font='HelveticaNeue')
    rcParams['font.size'] = 36
    rcParams['font.stretch'] = 'condensed'

    rcParams['axes.labelsize'] = 26
    rcParams['axes.labelweight'] = 'bold'
    rcParams['xtick.labelsize'] = 18
    rcParams['ytick.labelsize'] = 18

    rcParams['figure.dpi'] = 300

    rcParams['font.family'] = 'Roboto'
    return rcParams

def addStatsLines(ax, x, y,  data, method='mean', hue=None, order = None, hue_order = None, dodge=True, linewidth=2, meanwidth=0.35, err_ratio=0.5, meancolor='black',color='black'):
    hw = meanwidth
    er = err_ratio

    xticks = [text.get_text() for text in ax.get_xticklabels()]
    dotlocs = [coll.get_offsets() for coll in ax.collections]
    sns_box_plotter = sns.categorical._BoxPlotter(x, y, hue, data, order, hue_order, orient=None, width=.8, color=None, palette=None, saturation=.75,  dodge=dodge, fliersize=5, linewidth=None)
    print (sns_box_plotter.group_names)
    if hue:
        hueOffsets = {hue:sns_box_plotter.hue_offsets[sns_box_plotter.hue_names.index(hue)] for hue in sns_box_plotter.hue_names}
        print (xticks)
        xlocs = {group:xticks.index(str(group[0]))+hueOffsets[group[1]] for group in product(sns_box_plotter.group_names, hueOffsets.keys())}

        groups = [x,hue]
        hw = hw/len(sns_box_plotter.hue_names)

    else:
        groups=[x]
        xlocs = {group:xcat for group,xcat in zip(sns_box_plotter.group_names, ax.xaxis.get_ticklocs())}

    for xcat,df in data.groupby(groups):
        xloc= xlocs[xcat]
        if method == 'median':
            middle = df[y].median()
            uppererror = np.percentile(df[y].dropna(), 75)-middle
            lowererror = middle - np.percentile(df[y].dropna(), 25)
            print (f'{middle}, 95th upper-lower {np.percentile(df[y].dropna(), 95)}, {np.percentile(df[y].dropna(), 5)}, 25th upper-lower {np.percentile(df[y].dropna(), 75)}, {np.percentile(df[y].dropna(), 25)}')
        else:
            middle = df[y].mean()
            uppererror = lowererror = df[y].sem()*1.96
        ax.hlines(middle, xloc-hw, xloc+hw, zorder=10, linewidth=linewidth, color=meancolor)
        ax.hlines((middle-lowererror, middle+uppererror), xloc-hw*er, xloc+hw*er, zorder=10,linewidth=linewidth,color=color)
        ax.vlines(xloc, middle-lowererror, middle+uppererror,zorder=10,linewidth=linewidth,color=color)
    return ax


def addRegStats(ax, x, y, data):
    def form(x):
        if x<0.01:
            return f'{x:.2e}'
        elif  x>10:
            exp = x//10
            exp += 3
            return f'{x:.3e}'
        else:
            return f'{x:.3}'

    m, b, r_value, p_value, std_err = stats.linregress(data[[x,y]].dropna().to_numpy())

    print (m, b, p_value)
#     props = dict(boxstyle='round', alpha=0.5,color=sns.color_palette()[0])
    textstr = f'y = {form(m)}x + {form(b)}\n$r^2$ = {form(r_value**2)}\nci = {form(std_err*1.96)}\np = {form(p_value)}'
    ax.text(0.05, .78, textstr, transform=ax.transAxes, ha="left", fontsize=rcParams['font.size']*.66)

def errorbar(x, y, low, high, order, color, ax):
    ynum = [order.index(y_i) for y_i in y]
    lowerrbar = [x-low for x, low in zip(x, low)]
    uppererrbar = [high-x for x, high in zip(x, high)]
    return ax.errorbar(ynum, x, yerr=(lowerrbar, uppererrbar), fmt="none", color=color, elinewidth=1, capsize=5)

def calc_ci(array, z=1.96):
    x_bar = np.mean(array) # mean of vector
    s = np.std(array) # std of vector
    n = len(array) # number of obs

    return (z * (s/math.sqrt(n)))

def bootstrap(array, num_of_bootstraps, function,*args, **kwargs):
    x_bar = function(array, *args, **kwargs)
    sampled_results = np.zeros(num_of_bootstraps)
    for i in range(num_of_bootstraps):
        sample = np.random.choice(array, len(array), replace=True)
        sampled_results[i] = function(sample, *args, **kwargs)
    deltastar = sampled_results-x_bar
    ci = np.percentile(deltastar, 2.5)
    return ci

def bootstrap_df(df, num_of_bootstraps, function,*args, **kwargs):
    x_bar = function(df, *args, **kwargs)
    sampled_results = np.zeros(num_of_bootstraps)
    for i in range(num_of_bootstraps):
        sample = df.sample(n=len(df), replace=True)#np.random.choice(array, len(array), replace=True)
        sampled_results[i] = function(sample, *args, **kwargs)
    deltastar = sampled_results-x_bar
    ci = np.nanpercentile(deltastar, 2.5)
    if ci==np.nan:
        print (sampled_results)
    return ci

def convertListofClassicH3N2SitestoZeroIndexedMStart(listOfSites):
    return [site+15 for site in listOfSites]

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
        gene_name = gene_name.replace("gene_id ","")
        gene_name = gene_name.replace("\"","")

        if annotation_type.lower() == "cds":
            if segment_name not in coding_regions:
                coding_regions[segment_name] = {}
                coding_regions[segment_name][gene_name] = [[start, stop]]
            elif segment_name in coding_regions and gene_name not in coding_regions[segment_name]:
                coding_regions[segment_name][gene_name] = [[start, stop]]
            elif gene_name in coding_regions[segment_name]:
                coding_regions[segment_name][gene_name].append([start, stop])

    return coding_regions

def makeManhattanPlot(ax, y, data, nrows=2, subtype=None, geneorder=None, antigenic=True, hue=None, hue_order=['Nonsynonymous','Synonymous'], palette_type = 'synon', color=None, dotsize=40, linewidth=0, alpha=1, negativeDataSet=False, y_label='Minor Allele\nFrequency', no_zero_for_ha=True):
    mother_ax = ax
    del ax
    mother_ax.get_yaxis().set_visible(False)
    mother_ax.get_xaxis().set_visible(False)
    mother_ax.spines['right'].set_visible(False)
    mother_ax.spines['top'].set_visible(False)
    mother_ax.spines['left'].set_visible(False)
    mother_ax.spines['bottom'].set_visible(False)
    if subtype:
        if subtype=='H1N1pdm':
            subtype = 'H1N1'
            data = data.replace('H1N1pdm','H1N1')
        data=data.loc[data.subtype==subtype]
    else:
        subtype = data.subtype.first()
    if geneorder:
        pass
    elif antigenic:
        geneorder = ["PB2", "PB1", "PA", "HA", "HA_antigenic","HA_nonantigenic", "NP",  "NA", "M1", "M2", "NS1", "NEP",'PB1-F2', 'PA-X']
    else:
        geneorder = ["PB2", "PB1", "PA", "HA", "NP",  "NA", "M1", "M2", "NS1", "NEP",'PB1-F2', 'PA-X']

    inDir = installDir+''

    refgeneEnds=dict()
    refgeneEndsCodons=dict()
    refgeneStarts=dict()

    args={'y':y,'hue':hue,'color':color, 'hue_order':hue_order,'palette':palettes[palette_type], 'alpha':alpha}
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
    ordered_gene_list = [x[0] for x in ordered_gene_lengths]

    current_length=0
    end_of_row = sum([length for gene, length in ordered_gene_lengths])/nrows


    reset_points = [0]
    for i, (gene, length) in enumerate(ordered_gene_lengths):
        if current_length > end_of_row:
            reset_points.append(i)
            current_length = length
        else:
            current_length += length

    ncolumns = reset_points[1]
    minorrowadjust = int(abs(sum(lengths[0:ncolumns])-sum(lengths[ncolumns:]))/2)

    #make subaxes
    ax_x_positions = list()
    row_x_positions = list()
    current_x_pos = 0
    max_x_pos = 0
    for i, (gene, length) in enumerate(ordered_gene_lengths):
        if (i not in reset_points) or i==0:
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

    #convert from data to axis positions
    text_offset = 0.15
    ax_x_positions = [(start/max_x_pos, ((nrows-(i+1))/nrows)+text_offset, length/max_x_pos, (1/nrows)-text_offset) for i, row in enumerate(ax_x_positions) for start, length in row]

    axes = [mother_ax.inset_axes(bounds) for bounds in ax_x_positions]
    properGeneName = {gene:gene for gene in geneOrder}
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
        ax.set_ylim(ymin-ymax*0.04, ymax+ymax*0.04)
        ax.tick_params(reset=True, which='both', axis='x', bottom=True, top=False)
        print (gene)
        if no_zero_for_ha and gene=='HA' and subtype=='H3N2':
            print (ax.xaxis.get_major_ticks())
            ax.xaxis.get_major_ticks()[0].draw = lambda *args:None
            ax.xaxis.get_major_ticks()[1].draw = lambda *args:None
            # new_labels = ax.get_xticklabels()
            # # new_labels[0].set_text('')
            # print (new_labels)
            # ax.set_xticklabels(new_labels)
        # ax.tick_params(spad=0)
    return mother_ax

def makeRollingManhattanPlot(y, data, hue=None, hue_order=None, color=None,subtype='H3N2'):
    data=data.loc[data.subtype==subtype]
    print (subtype)
    print(len(data))
    args={'y':y,'hue':hue,'color':color, 'hue_order':hue_order, 'palette':sns.color_palette([snsblue, snsorange], 2)}
    
    ymax = data[y].mean()+data[y].sem()*1.96*1.1#*0.8#raise Exception
    ymin = 0
    
    genelist = list(data['product'].unique())
    if np.nan in genelist:
        genelist.remove(np.nan)
    
    nrows=2
    geneEnds=refgeneEnds[subtype]
    lengths = sorted([int(data.loc[data['product']==gene, 'codon'].max()) for gene in genelist], reverse=True)
    genelengths = [(gene,int(data.loc[data['product']==gene, 'codon'].max())) for gene in genelist]
    genelengths.sort(key=itemgetter(1),reverse=True)
    genelist = list(map(itemgetter(0),genelengths))
    genelengths = [tuple(list(tup)+[i]) for i, tup in enumerate(genelengths)]

    halfway=0
    goal = sum(lengths)/2
    for gene, length, i in genelengths:
        if halfway > goal:
            break
        else:
            halfway += length

    ncolumns = i-1
    currentrcParams = rcParams
    #rcParams.update({'axes.labelpad': rcParams['axes.labelpad']*3})

#     rcParams['font.family']='SF Pro Display'
    rcParams['font.weight']='bold'
    rcParams['font.size'] = 36
    rcParams['font.stretch'] = 'condensed'

    rcParams['axes.labelsize'] = 30
    rcParams['axes.labelweight'] = 'bold'
    rcParams['xtick.labelsize'] = 24
    rcParams['ytick.labelsize'] = 24
    
    fig = plt.figure(figsize=(27,12))
    

    width = int(max(sum(lengths[0:ncolumns]), sum(lengths[ncolumns:])))
    
    gs=GridSpec(2,width, figure=fig, wspace=0.05, hspace=0.3) # 2 rows, 7000 AA's long
    
    minorrowadjust = int(abs(sum(lengths[0:ncolumns])-sum(lengths[ncolumns:]))/2)
    
    start = minorrowadjust
    end = minorrowadjust
    row = 0
    
        #coords = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    for gene, length, i in genelengths:
        end = end+length
        innergs = GridSpecFromSubplotSpec(10,1,subplot_spec=gs[row,start:end], wspace=0, hspace=0.1)

        ax0 = plt.subplot(innergs[0,0])
        ax0.get_yaxis().set_visible(False)
        ax0.get_xaxis().set_visible(False)
        #When fully implemented this will be a for loop through regions.
        #Size of each subbar will be [widthofbar]/length, from [lengthofpreviousbars]/length

        #Commenting out bars displaying regions
        #bar = Rectangle((0,0), width=length/length, height = 1, transform = ax0.transAxes, facecolor='red')
        #ax0.add_patch(bar)

        ax = plt.subplot(innergs[0:10,0])#make it innergs[1:10,0] to add bar back
        #ax = plt.subplot(innergs[1:10,0])
        ax = sns.lineplot(x='codon', data = data.loc[(data['product']==gene)], legend=False, ax=ax, **args)

        #ax = sns.scatterplot(x='inGenePos', data=data.loc[(data['product'] == gene)&(data[y]>0.01)], legend=False, ax=ax, linewidth=0, **args)
        if gene=='PB1-F2':
            gene = ' PB1\n-F2'
        ax.set_xlim(left=0.001, right=length)
        ax.set_xlabel(gene, labelpad=8)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(250))
        ax.set_ylim(ymin-ymax*0.05, ymax+ymax*0.05)
        ax.tick_params(pad=0)

        start = start+length
        if i != 0:# or i != ncolumns:
            ax.get_yaxis().set_visible(False)
        else:
            ax.set_ylabel(None)
        if i == ncolumns:
            ax.get_yaxis().set_visible(True)
            ax.set_ylabel(None)

        if i+1 == ncolumns:
            start = 0
            end = 0
            row = 1
    plt.gca().set_ylabel('Pi', labelpad=10)
    rcParams.update(currentrcParams)
    return fig



def createBoxEventChart(df, subtype, season, generateImage=False, color=None): #chopping block: height+width
    # instantiate a new figure object
    #fig = plt.figure(figsize=(20,5))

    #Data should be in observation format with dates
    
    
    df = df.loc[(df.subtype==subtype)&(df.season==season)]

    df = df.sort_values('week').reset_index()
    daterange = pd.to_datetime(df.sample_date.values)
    daterange = pd.date_range(daterange.min()-pd.to_timedelta(1, unit='W'), daterange.max(), freq='W').strftime('%m/%d').to_list()

    #print (df.loc[df['sampleID']=='18VR003060'])
    
    #process dates in metafile, get starting day of the week each infection was during
    
    distanceFile = sampleFolderDict[subtype][season]+'/snp_calls/sequencedistances.tsv'
    distances = pd.read_csv(distanceFile, **read_tsv_args).set_index('Unnamed: 0')
    distances = distances.loc[distances.index.isin(df['sampleID']), distances.columns.isin(df['sampleID'])]
    
    enddate = df.week.max()
    width = len(daterange)
    
    #print (list(distances.index).index('18VR003060'))
    #print (list(distances.columns).index('18VR004483'))
    
    df['x'] = abs(df.week-26)
    df['x'] = df.x-df.x.min()
    df['y'] = df.groupby('x').cumcount()
    height = df.y.max()+1
    df = df.reset_index(drop=True)
    df = df.sort_values('sampleID')
    #assert (list(df.index)[20] == list(distancefile.index)[20])
    #data = df.iloc[1:,2:-6].values


    # df = df[['x','y','distA','distB','distC']]
    # colors=df.loc[:,'distA':'distC'].values
    # colors=abs(colors-1) #invert colors so that low distance = bright int that channel


    #create PCA analysis of 3 variables:
    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled=scaler.fit_transform(distances.values)
    print (data_rescaled)
    pca=PCA().fit(data_rescaled)
    #plt.figure()
    #plt.plot(np.cumsum(pca.explained_variance_ratio_)) #(gives you sense of how much variance is explained as you drop each variable)
    pca3=PCA(n_components=components)
    PCs=pca3.fit_transform(data_rescaled)
    colors=scaler.fit_transform(PCs)
    df = df.reset_index(drop=True)
    if components == 3:
        hsvcolors = rgb_to_hsv(colors)
        hsvcolors[:,2] *=.2 #compress value (aka brightness) to reduce improper effect of brightness on notability while still retaining that data
        hsvcolors[:,2] = abs(1-hsvcolors[:,2]) #invert brightness to make the graph less dark and more pretty :)
        colors = hsv_to_rgb(hsvcolors)
        df[['hue','sat','val']] = pd.DataFrame(hsvcolors)
        df[['red','green','blue']] = pd.DataFrame(colors)
        df[['PC1','PC2','PC3']] = pd.DataFrame(PCs)
    elif components==2:
        df[['PC1','PC2']] = pd.DataFrame(PCs)
        df[['PC1_scaled','PC2_scaled']] = pd.DataFrame(colors)
        return df
    #print (df.loc[df['sampleID']=='18VR004997'])

    df=df.sort_values(by='hue', ascending=False).reset_index(drop=True) #sort by hue
    df['y'] = df.groupby('x').cumcount()
    print(df.columns)
    df = df[['sampleID','week', 'x','y','hue','sat','val','red','green','blue','PC1','PC2','PC3']]
    colors=df.loc[:,'red':'blue'].values
    
    #export colors and metadata for phylogeny
    df['colors']=[rgb2hex(colors[i,:]) for i in range(colors.shape[0])]
    toExport = df[['sampleID','week','colors']]
    toExport['week'] = 'label'
    header = pd.DataFrame([{'ID':'TREE_COLORS', 'week':"", 'colors':''},{'ID':'SEPARATOR TAB', 'week':"", 'colors':''},{'ID':'DATA', 'week':"", 'colors':''}])
    toExport = pd.concat([header, toExport], sort=True).reset_index(drop=True)
    toExport = toExport[['ID','week','colors']]
    toExport.to_csv(os.path.join(os.path.dirname(distanceFile),"Phylogeny_metadata.txt"),sep='\t',index=False, header=False)
    #create 2d array from "x", "y", "color":

    #colors=np.vstack(df.color)
    if generateImage:
        plt_shp = (df.y.max()+1, df.x.max()+1, 3)

        pltarray=np.full(plt_shp,float(1))
        pltarray[df.y,df.x] = colors


        plt.figure(figsize=(width,height))

        plt.imshow(pltarray, origin='lower', aspect='equal')


        # get the axis
        ax = plt.gca()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)

        # set minor ticks
        ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
        ax.set_yticks(np.arange(-.5, (height-1), 1), minor=True)


        # ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.tick_params(axis='both', which='minor',color='w')

        # add gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        print (height, width)
        plt.xticks(range(0,len(daterange)), daterange, fontsize=17, fontweight='bold')
        plt.yticks([])
    return df
#     #fig.savefig(os.path.join(os.path.dirname(distanceFile),"waffleChart.svg"))


# Functions related to calculating bottleneck size

def getReadDepth(sample, segment, pos, alt):
    hongkongContigs = {'NP':'A_Hong_Kong_4801_2014_834574_NP', 'NS':'A_Hong_Kong_4801_2014_834575_NS', 
                   'MP':'A_Hong_Kong_4801_2014_834576_MP', 'PA':'A_Hong_Kong_4801_2014_834577_PA',
                   'PB2':'A_Hong_Kong_4801_2014_834578_PB2', 'PB1':'A_Hong_Kong_4801_2014_834579_PB1',
                  'NA':'A_Hong_Kong_4801_2014_834580_NA','HA':'A_Hong_Kong_4801_2014_834581_HA'}
    
    reffile = SNPs.loc[SNPs['sampleID']==sample, 'referenceFile'].iloc[0]
    ref = reffile.split('/')[5]
    if 'Hong_Kong' in reffile:
        chrom = hongkongContigs[segment]
    elif 'Michigan' in reffile:
        chrom = ref[:-7]+segment
    elif ref[-2:] in ['17','18','19']:
        chrom = ref[:-2]+segment
    else:
        chrom = ref+'_'+segment
    bamfile = '/'.join(reffile.split('/')[0:6])+'/map_to_consensus/'+sample+'.bam'
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
    frequency = round(altreads/column.get_num_aligned(),4)
    depth = column.get_num_aligned()
    return frequency, altreads, depth

def makeBottleneckInputFile(pairings, category):
    exportdata = pd.DataFrame()
    pairings = list(pairings)
    indexes = [pairing[0] for pairing in pairings]
    contacts = [pairing[1] for pairing in pairings]
    export = transmissionSNPs.loc[(transmissionSNPs['index'].isin(indexes)) & (transmissionSNPs.contact.isin(contacts)), ['index','contact','segment', 'pos', 'ref_nuc','alt_nuc', 'SNP_frequency_index', 'AD_index', 'depth_index','SNP_frequency_contact', 'AD_contact', 'depth_contact']]
    for ix, row in export.iterrows():
        if pd.isna(row.depth_contact):
            export.loc[ix,['SNP_frequency_contact','AD_contact','depth_contact']] = getReadDepth(contact, ix[0],ix[1],row.alt_nuc_contact)
    export.fillna(0)
    filename = figures+'/bottleneck_figures/'+category.replace(' ','_')+'.txt'
    export.to_csv(filename[:-4]+'.tsv', sep='\t')
    export = export.loc[(0.99 > export.SNP_frequency_index) & (export.SNP_frequency_index > 0.01)]
    export.loc[export['depth_contact']==0, ['SNP_frequency_contact','depth_contact', 'AD_contact']] = export.loc[export["depth_contact"]==0].apply(lambda x:getReadDepth(x['contact'], x['segment'], x['pos'], x['alt_nuc']), axis=1)
    export = export.loc[~export.duplicated()]
    export = export[['SNP_frequency_index','SNP_frequency_contact','depth_contact','AD_contact']].round(5)
    

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
        
        pairings = (zip(indexes,contacts))
        
        filename = makeBottleneckInputFile(pairings, group)
    
        bottleneckregex = r"(?:size\n)(\d*)"
        lowerboundregex = r"(?:left bound\n)(\d*)"
        upperboundregex = r"(?:right bound\n)(\d*)"
        print (f"{figures}/betabinomialResults_exact.log")
        with open(f"{figures}/betabinomialResults_exact.log", 'a+') as outputFile:
            cmd = f'Rscript /d/orchards/betaBinomial/Bottleneck_size_estimation_exact.r --file {filename} --plot_bool TRUE --var_calling_threshold {SNP_frequency_cutoff} --Nb_min 1 --Nb_max 200 --confidence_level .95'
            outputFile.write(f"\n\n--------------------\n\n{group}\n\n")
            print (cmd)
            try:
                results = subprocess.run(cmd.split(" "), text=True, stdout=subprocess.PIPE)
            except:
                raise Exception
            print (results.stdout)
            bottleneck = int(re.search(bottleneckregex, results.stdout).group(1))
            lowerbound = int(re.search(lowerboundregex, results.stdout).group(1))
            upperbound = int(re.search(upperboundregex, results.stdout).group(1))
            outputFile.write(f"{group}: {lowerbound}|--- {bottleneck} ---|{upperbound}")
            print (f"{group}: {lowerbound}|--- {bottleneck} ---|{upperbound}")
            try:
                os.rename('/mnt/d/orchards/betaBinomial/exact_plot.svg', f'{figures}/{pair}_bottleneckplot_exact.svg')
            except:
                print (f'{category} doesn\'t have an exact plot svg')
        returnlist.append({'Pairing Category':group, 'Lower CI':lowerbound, 'Avg Bottleneck':bottleneck, 'Upper CI':upperbound})
    
    return returnlist