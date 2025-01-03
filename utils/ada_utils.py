from .metadata_utils import csv2df
from .qmri_utils import *

import os
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon,ks_2samp, ttest_ind, f_oneway,  ttest_rel,mannwhitneyu, friedmanchisquare, shapiro, tukey_hsd,kruskal

from scipy.optimize import curve_fit
from scipy.ndimage import binary_erosion
from scipy.ndimage import generate_binary_structure
from scipy import stats


from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

import statsmodels.api as sm
import xml.etree.ElementTree as ET

# Data import
def parse_xml_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []
    for label in root.findall('.//label'):
        index = int(label.get('index'))
        x = int(label.get('x'))
        y = int(label.get('y'))
        z = int(label.get('z'))
        name = label.text
        labels.append({'index': index+1, 'x': x, 'y': y, 'z': z, 'name': name})
    
    return labels

def get_labels_as_dict():

    print('*********** Import Labels Index and Name ************')
    # Import HO/JHU Cortical Labels
    labels_ho = parse_xml_labels("/home/mroulet/fsl/data/atlases/HarvardOxford-Cortical.xml" )
    labels_jhu = parse_xml_labels("/home/mroulet/fsl/data/atlases/JHU-labels.xml")

    # Filter the list of dictionaries based on wm_lst
    #wm_lst = list(range(16, 36))
    #labels_jhu = [label for label in labels_jhu if label['index'] in wm_lst]

    # Set FeTA Labels
    labels_feta = []
    for index,name in zip(range(8),["background","csf","gm","wm","ventr","cerebellum","deep_gm","bs"]):
        labels_feta.append({'index': index, 'name': name})

    return labels_ho, labels_jhu, labels_feta

def import_maps_as_dict(metadata, bids_path,t2map_dirname):
    
    print("*************** Import Maps ****************")

    t2map = {}
    feta = {}
    t2recon= {}
    jhu = {}
    ho = {}

    for (sub,ses), acq  in metadata.groupby(["sub","ses"]):

        recon_dirname = 'recon_1mm'
        labels_dirname = recon_dirname + '_feta'
        jhu_dirname = recon_dirname + '_jhu'
        ho_dirname = recon_dirname + '_ho'

        print(f"{sub}_{ses}")
        if sub not in t2map.keys():
            t2map[sub] = {}
            feta[sub] = {}
            jhu[sub] = {}
            ho[sub] = {}
            t2recon[sub] = {}
        t2recon[sub][ses] = {}

        # import t2map
        t2map_dir = os.path.join(bids_path, 'prj-004','derivatives',t2map_dirname,sub,ses,'anat')
        t2map_dir = os.path.join(bids_path, 'prj-004','derivatives',t2map_dirname,sub,ses,'anat')
        print(t2map_dir)
        t2_flnm = glob.glob(f"{t2map_dir}/*t2map*.nii.gz")
        print("t2map: ",t2_flnm[0])
        t2map_img = sitk.ReadImage(t2_flnm[0])
        t2map[sub][ses] = sitk.GetArrayFromImage(t2map_img)

        # import labels
        t2lbls_dir = os.path.join(bids_path, 'prj-004','derivatives',labels_dirname,sub,ses,'anat')
        t2lbl_flnm = glob.glob(f"{t2lbls_dir}/*.nii.gz")
        t2lbl_flnm.sort()
        print("feta :",t2lbl_flnm[0])
        label_img = sitk.ReadImage(t2lbl_flnm[0])
        feta[sub][ses] = sitk.GetArrayFromImage(label_img)

        # import jhu labels
        t2lbls_dir = os.path.join(bids_path, 'prj-004','derivatives',jhu_dirname,sub,ses,'anat')
        t2lbl_flnm = glob.glob(f"{t2lbls_dir}/*.nii.gz")
        print("jhu label: ",t2lbl_flnm[0])
        label_img = sitk.ReadImage(t2lbl_flnm[0])
        jhu[sub][ses] = sitk.GetArrayFromImage(label_img)
        # import HO labels
        t2lbls_dir = os.path.join(bids_path, 'prj-004','derivatives',ho_dirname,sub,ses,'anat')
        t2lbl_flnm = glob.glob(f"{t2lbls_dir}/*.nii.gz")
        print("ho_label: ",t2lbl_flnm[0])
        label_img = sitk.ReadImage(t2lbl_flnm[0])
        ho[sub][ses] = sitk.GetArrayFromImage(label_img)

        # import reconstruction
        recon_dir = os.path.join(bids_path, 'prj-004','derivatives',recon_dirname,sub,ses,'anat')
        recon_flnms = glob.glob(f"{recon_dir}/*.nii.gz")
        recon_flnms = [s for s in recon_flnms]
        for recon_flnm in sorted(recon_flnms):
            #print(recon_flnm)
            recon_img = sitk.ReadImage(recon_flnm)
            print(recon_flnm)
            match = re.search(r'te-(\d+)_recon', recon_flnm)
            te = int(match.group(1))
            t2recon[sub][ses][te] = sitk.GetArrayFromImage(recon_img)

        
    return t2map, t2recon, jhu, ho, feta

def get_t2_per_roi(t2map,feta,ho,labels_ho,jhu,labels_jhu):
    
    print("*************** Get T2 per ROI ****************")

    t2ho = {}
    t2jhu = {}
    t2ho_csv = []
    t2jhu_csv = []

    # Define structural element (cross-shaped) for erosion
    structure_element = generate_binary_structure(3, 3)

    for sub in t2map.keys():
        if sub not in t2ho.keys():
            t2ho[sub] = {}
            t2jhu[sub] = {}

        for ses in t2map[sub].keys():
            print(sub, ses)
            if ses == 'ses-02':
                scanner = 'sola'
            else:
                scanner = 'freemax'

            data_feta = feta[sub][ses]
            data_ho = ho[sub][ses]
            data_jhu = jhu[sub][ses]
            data = t2map[sub][ses]

            t2ho[sub][ses] = {}
            t2jhu[sub][ses] = {}

            # **************** HO *****************
            # iterate through labels except background
            for label in labels_ho:
                # Find the intersection indices (take both within wm and gm)
                intersection_indices = np.logical_and((data_feta == 2), data_ho == label["index"])

                # Perform erosion
                intersection_indices = binary_erosion(intersection_indices, structure=structure_element)
                
                data_label = data[intersection_indices]
                data_label = data_label.flatten()

                t2ho[sub][ses][label['index']] =  {'name': label['name'], 
                                                    'data': data_label, 
                                                    'n_data': len(data_label), 
                                                    'mean': np.mean(data_label),
                                                    'median': np.median(data_label),
                                                    'std': np.std(data_label)}
                
                t2ho_csv.append({ 'sub': sub,
                                'ses': ses,
                                'scanner': scanner,
                                'roi': label['name'],
                                'mean': np.mean(data_label),
                                'std': np.std(data_label),
                                'nvoxel': len(data_label)})
                
            # **************** JHU *****************
            # iterate through labels except background
            for label in labels_jhu:
                # Find the intersection indices (take both within wm and gm)
                intersection_indices = np.logical_and((data_feta == 3), data_jhu == label["index"])

                # Perform erosion
                intersection_indices = binary_erosion(intersection_indices, structure=structure_element)
                
                data_label = data[intersection_indices]
                data_label = data_label.flatten()

                t2jhu[sub][ses][label['index']] = { 'name': label['name'], 
                                                    'data': data_label, 
                                                    'n_data': len(data_label), 
                                                    'mean': np.mean(data_label),
                                                    'median': np.median(data_label),
                                                    'std': np.std(data_label)}
                
                t2jhu_csv.append({ 'sub': sub,
                                'ses': ses,
                                'scanner': scanner,
                                'roi': label['name'],
                                'mean': np.mean(data_label),
                                'std': np.std(data_label),
                                'nvoxel': len(data_label)})

    return t2ho, t2jhu, pd.DataFrame(t2jhu_csv),pd.DataFrame(t2ho_csv)

def plot_cov_boxplot_new(t2jhu,t2ho,wm_lst, gm_lst):
    colors = ['#5ab6b1','#1e4747']
    facecolors = ['#2fac66','#2fac66','#2fac66','#6A8EAE']
    for t2,tissuetype,tiss_list,j in zip([t2jhu,t2ho],["White Matter","Cortical Gray Matter"],[wm_lst,gm_lst],[1,2]):
        print(tissuetype)
        CoV_interrun = []
        for index in tiss_list:
            roi_mean = []
            for sub,ses in [['sub-003','ses-03'],['sub-003','ses-04']]:
                roi_mean.append(t2[sub][ses][index]['mean'])    
            roi_mean = [x for x in roi_mean if not np.isnan(x)]
            if len(roi_mean) > 1 :
                CoV_interrun.append(100*np.std(roi_mean)/np.mean(roi_mean))
                if 100*np.std(roi_mean)/np.mean(roi_mean) > 15:
                    print(t2[sub][ses][index]['name'])

        CoV_interses = []
        for index in tiss_list:
            roi_mean = []
            for sub,ses in [['sub-003','ses-01'],['sub-003','ses-03']]:
                roi_mean.append(t2[sub][ses][index]['mean'])
            roi_mean = [x for x in roi_mean if not np.isnan(x)]
            if len(roi_mean) > 1 :
                CoV_interses.append(100*np.std(roi_mean)/np.mean(roi_mean))

                
        CoV_intersub = []
        for index in tiss_list:
            roi_mean = []
            for sub in ['sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008','sub-009','sub-010','sub-011']:
                roi_mean.append(t2[sub]['ses-01'][index]['mean'])
            roi_mean = [x for x in roi_mean if not np.isnan(x)]
            if len(roi_mean) > 1 :
                CoV_intersub.append(100*np.std(roi_mean)/np.mean(roi_mean))
                if 100*np.std(roi_mean)/np.mean(roi_mean) > 15:
                    print(t2[sub]['ses-01'][index]['name'])

        CoV_intersub_hf = []
        for index in tiss_list:
            roi_mean = []
            for sub in ['sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008','sub-009','sub-010','sub-011']:
                roi_mean.append(t2[sub]['ses-02'][index]['mean'])
            roi_mean = [x for x in roi_mean if not np.isnan(x)]
            if len(roi_mean) > 1 :
                CoV_intersub_hf.append(100*np.std(roi_mean)/np.mean(roi_mean))
                if 100*np.std(roi_mean)/np.mean(roi_mean) > 15:
                    print(t2[sub]['ses-01'][index]['name'])

        #CoVs = [CoV_interrun, CoV_interses, CoV_intersub]
        CoVs = [ CoV_interrun, CoV_interses, CoV_intersub, CoV_intersub_hf]

        plt.subplots(figsize=(4,4))

        for CoV, pos,cov_col in zip(CoVs,range(1,5),facecolors):

            # Generate random x-values for scatter plot
            random_x = np.random.rand(len(CoV)) * 2 + 1  # Random x-values between 1 and 3

            #box = plt.boxplot(CoV_interrun, labels=['CV'])
            boxcol= '#222222'
            bplot = plt.boxplot(CoV, positions = [pos], showfliers=False,
                                    widths=0.5,patch_artist=True,boxprops=dict(edgecolor=boxcol,  facecolor=( *mcolors.hex2color(cov_col),0.9)),
                                    whiskerprops=dict(color=boxcol),medianprops=dict(color=boxcol),
                                    capprops=dict(color=boxcol), meanline=True,showmeans=True,meanprops=dict(color='red'),zorder=2)

            # Calculate the range for random x-values
            box_x = [item.get_xdata() for item in bplot['whiskers']]
            box_width = box_x[1][1] - box_x[0][1]  # Width of the boxplot
            center_x = (box_x[0][1] + box_x[1][1]) / 2  # Center x-coordinate of the boxplot
            x_min = pos - 0.5 / 5  # Minimum x-value for random_x
            x_max = pos + 0.5 / 5  # Maximum x-value for random_x
            print(f'MEAN COV: {np.mean(CoV), np.std(CoV), len(CoV)}')
            # Generate random x-values for scatter plot
            random_x = np.random.uniform(x_min, x_max, len(CoV))

            # Plot scatter plot of all points
            plt.scatter(random_x, CoV, alpha=0.4, color='gray',edgecolors='none',zorder=2)


        print('\na=0.001',0.001 / len(tiss_list))
        print('a=0.05',0.05 / len(tiss_list))
        print('a=0.01',0.01 / len(tiss_list))

        # wilcoxon test
        t_statistic, p_value = wilcoxon(CoV_interrun,CoV_interses)
        print("\nRUN vs SES ANALYSIS")
        print("T-statistic:", np.round(t_statistic))
        print("P-value:", p_value)

        # Interpret the results
        
        alpha = 0.01 / len(tiss_list)
        if p_value < alpha:
            print("REJECT the null hypothesis: There is a significant difference between the means of the two samples.\n\n")
        else:
            print("DO NOT REJECT the null hypothesis: There is no significant difference between the means of the two samples.\n\n")
        
        # wilcoxon test
        t_statistic, p_value = wilcoxon(CoV_intersub,CoV_interses)
        print("SES vs SUB COV ANALYSIS")
        print("T-statistic:", np.round(t_statistic))
        print("P-value:", p_value)

        # Interpret the results
        alpha = 0.01 / len(tiss_list)
        if p_value < alpha:
            print("REJECT the null hypothesis: There is a significant difference between the means of the two samples.\n\n")
        else:
            print("DO NOT REJECT the null hypothesis: There is no significant difference between the means of the two samples.\n\n")


        # wilcoxon test
        t_statistic, p_value = wilcoxon(CoV_intersub,CoV_intersub_hf)
        print("0.55 vs 1.5 COV ANALYSIS")
        print("T-statistic:", np.round(t_statistic))
        print("P-value:", p_value)

        # Interpret the results
        alpha = 0.01 / len(tiss_list)
        if p_value < alpha:
            print("REJECT the null hypothesis: There is a significant difference between the means of the two samples.\n\n")
        else:
            print("DO NOT REJECT the null hypothesis: There is no significant difference between the means of the two samples.\n\n")



        plt.ylabel('CoV (%)',fontsize=13)
        #plt.title(f'CoV of {tissuetype} (%)')
        plt.grid('on',zorder =0)
        if tissuetype == "White Matter":
            plt.ylim([0,10])
            plt.yticks([0,1,2,3,4,5,6,7,8],fontsize=13)
        else:
            plt.ylim([0,25])
            plt.yticks([0,2,5,5,10,15,20], fontsize=13)
        
        plt.xticks([1,2,3,4], ['inter\nrun\n(0.55T)','inter\nsession\n(0.55T)', 'inter\nsubject\n(0.55T)', 'inter\nsubject\n(1.5T)'],fontsize=12)
        #plt.xticks([1,2], ['inter-run','inter-ses'])
        #plt.savefig(f"/home/mroulet/Desktop/final/cov_{str(i)}.pdf")
        plt.savefig(f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/ada/figures/cov_{str(j)}_repeat.pdf')
        plt.show()

def plot_pearson_corr_new(t2jhu,t2ho,wm_lst, gm_lst):

    ## Regression LINE
    ## INTER-RUN ###################################################################################
    k = 0
    # Define the pattern to match text within parentheses
    pattern = r'\([^)]*\)'
    colors = ['#5ab6b1','#1e4747']
    stat = {}
    lims = [[70,100], [90,120]]
    #lims = [[100,140], [120,220]]
    for t2,tissue_lst,tissuetype,lim,tisscolor in zip([t2jhu,t2ho],[wm_lst,gm_lst],["White Matter","Cortical Gray Matter"],lims,colors):
        k+=1
        data_x_std = []
        data_y_std = []
        data_x_mean = []
        data_y_mean = []

        for index in tissue_lst:
            data_x_mean.append(t2['sub-003']['ses-03'][index]["mean"])
            data_x_std.append(t2['sub-003']['ses-03'][index]["std"])

        for index in tissue_lst:
            data_y_mean.append(t2['sub-003']['ses-04'][index]["mean"])
            data_y_std.append(t2['sub-003']['ses-04'][index]["std"])

        data_mean = np.stack((data_x_mean, data_y_mean), axis=1)
        data_std = np.stack((data_x_std, data_y_std), axis=1)

        # Find rows containing NaN
        nan_rows = np.isnan(data_mean).any(axis=1)

        # Filter out rows containing NaN
        data_mean = data_mean[~nan_rows]
        data_std = data_std[~nan_rows]

        print(data_std.shape)
        print(data_mean.shape)

        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(data_mean[:,0],data_mean[:,1])
        print(r_value, p_value)
        line_x = np.array([0,500])
        line_y = slope * line_x + intercept

        # Create scatter plot with error bars
        plt.figure(figsize=(4, 4))

        #for i in range(len(data_std)):
        #    plt.errorbar(data_mean[i,0], data_mean[i,1], xerr=data_std[i,0], yerr=data_std[i,1], fmt='o', markersize=10, color='#555555', ecolor='gray', capsize=5)

        plt.scatter(data_mean[:,0], data_mean[:,1], color=tisscolor, marker='o', s=50, edgecolor='none',alpha=1)

        # Plot regression line
        plt.plot(line_x, line_y, color=tisscolor, linestyle='--', label='Regression Line')
        plt.plot([0,500],[0,500],color = 'black',linestyle = '--', alpha= 0.2)

        # Add labels and title
        plt.xlabel('Run 1 - T2 (ms)',fontsize=13)
        plt.ylabel('Run 2 - T2 (ms)',fontsize=13)
        #plt.title('Inter-Session')
        plt.xlim(lim)
        plt.ylim(lim)

        # Add Pearson correlation coefficient and p-value to the plot
        #plt.text(0.1, 0.9, f'Pearson correlation coefficient: {r_value:.2f}\nP-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=10)
        if tissuetype == 'White Matter':
            decision= '*' if p_value < 0.05/27 else 'ns'
            decision= '**' if p_value < 0.01/27 else '*'
            decision= '***' if p_value < 0.001/27 else '**'
        else:
            decision= '*' if p_value < 0.05/41 else 'ns'
            decision= '**' if p_value < 0.01/41 else '*'
            decision= '***' if p_value < 0.001/41 else '**'
        plt.text(0.99, 0.01, f'Pearson correlation coef.: {r_value:.2f}\n{decision} p = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=11,horizontalalignment='right')

        # Add legend
        #plt.legend(loc='lower right')
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        # Show plot
        plt.grid(True,zorder=0)
        plt.savefig(f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/ada/figures/reg_{str(k)}.pdf')
        plt.show()

    ## INTER-SESSION ###################################################################################

    # Define the pattern to match text within parentheses
    pattern = r'\([^)]*\)'
    colors = ['#5ab6b1','#1e4747']
    stat = {}
    for t2,tissue_lst,tissuetype,lim,tisscolor in zip([t2jhu,t2ho],[wm_lst,gm_lst],["White Matter","Cortical Gray Matter"],lims,colors):
        k+=1
        data_x_std = []
        data_y_std = []
        data_x_mean = []
        data_y_mean = []

        for index in tissue_lst:
            data_x_mean.append(t2['sub-003']['ses-01'][index]["mean"])
            data_x_std.append(t2['sub-003']['ses-01'][index]["std"])

        for index in tissue_lst:
            data_y_mean.append(t2['sub-003']['ses-04'][index]["mean"])
            data_y_std.append(t2['sub-003']['ses-04'][index]["std"])

        data_mean = np.stack((data_x_mean, data_y_mean), axis=1)
        data_std = np.stack((data_x_std, data_y_std), axis=1)

        # Find rows containing NaN
        nan_rows = np.isnan(data_mean).any(axis=1)

        # Filter out rows containing NaN
        data_mean = data_mean[~nan_rows]
        data_std = data_std[~nan_rows]

        print(data_std.shape)
        print(data_mean.shape)

        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(data_mean[:,0],data_mean[:,1])
        print(r_value, p_value)
        line_x = np.array([0,500])
        line_y = slope * line_x + intercept

        # Create scatter plot with error bars
        plt.figure(figsize=(4,4))

        #for i in range(len(data_std)):
        #    plt.errorbar(data_mean[i,0], data_mean[i,1], xerr=data_std[i,0], yerr=data_std[i,1], fmt='o', markersize=10, color='#555555', ecolor='gray', capsize=5)

        plt.scatter(data_mean[:,0], data_mean[:,1], color=tisscolor, marker='o', s=50, edgecolor='none',alpha=1)

        # Plot regression line
        plt.plot(line_x, line_y, color=tisscolor, linestyle='--', label='Regression Line')
        plt.plot([0,500],[0,500],color = 'black',linestyle = '--', alpha= 0.2)

        # Add labels and title
        plt.xlabel('Session 1 - T2 (ms)',fontsize=13)
        plt.ylabel('Session 2 - T2 (ms)',fontsize=13)
        #plt.title('Inter-Subject')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        # Add Pearson correlation coefficient and p-value to the plot
        #plt.text(0.1, 0.9, f'Pearson correlation coefficient: {r_value:.2f}\nP-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=10)
        if tissuetype == "White Matter":
            decision= '*' if p_value < 0.05/27 else 'ns'
            decision= '**' if p_value < 0.01/27 else '*'
            decision= '***' if p_value < 0.001/27 else '**'
            plt.text(0.99, 0.01, f'Pearson correlation coef.: {r_value:.2f}\n{decision} p = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=11,horizontalalignment='right')
        else:
            decision= '*' if p_value < 0.05/41 else 'ns'
            decision= '**' if p_value < 0.01/41 else '*'
            decision= '***' if p_value < 0.001/41 else '**'
            plt.text(0.99, 0.01, f'Pearson correlation coef.: {r_value:.2f}\n{decision} p = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=11,horizontalalignment='right')

        # Add legend
        #plt.legend(loc='lower right')

        # Show plot
        plt.grid(True,zorder=0)
        plt.savefig(f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/ada/figures/reg_{str(k)}.pdf')
        plt.show()



    ## INTER-SUBJECT ###################################################################################

    # Define the pattern to match text within parentheses
    pattern = r'\([^)]*\)'
    colors = ['#5ab6b1','#1e4747']
    stat = {}
    for t2,tissue_lst,tissuetype,lim,tisscolor in zip([t2jhu,t2ho],[wm_lst,gm_lst],["White Matter","Cortical Gray Matter"],lims,colors):
        k+=1
        data_x_std = []
        data_y_std = []
        data_x_mean = []
        data_y_mean = []

        for index in tissue_lst:
            data_x_mean.append(t2['sub-003']['ses-01'][index]["mean"])
            data_x_std.append(t2['sub-003']['ses-01'][index]["std"])

        for index in tissue_lst:
            data_y_mean.append(t2['sub-004']['ses-01'][index]["mean"])
            data_y_std.append(t2['sub-004']['ses-01'][index]["std"])

        data_mean = np.stack((data_x_mean, data_y_mean), axis=1)
        data_std = np.stack((data_x_std, data_y_std), axis=1)

        # Find rows containing NaN
        nan_rows = np.isnan(data_mean).any(axis=1)

        # Filter out rows containing NaN
        data_mean = data_mean[~nan_rows]
        data_std = data_std[~nan_rows]

        print(data_std.shape)
        print(data_mean.shape)

        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(data_mean[:,0],data_mean[:,1])
        print(r_value, p_value)
        line_x = np.array([0,500])
        line_y = slope * line_x + intercept

        # Create scatter plot with error bars
        plt.figure(figsize=(4,4))

        #for i in range(len(data_std)):
        #    plt.errorbar(data_mean[i,0], data_mean[i,1], xerr=data_std[i,0], yerr=data_std[i,1], fmt='o', markersize=10, color='#555555', ecolor='gray', capsize=5)

        plt.scatter(data_mean[:,0], data_mean[:,1], color=tisscolor, marker='o', s=50, edgecolor='none',alpha=1)

        # Plot regression line
        plt.plot(line_x, line_y, color=tisscolor, linestyle='--', label='Regression Line')
        plt.plot([0,500],[0,500],color = 'black',linestyle = '--', alpha= 0.2)

        # Add labels and title
        plt.xlabel('Subject 1 - T2 (ms)',fontsize=13)
        plt.ylabel('Subject 2 - T2 (ms)',fontsize=13)
        #plt.title('Inter-Subject')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        # Add Pearson correlation coefficient and p-value to the plot
        #plt.text(0.1, 0.9, f'Pearson correlation coefficient: {r_value:.2f}\nP-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=10)
        if tissuetype == "White Matter":
            decision= '*' if p_value < 0.05/27 else 'ns'
            decision= '**' if p_value < 0.01/27 else '*'
            decision= '***' if p_value < 0.001/27 else '**'
            plt.text(0.99, 0.01, f'Pearson correlation coef.: {r_value:.2f}\n{decision} p = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=11,horizontalalignment='right')
        else:
            decision= '*' if p_value < 0.05/41 else 'ns'
            decision= '**' if p_value < 0.01/41 else '*'
            decision= '***' if p_value < 0.001/41 else '**'
            plt.text(0.99, 0.01, f'Pearson correlation coef.: {r_value:.2f}\n{decision} p = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=11,horizontalalignment='right')

        # Add legend
        #plt.legend(loc='lower right')

        # Show plot
        plt.grid(True,zorder=0)
        plt.savefig(f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/ada/figures/reg_{str(k)}.pdf')
        plt.show()

    ## INTER-FIELD ###################################################################################
    k=1
    subs = ['sub-002','sub-003','sub-004','sub-005','sub-006','sub-007','sub-008','sub-009','sub-010','sub-011']
    for sub in subs:
        # Define the pattern to match text within parentheses
        pattern = r'\([^)]*\)'
        colors = ['#5ab6b1','#1e4747']
        stat = {}
        k+=1
        for t2,tissue_lst,tissuetype,lim,tisscolor in zip([t2jhu,t2ho],[wm_lst,gm_lst],["White Matter","Cortical Gray Matter"],lims,colors):

            data_x_std = []
            data_y_std = []
            data_x_mean = []
            data_y_mean = []

            for index in tissue_lst:
                data_x_mean.append(t2[sub]['ses-01'][index]["mean"])
                data_x_std.append(t2[sub]['ses-01'][index]["std"])

            for index in tissue_lst:
                data_y_mean.append(t2[sub]['ses-02'][index]["mean"])
                data_y_std.append(t2[sub]['ses-02'][index]["std"])

            data_mean = np.stack((data_x_mean, data_y_mean), axis=1)
            data_std = np.stack((data_x_std, data_y_std), axis=1)

            # Find rows containing NaN
            nan_rows = np.isnan(data_mean).any(axis=1)

            # Filter out rows containing NaN
            data_mean = data_mean[~nan_rows]
            data_std = data_std[~nan_rows]

            print(data_std.shape)
            print(data_mean.shape)

            # Calculate regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(data_mean[:,0],data_mean[:,1])
            print(r_value, p_value)
            line_x = np.array([0,500])
            line_y = slope * line_x + intercept

            # Create scatter plot with error bars
            plt.figure(figsize=(4,4))

            #for i in range(len(data_std)):
            #    plt.errorbar(data_mean[i,0], data_mean[i,1], xerr=data_std[i,0], yerr=data_std[i,1], fmt='o', markersize=10, color='#555555', ecolor='gray', capsize=5)

            plt.scatter(data_mean[:,0], data_mean[:,1], color=tisscolor, marker='o', s=50, edgecolor='none',alpha=1)

            # Plot regression line
            plt.plot(line_x, line_y, color=tisscolor, linestyle='--', label='Regression Line')
            plt.plot([0,500],[0,500],color = 'black',linestyle = '--', alpha= 0.2)

            # Add labels and title
            plt.xlabel('0.55 T - T2 (ms)',fontsize=13)
            plt.ylabel('1.5 T - T2 (ms)',fontsize=13)
            #plt.title('Inter-Subject')
            plt.xlim(lim)
            if tissuetype != "White Matter":
                plt.xlim([80,160])
                plt.ylim([80,160])
            else:
                plt.ylim([70,105])
                plt.xlim([70,105])
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)

            # Add Pearson correlation coefficient and p-value to the plot
            #plt.text(0.1, 0.9, f'Pearson correlation coefficient: {r_value:.2f}\nP-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=10)
            if tissuetype == "White Matter":
                decision= '*' if p_value < 0.05/27 else 'ns'
                decision= '**' if p_value < 0.01/27 else '*'
                decision= '***' if p_value < 0.001/27 else '**'
                plt.text(0.99, 0.01, f'Pearson correlation coef.: {r_value:.2f}\n{decision} p = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=11,horizontalalignment='right')
                tiss = 'wm'
            else:
                decision= '*' if p_value < 0.05/41 else 'ns'
                decision= '**' if p_value < 0.01/41 else '*'
                decision= '***' if p_value < 0.001/41 else '**'
                plt.text(0.99, 0.01, f'Pearson correlation coef.: {r_value:.2f}\n{decision} p = {p_value:.1e}', transform=plt.gca().transAxes, fontsize=11,horizontalalignment='right')
                tiss = 'gm'

            # Add legend
            #plt.legend(loc='lower right')

            # Show plot
            plt.grid(True,zorder=0)
            plt.savefig(f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/ada/figures/reg_field_{tiss}_sub-{str(k)}.pdf')
            plt.show()

def plot_violin(t2jhu,t2ho,wm_lst, gm_lst):
    # Number of subjects and colors for WM and GM
    subjects = 10
    colors = ['#5ab6b1', '#1e4747']  # Color for WM and GM
    colors_bis = ['#cee5e4', '#7c8f91']  # Color for WM and GM

    # List of T2 datasets
    t2s = [t2jhu, t2ho]  # Assuming t2jhu (WM) and t2ho (GM) are defined


    for field,ses in zip(['lf', 'hf'], ['ses-01','ses-02']): 
        print(field,ses)
        roi_data = {}
        # Collect mean values for each subject for both WM and GM
        for t2 in t2s:
            for sub in ['sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011']:
                data = []
                #print(f"Processing {sub} for T2...")
                for index in wm_lst:
                    mean_value = t2[sub][ses][index]["mean"]
                    if np.isnan(mean_value):
                        print(f"Warning: NaN found for {sub}, ROI {index}.")
                    else:
                        data.append(mean_value)
                roi_data.setdefault(sub, []).append(data)  # Store data for the subject

        # Organizing data for the violin plot
        data_for_plot = [[roi_data[sub][0] for sub in roi_data], [roi_data[sub][1] for sub in roi_data]]

        # Creating a violin plot
        plt.rc('axes', axisbelow=True)
        plt.figure(figsize=(9, 5))
        positions = np.arange(1, subjects + 1)
        plt.grid(linestyle='--',linewidth=0.5,zorder=-1) 
        plt.ylim([70,260])
        # Plot WM data
        parts_wm = plt.violinplot(data_for_plot[0], positions=positions - 0.15, showmeans=True, showmedians=True)
        # Plot GM data
        parts_gm = plt.violinplot(data_for_plot[1], positions=positions + 0.15, showmeans=True, showmedians=True)
        print(parts_wm.keys())
        # Formatting the plot
        plt.xticks(positions, roi_data.keys())  # Keep the original x-ticks
        #plt.yticks([70,80,90,100,110,120,130,140,150,160])
        plt.ylabel('T2 (ms)')

        # Set colors for WM and GM
        for pc in parts_wm['bodies']:
            pc.set_facecolor(colors_bis[0])  # WM color
            pc.set_alpha(0.9)

        for pc in parts_gm['bodies']:
            pc.set_facecolor(colors_bis[1])  # GM color
            pc.set_alpha(0.9)


        parts_gm['cmedians'].set_color(colors[1])  # GM mean color
        parts_wm['cmedians'].set_color(colors[0])  # WM mean color
        parts_gm['cbars'].set_color(colors[1])  # GM mean color
        parts_wm['cbars'].set_color(colors[0])  # WM mean color
        parts_gm['cmins'].set_color(colors[1])  # GM mean color
        parts_wm['cmins'].set_color(colors[0])  # WM mean color
        parts_gm['cmaxes'].set_color(colors[1])  # GM mean color
        parts_wm['cmaxes'].set_color(colors[0])  # WM mean color
        parts_gm['cmeans'].set_color(colors[1])  # GM mean color
        """ parts_gm['cmedians'].set_color('black')  # GM mean color
        parts_wm['cmedians'].set_color('black')  # WM mean color
        parts_gm['cbars'].set_color('black')  # GM mean color
        parts_wm['cbars'].set_color('black')  # WM mean color
        parts_gm['cmins'].set_color('black')  # GM mean color
        parts_wm['cmins'].set_color('black')  # WM mean color
        parts_gm['cmaxes'].set_color('black')  # GM mean color
        parts_wm['cmaxes'].set_color('black')  # WM mean color
        parts_gm['cmeans'].set_color('black')  # GM mean color """
        parts_gm['cmeans'].set_linestyle('--')  # Set mean line style to dashed
        parts_wm['cmeans'].set_color('red')  # GM mean color
        parts_gm['cmeans'].set_color('red')  # GM mean color
        a=1.5
        parts_wm['cmeans'].set_linestyle('--')  # Set mean line style to dashed
        parts_wm['cmeans'].set_linewidth(0.75)  # Set mean line style to dashed
        parts_gm['cmeans'].set_linewidth(0.75)  # Set mean line style to dashed
        parts_gm['cbars'].set_linewidth(a)  # Set mean line style to dashed
        parts_wm['cbars'].set_linewidth(a)  # Set mean line style to dashed
        parts_gm['cmedians'].set_linewidth(a)  # Set mean line style to dashed
        parts_wm['cmedians'].set_linewidth(a)  # Set mean line style to dashed
        parts_wm['cmaxes'].set_linewidth(a)  # Set mean line style to dashed
        parts_gm['cmaxes'].set_linewidth(a)  # Set mean line style to dashed
        parts_wm['cmins'].set_linewidth(a)  # Set mean line style to dashed
        parts_gm['cmins'].set_linewidth(a)  # Set mean line style to dashed

        plt.legend(['WM', 'GM'], loc='upper left', frameon=True)
        plt.savefig(f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/ada/figures/violin_{field}.pdf')
        plt.show()

def plot_t2_boxplot(t2jhu,t2ho,wm_lst, gm_lst):

    stat = {}
    for t2,tissuetype,tiss_list, lim,j in zip([t2jhu,t2ho],["White Matter","Cortical Gray Matter"],[wm_lst,gm_lst], [[70,120],[70,120]],[1,2]):
        pos = 0
        xtickpos = []

        # Create a 4-subplot layout
        fig, axs = plt.subplots(figsize=(4,4))
        if tissuetype == "White Matter":
            facecolors= ['#5ab6b1','#5ab6b1','#5ab6b1','#5ab6b1','#5ab6b1','#5ab6b1']
        else:
            facecolors = ['#1e4747','#1e4747','#1e4747','#1e4747','#1e4747','#1e4747']
        
        facecolors = ['#b6e6ff','#7fd0f7','#36a9e1','#1d71b8','#2fac66','#e94e1b']
        facecolors = ['#b6e6ff','#7fd0f7','#2fac66','#e94e1b','#e94e1b']

        if tissuetype not in stat.keys():
            stat[tissuetype] = {}
        #stat[tissuetype] = {}
        for (sub,ses),i in zip([['sub-003','ses-03'],['sub-003','ses-04'],['sub-003','ses-01'],['sub-004','ses-01'],['sub-005','ses-01']],range(5)):
            
            if sub not in stat[tissuetype].keys():
                stat[tissuetype][sub] = {}

            mean_t2 = []
            median_t2 = []
            pos += 1
            xtickpos.append(pos)

            for index in tiss_list:
                mean_t2.append(t2[sub][ses][index]["mean"])
                median_t2.append(t2[sub][ses][index]["median"])
            
            mean_t2 = [x for x in mean_t2 if not np.isnan(x)]

            stat[tissuetype][sub][ses] = mean_t2

            # Generate random x-values for scatter plot
            random_x = np.random.rand(len(mean_t2)) * 2 + 1  # Random x-values between 1 and 3

            #box = plt.boxplot(CoV_interrun, labels=['CV'])
            boxcol= '#555555'
            bplot = plt.boxplot(mean_t2, positions = [pos], showfliers=False,
                                    widths=0.6,patch_artist=True,boxprops=dict(edgecolor=boxcol, facecolor= (*mcolors.hex2color(facecolors[i]),0.9)),
                                    whiskerprops=dict(color=boxcol),medianprops=dict(color=boxcol),
                                    capprops=dict(color=boxcol), meanline=True,showmeans=True,meanprops=dict(color='red'))

            # Calculate the range for random x-values
            box_x = [item.get_xdata() for item in bplot['whiskers']]
            box_width = box_x[1][1] - box_x[0][1]  # Width of the boxplot
            center_x = (box_x[0][1] + box_x[1][1]) / 2  # Center x-coordinate of the boxplot
            x_min = pos - 0.5 / 4  # Minimum x-value for random_x
            x_max = pos + 0.5 / 4  # Maximum x-value for random_x

            # Generate random x-values for scatter plot
            random_x = np.random.uniform(x_min, x_max, len(mean_t2))

            # Plot scatter plot of all points
            plt.scatter(random_x, mean_t2, alpha=0.3, color=boxcol,edgecolors='none',zorder=2)


        #plt.ylabel('T2 (ms)')
        #plt.title(f'T2 of {tissuetype}')
        plt.grid('on',zorder=0)
        plt.ylim(lim)
        plt.ylabel("T2 (ms)",fontsize=13)
        plt.xticks([])
        plt.yticks(fontsize=13)
        """ plt.xticks(range(1,7), [
        "Subject 1 \nSession 1 \nRun 1 (LR)",
        "Subject 1 \nSession 1 \nRun 2 (LR)",
        "Subject 1 \nSession 1 \nRun 3 (LR)",
        "Subject 1 \nSession 1 \nRun 4 (HR)",
        "Subject 1 \nSession 2 \nRun 2 (HR)",
        "Subject 2 \nSession 1 \nRun 2 (HR)",
    ],fontsize =8) """

        """ legend_patches = [  Patch(facecolor = facecolors[0], edgecolor=boxcol, label='sub-002_ses-02_run-01'),
                            Patch(facecolor = facecolors[1], edgecolor=boxcol, label='sub-002_ses-02_run-02'),
                            Patch(facecolor = facecolors[2], edgecolor=boxcol, label='sub-002_ses-02_run-03'),
                            Patch(facecolor = facecolors[3], edgecolor=boxcol, label='sub-002_ses-01_interp'),
                            Patch(facecolor = facecolors[4], edgecolor=boxcol, label='sub-002_ses-02_interp'),
                            Patch(facecolor = facecolors[5], edgecolor=boxcol, label='sub-003_ses-01_interp')] """
        #plt.legend(handles=legend_patches,bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
        #plt.xticks([1,2,3], ['inter-run','inter-ses', 'inter-sub'])
        plt.savefig(f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/ada/figures/t2_{str(j)}_repeat.pdf')
        plt.show()

def compute_t2_per_tissue_feta(recon_dir='recon_1mm_t2map'):
    # NOT CLEAN PATH WRITTEN RAW IN FUNCTION
    t2feta_csv = []
    fit = 'gaussian'
    sim = '0'
    subs = ['sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011']
    for sub in subs:
        if sub == 'sub-003':
            sess = ['ses-01','ses-02','ses-03','ses-04']
        elif sub == 'sub-004':
            sess = ['ses-01','ses-02','ses-03']
        else:
            sess = ['ses-01','ses-02']

        for ses in sess:
            if ses == 'ses-02':
                scanner = 'sola'
                te= '115'
            else:
                scanner = 'freemax'
                te = '114'

            label_map_path = f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/derivatives/recon_1mm_feta/{sub}/{ses}/anat/{sub}_{ses}_te-{te}_recon_1mm_feta.nii.gz'
            t2_map_path = f'/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-004/derivatives/{recon_dir}/{sub}/{ses}/anat/{sub}_{ses}_recon_1mm_sim-{sim}_t2map_ada-{fit}.nii.gz'

            # Read the images
            if not os.path.exists(t2_map_path):
                print(t2_map_path, " does not exist." )
            else:
                label_map = sitk.ReadImage(label_map_path)
                t2_map = sitk.ReadImage(t2_map_path)

                # Erode the label map by 2 voxels for each label
                # Define the erosion radius (2 voxels in this case)
                erosion_radius = 1

                # Create an empty dictionary to store eroded label masks
                eroded_masks = {}

                # Perform erosion for each label (2=GM and 3=WM)
                for label in [2,3]:
                    # Create a binary image for the current label
                    label_binary = label_map == label
                    
                    # Erode the binary label mask by the defined radius
                    eroded_label = sitk.BinaryErode(label_binary, erosion_radius)
                    
                    # Store the eroded mask for later use
                    eroded_masks[label] = sitk.GetArrayFromImage(eroded_label)

                # Convert the T2 map to a numpy array
                t2_map_array = sitk.GetArrayFromImage(t2_map)

                # Initialize a dictionary to store results
                results = {}

                # Loop over the labels 1 and 2
                gts = [0,0,284,167,80,40]
                gts= [0,0,112,89]
                for label,tissue in zip([2,3],['gm','wm']):
                    gt=gts[label]
                    # Get the eroded mask for the current label
                    mask = eroded_masks[label]
                    
                    # Get the corresponding T2 values for the eroded label mask
                    t2_values = t2_map_array[mask == 1]
                    
                    # Compute the mean and standard deviation
                    #print(gt)
                    mape_t2 = np.nanmean((t2_values - gt ) / t2_values)
                    mean_t2 = np.mean(t2_values)
                    std_t2 = np.std(t2_values)
                    
                    # Store the results
                    results[label] = {'mean': mean_t2, 'std': std_t2,'mape': mape_t2}

                    t2feta_csv.append({ 'sub': sub,
                                    'ses': ses,
                                    'scanner': scanner,
                                    'roi': tissue,
                                    'mean': mean_t2,
                                    'std': std_t2,
                                    'nvoxel': t2_values.size})

    return pd.DataFrame(t2feta_csv)