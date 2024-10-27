# ReadMe

This repository details the pipeline to process data for the research project : *T2 mapping at 0.55T using ultra-fast spin-echo sequence*

It contains three main scripts and related utils:
 - `run_dcm2csv`:                 convert dicom to nifty files using BIDS
 - `run_qmri_reconstruction`:     generate all derivatives (reconstruction volumes and segmentation maps)
 - `run_t2mapping`:               derive t2 maps from reconstructed high resolution volumes

Author: Margaux Roulet

## 0.0 Installation

Code running on MIALTron in `gomar` conda environment using `python 3.11.6`

Libraries used:

```bash
numpy                     1.26.0
pandas                    2.1.3 
pydicom                   2.4.3 
simpleitk-simpleelastix   2.0.0rc2.dev910
scipy                     1.11.3 
scikit-image              0.22.0  
```

Notes: Make sure to have FSL and FreeSurfer installed as synthseg and flirt are used [1,2]

## 0.1 Arborescence

Your data qMRI directory should be srtuctured as follows:

```
qMRI/
├── projects/
|   ├── code/                   (bash script can be saved here as well as eventual config files for reconstruction)
│   ├── prj-001/
|   │   ...
│   ├── prj-00N/
|   │   ├── derivatives         (all processed volumes should be stored here)
|   │   |   |   ...
|   │   |   ├── recon
|   │   |   ├── labels
|   │   |   └── t2maps
|   │   |
|   │   ├── sub-001             (acquired data in usual BIDS format)
|   │   |   ...
|   │   └── sub-XXX
|   │ 
│   └── prj_list.csv            (list projects and description)
|
├── dicom/
│   ├── logs/
|   │   ├── log_dcms.csv                    (LOG file where metadata of each acquisition session are saved, see section 0.2)
|   │   └── YYYYMMDDHH_MMSS0000.csv         (LOG files for each acquisition session which details run info  and links run ID to dicom ID - generated by run_dcm2csv)
|   |    
|   │
│   └── YYYYMMDDHH/
|       ├── localizers/                     (directory containing all localizers dicom)
|       ├── failed_acquisitions/            (directory containing all failed dicom due to sar or other issues)
|       ├── MMSS0000_1/                     (directory containing all dicom to be processed of a specific prj_sub_ses. (Indent if you have several session per sub on same day))
|       └── MMSS0000_2/                     
|           ├── XXXXXXXX_1                  (dicom ID of a specific prj_sub_ses_run)
|           ├── ...
|           └── XXXXXXXX_N
|    
└── protocol_pdfs/                          (Pdfs of the acquisition protocols)
```

## 0.2. Acquisition Logs

A simplified standard operating procedure to keep track of data acquired is put into place using csv log files. 

After an acquisition, fill-up the acquisition metadata in the following log: `/dicom/logs/log_dcms.csv`. Specify the project, subjects, session IDs, as well as other general information such as date, dicom reference, scanner used, short description and notes (check the log for more information).

Then you are ready to go !

## 1. Convert DCM to Nifti

main script: `run_dcm2csv`
related utils: `dcm_utils`

```
usage: run_dcm2csv.py [-h] --path PATH (--check | --process)

DICOM File Parser

options:
  -h, --help   show this help message and exit
  --path PATH  Path to dicoms directory qMRI/dicom/YYYYMMDDHH/MMSS0000/
  --check      Check for DICOM files prior Process
  --process    Process DICOM files
```

Some manual dicom selection is required before processing. You will need to identify the localizers and the failed acquisition volumes and put them in a separate folder. You can use the print on the terminal showing dicom ID and dicom description and then manually move them to another sub folders (see arboresence 0.1). In addition, if you acquired several run during that acquisition, create one subfolder for each session.

After running the script, BIDS data will be generated and nifti files as well as json files listing all acquisition parameters will be generated in the correct BIDS directory. In addition, a metadata csv log YYYYMMDDHH_MMSS0000.csv is generated including run metadata information (such as sub, ses,run ID, TE, orientation, TR,... etc). This metadata csv is subsequently used to run  `run_qmri_reconstruction` and  `t2_mapping` scripts.

*Important Note*: When exporting the dicom from the scanner software, check the tick box `use DICOM file system` and `Enchanced` mode ! Normally, the script should handle different types of dicom format, but bugs might persists if formatting differs.

## 2. Run qMRI reconstruction

main script: `run_qmri_reconstruction`
related utils: `qmri_utils`, `metadata_utils`

```
usage: run_qmri_reconstruction.py [-h] --path PATH --csv CSV [CSV ...] (--in_vivo | --in_vitro) (--lf | --hf)

QMRI Reconstruction Parser

options:
  -h, --help           show this help message and exit  
  --path PATH          Path to general directory ../qMRI/
  --csv CSV [CSV ...]  Can be either:
                         (1) Name of one or more metadata CSV log files (e.g., YYYYMMDDHH_MMSS0000.csv)
                         (2) Name of project to process all CSV log files in that project (e.g., prj-00X)
  --in_vivo            Process in vivo data
  --in_vitro           Process NIST Phantom data
  --lf                 Process low-field 0.55 T data
  --hf                 Process high-field 1.5 T data
```

Once Nifty images of the acquired volumes are generated, everything is set to generate some derivatives for both `in_vivo` or `in_vitro` data. 

### In-vivo adult brain

The `run_qmri_reconstruction` processing pipeline generates the following `in_vivo` derivatives:

```
derivatives/
├── recon_1mm/              (1mm3 high resolution volume for each TE in case of no motion using trilinear interpolation)
├── recon_1mm_bet/          (brain extracted HR volume)
├── recon_1mm_feta/         (feta segmentation map of HR volume)
├── recon_1mm_ho/           (cortical grey matter segmentation map of HR volume)
├── recon_1mm_jhu/          (white matter segmentation map of HR volume)
├── recon_1mm_mask/         (mask of HR volume)
├── recon_1mm_mni152/       (elastic registration of the mni152 to subject)
├── recon_1mm_synthseg/     (synthseg segmentation map)
└── resamp_1mm/             (1mm3 resampled volumes for each TE and for orientation)
```

ho = Harvard-Oxford cortical structural atlases [3]
jhu = JHU DTI-based white-matter atlases - ICBM-DTI-81 white-matter labels atlas [4]
HR = high-resolution

Processing pipeline steps:
 - 1. Resampling: each acquired volume is resampled to 1x1x1mm3 resolution.
 - 2. Reconstruction: Then for each TE, the three resampled orientation volumes are registered together and trilinear interpolation is used to reconstruct a 1x1x1mm3 resolution volume. Reconstruction is done for each TEs, and all final reconstructed volumes are registered with respect to the first TE. After interpolation and registration, Total-variation denoising is performed.
 - 3. Optional registration of 1.5 T reconstructed volumes to 0.55 T: For data analysis purpose, all reconstructed volumes at 1.5 T are registered to the 0.55 T volumes.
 - 4. Synthseg segmentation: FreeSurfer Synthseg is used to generate synthseg segmentation map. A bash script is written and run to process all HR reconstructed volumes and generate resepctive segmentation maps [2].
 - 5. Mask: mask of the brain is then generated using synthseg segmentation using a binary condition.
 - 6. Brain extraction: using HR reconstructed volumes and masks, brain is extracted. The brain extracted HR volumes (bet) will be used to run the T2 mapping.
 - 7. Feta segmentation: Synthseg segmentation is converted to feta segmentation maps
 - 8. HO and JHU Segmentation: Using FSL FLIRT and the MNI152 standard brain template and the JHU and HO brain atlases, specific regions within the white matter and gray matter are segmented [1]. The MNI152 brain template is aligned to the individual’s brain scan using elastic registration, which creates a transformation file (.omat file). This transformation file is then applied to the JHU and HO atlases to produce segmented maps of white matter (from JHU) and gray matter (from HO) that match the individual’s brain scan.

### In-vitro NIST Phantom

The `run_qmri_reconstruction` processing pipeline generates the following `in_vitro` derivatives:

```
derivatives/
├── mask/                   (low resolution mask of each orientation volume)
├── recon_1mm/              (1mm3 high resolution volume for each TE in case of no motion using trilinear interpolation)
├── recon_1mm_label/        (HR segmentation map of specific region-of-interest)
├── recon_1mm_mask/         (mask of HR volume)
└── resamp_1mm/             (1mm3 resampled volumes for each TE and for orientation)
```

Processing pipeline steps:
    
 - 0. Low resolution mask: optional step for SRR (not in use for now -> decomment if you want to generate LR mask)
 - 1. Resampling: same as in_vivo above
 - 2. Reconstruction: same as in_vivo above
 - 3. Mask: Mask of the HR reconstructed phantom volume is generated from reconstructed phantom volume.
 - 4. ROI Labels: ROI label maps

 *Important Note:* You need to provide the seeds for your region of interest *manually* to build then be able to generate the label maps!! Seeds for prj-003 are provided raw in the script for now.

## 3. T2 mapping

main script: `t2_mapping`
related utils: `t2map_utils`, `metadata_utils`

```
usage: run_t2mapping.py [-h] --path PATH --csv CSV [CSV ...] (--in_vivo | --in_vitro | --in_vitro_fast) (--gaussian | --gaussian_rician | --rician) (--lf | --hf) --sim SIM [--TEs TES [TES ...]] [--no_prior] [--norm]

T2 Mapping Parser

options:
  -h, --help           show this help message and exit
  --path PATH          Path to general directory ../qMRI/
  --csv CSV [CSV ...]  Can be either:
                         (1) Name of one or more metadata CSV log files (e.g., YYYYMMDDHH_MMSS0000.csv)
                         (2) Name of project to process all CSV log files in that project (e.g., prj-00X)
  --in_vivo            Process in vivo data
  --in_vitro           Process NIST Phantom data and generate full map
  --in_vitro_fast      Process NIST Phantom data only at ROI (useful for ablation studies)
  --gaussian           T2 fit under gaussian noise assumption
  --gaussian_rician    T2 fit under gaussian-rician noise assumption
  --rician             T2 fit under rician noise assumption
  --lf                 Process low-field 0.55 T data
  --hf                 Process high-field 1.5 T data
  --sim SIM            T2 fitting ID (can be a description or a simple int)
  --TEs TES [TES ...]  List of TEs to fit
                          default low-field: [114,202,299]
                          default high-field: [115,202,299]
  --no_prior           If set, will not restrict M0 bounds (default: False)
  --norm               If set, will normalize T2w data (default: False)
```

The `run_t2mapping` processing pipeline generates the `recon_1mm_t2map` derivatives. 

### Fitted parameters maps (t2, k, sigma)

Along with the t2 maps it also outputs the maps of the other fitted parameters, such as M0 (called k_map in); sigma (when fitting under rician or gaussian_rician noise assumption); and the residuals.

### Residual maps

The residual map is calculated by comparing the measured T2w images at each echo time (TE) with the T2w images generated using the fitted model parameters. Specifically:

 - 1. **Compute Residuals**: At each echo time, calculate the residuals, which represent the difference between the measured T2w images and the model-generated T2w images.
 - 2. **Determine Maximum Residual**: For each pixel, find the maximum residual value across all echo times. This value represents the highest deviation between the measured data and the fitted model at any echo time.

The final residual map is created by taking these maximum residual values for each pixel, providing a map of the maximum error across all echo times.

### In vitro NIST Phantom

When fitting T2 maps for the NIST phantom, an additional `.csv` file is generated to provide summary statistics for the fitted parameters. This file includes the mean and standard deviation values for each parameter: T2, k and sigma maps for each region of interest. Spectrometer ground truth is also available.

**Note:** The k map represents the M0 map, but it’s not labeled as M0 because it doesn’t purely measure the M0 signal. Artifacts such as residual noise and other sequence-related effects from the HASTE sequence influence the k parameter. These effects introduce variability that impacts the accuracy of M0 as measured by k. For this reason, we refer to it as the k map rather than a true M0 map.

### Convergence analysis

Convergence plots of the fitting process are output in the `ada/convergence_analysis/` directory.

## 4. Data analysis

Final data analysis is done in the following Jupyter Notebooks: `20240924_ada_qmri_jmri_invitro.ipynb` and `20240924_ada_qmri_jmri.ipynb`
related utils: `ada_utils`

Note: Other notebooks are available for your informations, they are named by date of creation followed by a short keyword describing analysis.

## Contact
Name: Margaux Roulet
Email: margaux.roulet@chuv.ch
GitHub: margaux-roulet

## References

[1] M. Jenkinson et S. Smith, « A global optimisation method for robust affine registration of brain images », Med. Image Anal., vol. 5, no 2, p. 143‑156, juin 2001, doi: 10.1016/s1361-8415(01)00036-6.
[2] SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias
Medical Image Analysis
[3] R. S. Desikan et al., « An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest », NeuroImage, vol. 31, no 3, p. 968‑980, juill. 2006
[4] « MRI Atlas of Human White Matter », AJNR Am. J. Neuroradiol., vol. 27, no 6, p. 1384‑1385, juin 2006.