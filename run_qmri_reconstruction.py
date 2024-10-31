from utils.qmri_utils import *
from utils.metadata_utils import *
import argparse

def process_qmri(bids_path,metadata,in_vivo,low_field):
    # IN VIVO #######################################################3
    if in_vivo:
        ## 2. Resample volume
        high_res = 1.
        run_resample_volume(metadata,high_res,bids_path,resamp_dirname,in_dirname,denoising=False)

        ## 3. Interpolate 1 high resolution 3D volume from 3 low resolution volumes + Registration each TE reconstructed volume + Denoising
        run_reconstruct_volume(metadata, bids_path,recon_dirname,resamp_dirname, denoising=True,orient_fix_type='ax')

        ## 3bis. Register high field to low field
        # note always process 0.55 prior 1.5 otherwise sip this step
        if not low_field:
            register_high_to_low_field(metadata,bids_path,recon_dirname)

        ## 4. Build Labels
        synthseg_sh_filepath = gen_synthseg_sh(metadata,bids_path,recon_dirname,synthseg_dirname)
        run_synthseg_sh(synthseg_sh_filepath)

        ## 5. Build Masks
        build_mask_from_labels(metadata,bids_path, synthseg_dirname, mask_dirname)

        ## 5bis. Brain Extraction        
        extract_brain(metadata, bids_path, recon_dirname, mask_dirname, bet_dirname)

        ## 6. Build FeTA labels
        convert_synthseg_to_feta(metadata,bids_path, synthseg_dirname, feta_dirname)

        ## 7. Build JHU and HO labels (atlas-based analysis)
        build_jhu_ho_labels(metadata,bids_path,bet_dirname,mni_dirname,jhu_dirname,ho_dirname,low_field)

    ######### PHANTOM ############################################################
    else:
        ## 0. Build low resolution masks (for SRR NOT REQUIRED for now)
        mask_dirname = 'mask'
        #build_phantom_masks(metadata,bids_path, 'anat', mask_dirname, low=True)

        ## 1. Resample volume
        high_res = 1.
        run_resample_volume(metadata,high_res,bids_path,resamp_dirname,in_dirname,denoising=False)

        ## 2. Interpolate 1 high resolution 3D volume from 3 low resolution volumes
        run_reconstruct_volume(metadata, bids_path,recon_dirname,resamp_dirname, denoising=True,orient_fix_type='sag')

        ## 3. Build high resolution masks
        build_phantom_masks(metadata,bids_path, recon_dirname, mask_dirname, low=False)

        ## 3. Build Phantom Labels
        # PRJ-003
        seeds = [[140,149,105],[194,129,105],[230,176,105],[195,224,105],[176,206,105]] # ses-01 / ses-02
        #seeds = [[135,161,103],[183,127,103],[230,163,103],[208,219,103],[185,206,103]] # ses-03
        #seeds = [[133,162,106],[181,128,106],[228,164,106],[206,220,103],[183,207,103]] #  ses-04
        #seeds = [[140,150,105],[195,132,105],[229,180,105],[193,228,105],[174,209,105]] # prj-002 ses-07
        #build_phantom_labels_v2(metadata,bids_path, recon_dirname, phantom_labels_dirname,seeds,low=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="QMRI Reconstruction Parser",formatter_class=argparse.RawTextHelpFormatter)
    
    # Required path argument
    parser.add_argument('--path', type=str, required=True, help="Path to general directory ../qMRI/")
    parser.add_argument('--csv', type=str, nargs='+', required=True, 
                        help=(  "Can be either:\n"
                                "  (1) Name of one or more metadata CSV log files (e.g., YYYYMMDDHH_MMSS0000.csv)\n"
                                "  (2) Name of project to process all CSV log files in that project (e.g., prj-00X)"))
    
    # Optional mutually exclusive arguments: --check or --process
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--in_vivo', action='store_true', help="Process in vivo data")
    group.add_argument('--in_vitro', action='store_true', help="Process NIST Phantom data")
    
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--lf', action='store_true', help="Process low-field 0.55 T data")
    group2.add_argument('--hf', action='store_true', help="Process high-field 1.5 T data")

    return parser.parse_args()

def main():
    # Parse arguments
    # path = /home/mroulet/Documents/Data/qMRI/CHUV/freemax
    args = parse_arguments()

    # Check if the provided path exists
    if not os.path.exists(args.path):
        print(f"Error: The specified path does not exist: {args.path}")
        exit(1)

    bids_path = os.path.join(args.path,'projects/')
    csv_path = os.path.join(args.path,'dicom/logs/')
    
    # Handle the chosen action (vivo/vitro lf/hf)
    if args.lf:
        low_field = True
    elif args.hf:
        low_field = False
    if args.in_vivo:
        in_vivo = True
    elif args.in_vitro:
        in_vivo = False

    metadata = set_metadata(csv_path,args.csv,low_field)

    # Launch processing pipeline
    process_qmri(bids_path,metadata,in_vivo,low_field)

if __name__ == "__main__":
    main()