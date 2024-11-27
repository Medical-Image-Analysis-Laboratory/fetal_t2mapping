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
        #run_resample_volume(metadata,high_res,bids_path,resamp_dirname,in_dirname,denoising=False)

        ## 2. Interpolate 1 high resolution 3D volume from 3 low resolution volumes
        #run_reconstruct_volume(metadata, bids_path,recon_dirname,resamp_dirname, denoising=True,orient_fix_type='sag')

        ## 3. Build high resolution masks
        #build_phantom_masks(metadata,bids_path, recon_dirname, mask_dirname, low=False)

        ## 3. Build Phantom Labels
        # PRJ-002 - SES-01 - MNCL2 PLATE 4 - LF - HEAD COIL
        # T2-1/2: [168,226,94],[168,221,64],
        # T2-12/13/14: [169,200,71],[168,161,65],[168,155,104]
        # T2-03 ---> T2-11
        #seeds = [[168,199,43],[168,168,38],[168,141,53],[168,128,80],
        #         [168,133,111],[169,155,133],[169,187,136],[169,213,123],[169,194,111]]
        
        # PRJ-002 - SES-01 - NICL2 PLATE 4 - LF - HEAD COIL
        # T2-1/2: [129,226,94],[129,222,64],
        # T2-12/13/14: ,[129,200,71],[128,161,65],[128,155,104]
        #seeds = [[128,199,43],[128,170,38],[128,141,53],[128,128,80],
        #         [129,133,111],[129,155,133],[129,187,136],[129,213,123],[129,194,111]]
        
        # PRJ-002 - SES-06 - MNCL2 PLATE 4 - HF - HEAD COIL
        # T2-01 ---> T2-14
        #seeds = [[155,221,102],[135,198,102],[134,167,102],[150,141,102],[178,129,102],[208,137,102],[227,160,102],
        #         [229,192,102],[212,218,102],[185,230,102],[188,207,102],[154,187,102],[175,152,102],[209,173,102]]
        
        # PRJ-002 - SES-06 - NICL2 PLATE 4 - HF - HEAD COIL
        # T2-01 ---> T2-14
        #seeds = [[155,221,143],[135,198,143],[133,167,143],[149,142,143],[178,130,143],[208,138,143],[227,161,143],
        #         [229,192,143],[212,218,142],[184,229,142],[188,207,142],[154,187,143],[175,152,143],[208,174,143]]

        # PRJ-003 - SES-01/SES-02 - MNCL2 PLATE 4 - LF - BODY COIL
        # T2-1/2: [141,207,105],[130,179,105],
        # T2-12/-13/-14: [152,175,105],[182,150,105],[207,181,105]
        seeds = [[139,149,105],[163,130,105],[194,129,105],[220,147,105],[229,176,105],
                 [221,206,105],[195,225,105],[165,226,105],[176,206,105]]

        # PRJ-003 - SES-01/SES-02 - NICL2 PLATE 4 - LF - BODY COIL
        # T2-1/2: [141,207,145],[130,179,145],
        # T2-12/13/14: ,[152,175,145],[182,150,145],[207,181,145]
        #seeds = [[139,149,145],[163,130,145],[194,129,145],[220,147,145],[229,176,145],
        #         [221,206,145],[195,225,145],[165,226,145],[176,206,145]]

        #PRJ-003 OTHER (not used)
        #seeds = [[135,161,103],[183,127,103],[230,163,103],[208,219,103],[185,206,103]] # ses-03
        #seeds = [[133,162,106],[181,128,106],[228,164,106],[206,220,103],[183,207,103]] #  ses-04
        #seeds = [[140,150,105],[195,132,105],[229,180,105],[193,228,105],[174,209,105]] # prj-002 ses-07
        build_phantom_labels_v2(metadata,bids_path, recon_dirname, phantom_labels_dirname,seeds,low=False)

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

# command: python run_qmri_reconstruction.py --path /home/mroulet/Documents/Data/qMRI/CHUV/freemax/ --csv 