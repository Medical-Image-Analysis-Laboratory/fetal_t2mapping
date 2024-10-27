import os
from utils.dcm_utils import *
from utils.qmri_utils import *
import pandas as pd
import argparse

def check_dicom(dicom_path):

        print("Please, before proceeding with process_dicom, put all localizers and failed acquisition dicoms in a separated folder. Please find below description of the acquired sequence")

        # Is DCM structured in sub folders:
        
        file_lst = sorted(os.listdir(dicom_path))

        for flnm in file_lst:
                dcm_fl_path = os.path.join(dicom_path,flnm)
                dcm = read_dcm(dcm_fl_path)
                print(flnm, dcm.SeriesDescription)
                #print(flnm, dcm.PerFrameFunctionalGroupsSequence[0].MREchoSequence[0].EffectiveEchoTime)
                #print(flnm, dcm.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].RepetitionTime)

def process_dicom(dicom_path,bids_path,csv_path,dcm_log_path):
        
        # 1. Build DATAFRAME WITH ALL RELEVANT INFO FOR qMRI
        dcm_structured = False
        dcms = get_dcms(dicom_path,dcm_structured)
        list_dcms_parent(pd.DataFrame(dcms))
        dcms = input_bids(dcms, dcm_log_path)

        run_dcm2niix(pd.DataFrame(dcms), bids_path,ref=False)
        keys = ["CoilString",
                "EchoTime", 
                "RepetitionTime", 
                "SliceThickness", 
                "FlipAngle", 
                "EchoTrainLength", 
                "PixelBandwidth",
                "PixelSpacingX", 
                "PixelSpacingY",
                "ImageOrientationPatientDICOM",
                "ImageOrientationPatientSTR"]
        dcms = get_metabids(pd.DataFrame(dcms),keys,bids_path,ref=False)
        dcms2csv(csv_path,dcms)

def mese_not_in_use():
     # Orientation is coded as follows (at least for this sub-002, ses-01)
    # [ 1 0 0 0 1 0 ] --> TrueAx
    # [ 1 0 0 0 0 -1 ] --> TrueCor
    # [ 0 1 0 0 0 -1 ] --> TrueSag


    # FOR MESE IMPORT
    """ # Directory containing the files
    directory = '/home/mroulet/Documents/Data/qMRI/CHUV/freemax/projects/prj-002/sub-001/ses-11/anat/'

    # Pattern to match files with eX in their names
    pattern = re.compile(r'(sub-\d+_ses-\d+_run-\d+)_e\d+(\.json|\.nii\.gz)')

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Construct the new filename
            new_filename = match.group(1) + match.group(2)
            
            # Create the full paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} -> {new_file}')
    """

def parse_arguments():
    parser = argparse.ArgumentParser(description="DICOM File Parser")
    
    # Required path argument
    parser.add_argument('--path', type=str, required=True, help="Path to dicoms directory qMRI/dicom/YYYYMMDDHH/MMSS0000/")

    # Optional mutually exclusive arguments: --check or --process
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help="Check for DICOM files prior Process")
    group.add_argument('--process', action='store_true', help="Process DICOM files")

    return parser.parse_args()

def main():
    # Parse arguments
    # path = /home/mroulet/Documents/Data/qMRI/CHUV/freemax
    args = parse_arguments()
    dicom_path = args.path
    path = args.path.split('dicom')[0]
    csv_path = os.path.join(path,'dicom/logs/')
    dcm_log_path = os.path.join(path,'dicom/logs/log_dcms.csv')
    bids_path = os.path.join(path,'projects/')

    # Check if the provided path exists
    if not os.path.exists(args.path):
        print(f"Error: The specified path does not exist: {args.path}")
        exit(1)

    # Handle the chosen action (check or process)
    if args.check:
        check_dicom(dicom_path)
    elif args.process:
        process_dicom(dicom_path,bids_path,csv_path,dcm_log_path)

if __name__ == "__main__":
    main()
