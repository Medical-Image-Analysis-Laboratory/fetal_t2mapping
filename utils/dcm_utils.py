import os
import pydicom
import numpy as np
import subprocess
import pandas as pd
import json
import shlex
from pathlib import Path


### ****************************************************************************************************************
### ****************************************************************************************************************

class InvalidDicomError(ValueError):
    pass

def is_dcm_file(file_path):
    try:
        dcm_file = pydicom.dcmread(file_path)
        # If the above line executes without raising an exception, it's a DICOM file.
        return True
    except pydicom.errors.InvalidDicomError:
        return False

def check_path_type(path):
    # NOT IN USE
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"{path_obj} does not exist.")

    if path_obj.is_file():
        if is_dcm_file(path_obj):
            return "dcm_file"
        else:
            raise InvalidDicomError(f"{path_obj} is not a valid DICOM file.")
    elif path_obj.is_dir():
        # Check if the directory or any subdirectory contains at least one DICOM file
        dicom_files = [file for file in path_obj.rglob('*') if file.is_file() and is_dcm_file(file)]
        if dicom_files:
            return "dcm_dir"
        else:
            raise InvalidDicomError(f"{path_obj} and its subdirectories do not contain any valid DICOM files.")
    else:
        raise ValueError(f"{path_obj} is neither a file nor a directory.")

def read_dcm(dcm_path):  
    try:   
        dcm = pydicom.read_file(dcm_path)
        return dcm
    except:
        raise InvalidDicomError(f"{dcm_path} cannot read DICOM file.")

def get_dcm_files(directory):
    dicom_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if is_dcm_file(file_path):
                    dicom_files.append(file_path)
            except InvalidDicomError:
                pass  # Ignore files that are not valid DICOM
    return dicom_files

def build_dcm_entry(dcm_fl_path, dcm_structured=False):
    
    if not dcm_structured:
        # Reading the dicomfile
        dcm = read_dcm(dcm_fl_path)
        dcm_split = dcm_fl_path.split('/')
        dcm_entry = {'date':dcm_split[-3], 
                'dcm_parent': dcm_split[-2],
                'acquisition_time': dcm.AcquisitionDateTime,
                'dcm_reference': os.path.basename(dcm_fl_path),
                'description': dcm.SeriesDescription.strip(),
                'path': dcm_fl_path,
                'prj': None,
                'sub': None,
                'ses': None,
                'run': None}
    else:
        dcm = read_dcm(dcm_fl_path)
        dcm_split = dcm_fl_path.split('/')
        dcm_entry = {'date':dcm_split[-4], 
                'dcm_parent': dcm_split[-3],
                'acquisition_time': dcm.AcquisitionDateTime,
                'dcm_reference': os.path.basename(dcm_fl_path),
                'description': dcm.SeriesDescription.strip(),
                'path': dcm_fl_path,
                'prj': None,
                'sub': None,
                'ses': None,
                'run': None}
    
    return dcm_entry

def list_dcms_parent(dcms):
    print(f"========= DCMs parents and count =========")
    print(dcms.groupby('dcm_parent').size().reset_index(name='count'))
  
def get_dcms(dicom_path,dcm_structured=False):
    # Check if input file is dcm directory or file
    try:
        # Read dcm file(s) and get specifications and series description
        print(f"Getting DCMs file in : {dicom_path}")        
        dcms = []
        dcmfiles = get_dcm_files(dicom_path)

        for dcm_fl_path in dcmfiles:
            dcms.append(build_dcm_entry(dcm_fl_path,dcm_structured))

        if dcms:
            #return sorted(dcms, key=lambda x: x['dcm_reference'])
            return sorted(dcms, key=lambda x: x['acquisition_time'])
        else: 
            raise InvalidDicomError(f"{dicom_path} DICOM list is empty")
    
    except (FileNotFoundError, InvalidDicomError, ValueError) as e:
        print(e)

def get_bids_from_dcms_log(dcm_log_path,dcm_date,dcm_parent):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(dcm_log_path)
        # Apply filters
        entry = df[(df['date'] == dcm_date) & (df['dcm_parent'] == dcm_parent)]
        # Extract the 'Value' based on the filters
        if not entry.empty:
            return entry['prj'].values[0], entry['sub'].values[0], entry['ses'].values[0]
        else:
            raise InvalidDicomError(f"No matching records found in {os.path.basename(dcm_log_path)}. Check date is int and reference is char")
        
    except (FileNotFoundError, InvalidDicomError, ValueError) as e:
        print(e)

def input_bids(dcms,dcm_log_path,prj=None,sub=None,ses=None):
    print("===== BIDS ID (prj, sub, ses) reading from log_dcms.csv =====")
    dcms_df = pd.DataFrame(dcms) 
    for (dcm_date, dcm_parent), sub_dcm in dcms_df.groupby(["date","dcm_parent"]):
        condition = (dcms_df['date'] == dcm_date) & (dcms_df['dcm_parent'] == dcm_parent)

        to_bids = input(f"Do you want to assign bids ID to {dcm_date} - {dcm_parent} ? (y/n):")

        if to_bids == 'y':
            prj_id, sub_id, ses_id = get_bids_from_dcms_log(dcm_log_path, int(dcm_date), dcm_parent)
            
            # Assuming 'condition' is a boolean condition for filtering rows in dcms_df
            selected_rows = dcms_df.loc[condition].copy()
            # Assign values to the selected rows
            selected_rows['prj'] = prj_id
            selected_rows['sub'] = sub_id
            selected_rows['ses'] = ses_id
            selected_rows['run'] = selected_rows.groupby(['prj', 'sub', 'ses']).cumcount() + 1
            selected_rows['run'] = 'run-' + selected_rows['run'].astype(str).str.zfill(2)

            # Update the original DataFrame with the modified values
            dcms_df.loc[condition] = selected_rows

        else:
            dcms_df = dcms_df[~condition].reset_index(drop=True)
    
    print("===== DCMs and BIDS IDs =====")
    print(dcms_df)
    return dcms_df.to_dict()

def dcms2csv(csv_path, dcms):
    print("===== DCMs CSV writing =====")
    for (dcm_date, dcm_parent), sub_dcm in pd.DataFrame(dcms).groupby(["date","dcm_parent"]):

        # Write DataFrame to CSV file
        csv_flnm  = os.path.join(csv_path,f"{dcm_date}_{dcm_parent}.csv")
        sub_dcm.to_csv(csv_flnm, index=False)
        print(f"CSV file '{csv_flnm}' has been created.")

def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    
    while True:
        output = process.stdout.readline()
        output = output.strip().decode('utf-8')
        if output == '' and process.poll() is not None:
            break
            
    rc = process.poll()
    return rc

def mk_bids_dir(bids_dir,*dirs):
    current_path = bids_dir

    for directory in dirs:
        current_path = os.path.join(current_path, directory)
        if not os.path.exists(current_path):
            os.mkdir(current_path)

def json2dict(json_file_path):
    with open(json_file_path, 'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict

def dict2json(json_dict, json_file_path):
    # Save the JSON structure to the file
    with open(json_file_path, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)

def get_metabids(dcms, keys, bids_path,ref=False):
    # Get the existing keys in dcms
    existing_keys = list(dcms.columns)
    new_keys = [key for key in keys if key not in existing_keys]
    if new_keys:
        new_df = read_metabids(dcms, bids_path, new_keys,ref)
        dcms = pd.concat([dcms, new_df], axis=1)
    
    return dcms

def read_metabids(dcms,bids_dir, keys,ref=False):
    data_dict = {key: [] for key in keys}
    for (prj, sub, ses), sub_dcm in dcms.groupby(['prj','sub', 'ses']):
         i=0
         for index, dcm_entry in sub_dcm.iterrows():
            # Build filename
            bids_filenam = f"{sub}_{ses}_{dcm_entry['run']}_T2w"

            # Build filename
            i+=1
            if ref:
                bids_filenam = f"{sub}_{ses}_{dcm_entry['run']}_e{int(i)}"
            else:
                bids_filenam = f"{sub}_{ses}_{dcm_entry['run']}_T2w"

            json_file_path = os.path.join(bids_dir, prj, sub, ses,'anat', bids_filenam + '.json')
            json_dict = json2dict(json_file_path)

            # Loop through each JSON file
            for key in keys:
                try:
                    data_dict[key].append(json_dict.get(key, None))
                except:
                    raise InvalidDicomError(f"Key {key} not present in dicom json file.")

        # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(data_dict)

def get_orientation_dcm(img_orientation):

    int_array = np.round(np.array(img_orientation)).astype(int)

    if np.array_equal(int_array, [1, 0, 0, 0, 1, 0]):
        orient = "ax"
    elif np.array_equal(int_array, [1, 0, 0, 0, 0, -1]):
        orient = "cor"
    elif np.array_equal(int_array, [0, 1, 0, 0, 0, -1]):
        orient = "sag"
    else:
        orient = "custom"
    return orient

def run_dcm2niix(dcm, bids_dir,ref=False):
    """CONVERT_DCM2NII generates a nii.gz image and a .json file 
    from a dcm dataframe (available in csv)
    """
    print("==== run dcm2niix ====")
    # Iterate through stack group

    for (prj, sub, ses), sub_dcm in dcm.groupby(['prj','sub', 'ses']):
        mk_bids_dir(bids_dir,prj,sub,ses,'anat')
        i=0
        for index, dcm_entry in sub_dcm.iterrows():
            # Build filename
            i+=1
            if ref:
                bids_filenam = f"{sub}_{ses}_{dcm_entry['run']}_e{int(i)}"
                print(bids_filenam)
            else:
                bids_filenam = f"{sub}_{ses}_{dcm_entry['run']}_T2w"
            if not os.path.exists(os.path.join(bids_dir, prj, sub, ses,'anat',bids_filenam + '_T2w.nii.gz')):
                # Convert DCM to Nifti and generate Json
                cmd = [ 'dcm2niix', 
                        '-f', bids_filenam, 
                        '-o', os.path.join(bids_dir, prj, sub, ses,'anat'), 
                        '-s', 'y',
                        '-b', 'y', 
                        '-ba', 'y',
                        '-z', 'y', 
                        dcm_entry['path']]
                
                run_command(' '.join(cmd))

                if not ref:
                    # dcm2niix generate a json file: read to modify
                    json_dict = json2dict( os.path.join(bids_dir, prj, sub, ses,'anat', bids_filenam +'.json') )
                    #match = re.search(r'\d+', dcm_entry['run'])
                    #runid = int(match.group())
                    #json_dict = json2dict( os.path.join(bids_dir, prj, sub, ses,'anat', bids_filenam + '_e' + str(runid) +'.json') )
                    ds = read_dcm(dcm_entry['path'])
                    json_dict["Rows"] = ds.Rows
                    json_dict["Columns"] = ds.Columns
                    #json_dict["Slices"] = len(json_dict["SliceTiming"])
                    json_dict["PixelSpacingX"]= ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]
                    json_dict["PixelSpacingY"]= ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[1]
                    json_dict["ImageOrientationPatientSTR"] = get_orientation_dcm(json_dict["ImageOrientationPatientDICOM"])
                    #json_dict["FOV"] = ds.SharedFunctionalGroupsSequence[0][0x0021,0x10fe][0][0x0021,0x105e][:], # Private attribute: to update if necessary for other freemax/prisma acq
                
                    # Save the JSON dict to the JSON file
                    dict2json(json_dict, os.path.join(bids_dir, prj, sub, ses,'anat', bids_filenam + '.json') )
                    #dict2json(json_dict, os.path.join(bids_dir, prj, sub, ses,'anat', bids_filenam + '_e' + str(runid) +'.json') )

                print(f"{prj}_{sub}_{ses}_{dcm_entry['run']} BIDSified")

            else:
                print(f"{prj}_{sub}_{ses}_{dcm_entry['run']} already IN")

### ****************************************************************************************************************
## BELOW NOT IN USE -- OBSOLETE
def gen_config_json(df, out_path, sr_id = 0, do_anat_orient = True):
    """
    Generate a json file given a set of dicom dataset for one or several subjects. 
    The structure of the sesion entry is as follows:
        sub_id
            ses_id
                sr-id: 0= default
                session
                stacks
                custom_interfaces
                    do_anat_orientation: True

    Args:
        IN:
            df: panda dataframe with variables sub_id and ses_id at least
            out_path: path were the json file is saved (for BIDS, should be in code/)
        OUT:
            output: the json struct
    
    """
    
    output = {}
    
    # Group DataFrame by 'sub-id' and 'ses-id' to get the number of stacks
    stack_count = df.groupby(['sub_id', 'ses_id']).size().to_dict()
    
    # Iterate through stack group
    for sub, ses in stack_count:
        
        # Create a session entry
        session_entry = {
            "sr-id": 0,
            "session": ses,
            "stacks": list(range(1, stack_count[sub, ses]+1)),
            "custom_interfaces": {
                "do_anat_orientation": True
            }
        }
    
        # Create a key if it doesn't exist
        if sub not in output:
            output[sub] = []
    
        # Append the session entry to the sub key
        output[sub].append(session_entry)
    
    # Save the JSON structure to the file
    with open(out_path + '001_params.json', 'w') as json_file:
        json.dump(output, json_file,indent=4)
    
    print(f"JSON structure saved to: {out_path}")

    return output
### ****************************************************************************************************************
# ! Obsolete ! dcm2niix generate sequence json with all important parameters
def gen_seq_json(ds, json_file_path):
    seq_json = {#Standard Elements
                "StudyDate": ds.StudyDate,             
                "Modality": ds.Modality,
                "MagneticFieldStrength": ds.MagneticFieldStrength,
                "Manufacturer": ds.Manufacturer,
                "ManufacturersModelName": ds.ManufacturerModelName,
                "SoftwareVersions": ds.SoftwareVersions,
                "StationName": ds.StationName,
                "DeviceSerialNumber": ds.DeviceSerialNumber,
        
                "InstitutionName": ds.InstitutionName,
                "InstitutionAddress": ds.InstitutionAddress,
                
                "StudyDescription": ds.StudyDescription,
                "SeriesDescription": ds.SeriesDescription,
                "ProtocolName": ds.ProtocolName,

                "MRAcquisitionType": ds.MRAcquisitionType,
                "FOV": ds.SharedFunctionalGroupsSequence[0][0x0021,0x10fe][0][0x0021,0x105e][:], # Private attribute: to update if necessary for other freemax/prisma acq
                "ParallelAcquisitionTechnique": ds.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelAcquisitionTechnique,
                "PartialFourier": ds.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].PartialFourier,

                # Timing Parameters
                "Repetition Time": ds.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].RepetitionTime,
                "EchoTrainLength": ds.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].EchoTrainLength,
                "FlipAngle": ds.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].FlipAngle,
                "RFEChoTrainLength": ds.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].RFEchoTrainLength,
                "EchoNumbers": ds.PerFrameFunctionalGroupsSequence[0].MREchoSequence[0].EchoNumbers,
                "EffectiveEchoTime": ds.PerFrameFunctionalGroupsSequence[0].MREchoSequence[0].EffectiveEchoTime,

                # Geometry
                "BodyPartExamined": ds.BodyPartExamined,
                "NumberOfFrames": ds.NumberOfFrames,
                "Rows": str(ds.Rows),
                "Columns": str(ds.Columns),
                "SliceThickness": ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness,
                "SpacingBetweenSlices": ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices,
                "PixelSpacingX": ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0],
                "PixelSpacingY": ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[1]
                }

    # Save the JSON structure to the file
    with open(json_file_path, 'w') as json_file:
        json.dump([seq_json], json_file, indent=4)

    print(f"JSON structure saved to: {json_file_path}")
    return seq_json
### ****************************************************************************************************************

