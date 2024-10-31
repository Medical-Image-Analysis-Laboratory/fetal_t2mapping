import pandas as pd
import os

# DERIVATIVES DIRECTORY NAME
in_dirname = 'anat'
resamp_dirname ='resamp_1mm'
resamp_dirname ='resamp_1mm'
recon_dirname = 'recon_1mm'
mask_dirname = 'recon_1mm_mask'
synthseg_dirname = 'recon_1mm_synthseg'
bet_dirname = 'recon_1mm_bet'
feta_dirname = 'recon_1mm_feta'
jhu_dirname = 'recon_1mm_jhu'
ho_dirname = 'recon_1mm_ho'
mni_dirname = 'recon_1mm_mni152'
phantom_labels_dirname = 'recon_1mm_label'
t2map_dirname = recon_dirname + '_t2map'

def prj_004(low_field):
    """ Bypass the command line --csv and list all the csvs of prj-004 instead
    """
    # IN VIVO DATA ********************************************
    if low_field:
        # PRJ-004 - 0.55T - IN VIVO
        csvs = ['2024083017_17510000.csv', # Emeline sub-002_ses-01
                '2024090320_55420000.csv', # Misha sub-003_ses-01
                '2024090618_37050000.csv', # Alyssa sub-004_ses-01
                '2024090811_14320000.csv', # Rizhong sub-005_ses-01
                '2024091017_53530000_1.csv', # Misha sub-003_ses-03
                '2024091017_53530000_2.csv', # Misha sub-003_ses-04
                '2024091020_45220000.csv', # Yasser sub-006_ses-01
                '2024091320_23400000.csv', # Alyssa sub-004_ses-03
                '2024091321_22550000.csv', # Michael sub-007_ses-01
                '2024091322_27490000.csv', # Félice sub-008_ses-01
                '2024092720_10110000.csv', # Nataliia sub-009_ses-01
                '2024092719_10310000.csv', # Andreea sub-010_ses-01
                '2024102120_48480000.csv'] # Jaime sub-011_ses-01
    else:
        # PRJ-004 - 1.5 T - IN VIVO
        csvs = ['2024083019_26300000.csv', # Emeline sub-002_ses-02
                '2024090322_28560000.csv', # Misha sub-003_ses-02
                '2024090619_26370000.csv', # Alyssa sub-004_ses-02
                '2024090812_21470000.csv', # Rizhong sub-005_ses-02
                '2024091021_57280000.csv', # Yasser sub-006_ses-02
                '2024091319_13240000.csv', # Michael sub-007_ses-02
                '2024091318_13560000.csv', # Felice sub-008_ses-02
                '2024092721_25410000.csv', # Nataliia sub-009_ses-02
                '2024102616_18560000.csv', # Andreea sub-010_ses-02
                '2024102122_28450000.csv' # Jaime sub-011_ses-02
                ]

    return csvs

def prj_003(low_field):
    """ Bypass the command line --csv and list all the csvs of prj-003 instead
    prj-003 : in vitro NIST Phantom analysis at low-field using abdominal coil M
    """
    if low_field:
        # PRJ-003  - 0.55 T - IN VITRO - ABD COIL M
        csvs = ['20240806_30540000_1.csv', # sub-001_ses-01
                '20240806_30540000_2.csv', # sub-001_ses-02
                '2024080811_19360000_1.csv', # sub-001_ses-03
                '2024080811_19360000_2.csv' # sub-001_ses-04 # dont do rician fit on this, for some reason explode ram
                ]
    else:
        print("Error: no data to process yet at 1.5 T.")
        exit(1)    
    return csvs

def prj_002(low_field):
    """ Bypass the command line --csv and list all the csvs of prj-002 instead
    Note: only data used for paper submissio are processed
    prj-002 : in vitro NIST Phantom analysis at low-field using head coil
    """
    if low_field:
        # PRJ-002  - 0.55 T - IN VITRO - Headcoil (Prescan ON)
        csvs = ['20240527_095111_2.csv', # sub-001_ses-02
                #'20240527_095111_4.csv', # sub-001_ses-04
                #'20240530_092341_2.csv' # sub-001_ses-08
                ]
    else:
        # PRJ-002  - 1.5 T - IN VITRO - Headcoil (Prescan ON)
        csvs = ['20240609_50140000_2.csv'] # sub-001_ses-06
    
    return csvs

def csv2df(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    return df

def set_metadata(csv_path, csvs,low_field):
    """ Import metadata as a dataframe from log csv files
    """

    # Set metadata dataframe given csvs listed
    if csvs[0] == 'prj-004':
        print("***************************************************************")
        print("PRJ-004 - In vivo adult brain data acquired using the head coil")
        print("***************************************************************")
        csvs = prj_004(low_field)
    elif csvs[0] == 'prj-003':
        print("************************************************************************")
        print("PRJ-003 - In vitro NIST Phantom data acquired using the abdominal coil M")
        print("************************************************************************")
        csvs = prj_003(low_field)
    elif csvs[0] == 'prj-002':
        print("******************************************************************")
        print("PRJ-002 - In vitro NIST Phantom data acquired using the head coil.") 
        print("Notes: only data selected for paper submission are processed.")
        print("******************************************************************")
        csvs = prj_002(low_field)
    elif ".csv" not in csvs[0].lower():
        print(f"Error: {csvs} is not a valid metadata log file nor a valid project to process (only prj-002, prj-003 and prj-004 metadata can be processed all at once.)")
        exit(1)

    # Convert csvs lsit to pd dataframe
    for i,csv in enumerate(csvs):
        if i==0:
            metadata = csv2df(os.path.join(csv_path,csv))
        else:
            metadata = pd.concat([metadata,csv2df(os.path.join(csv_path,csv))])
    
    return metadata
