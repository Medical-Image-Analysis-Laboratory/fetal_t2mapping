import os
import numpy as np
import SimpleITK as sitk
import subprocess
import glob
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
from scipy.interpolate import RegularGridInterpolator
from skimage import restoration
import re
import pandas as pd
from .dcm_utils import mk_bids_dir

def get_img_path(bids_path, acq, type:str="anat"):

    if type == "anat":
        img_dirs = [acq["prj"], acq["sub"], acq["ses"], "anat"]
        img_flnm =  "_".join([ acq["sub"], acq["ses"], acq["run"]+ "_T2w.nii.gz" ])
    elif "t2map" in type:
        img_dirs = [acq["prj"], "derivatives",  type, acq["sub"], acq["ses"], "anat"]
        img_flnm = "_".join([acq["sub"], acq["ses"], type +".nii.gz"])
    elif "recon" in type:
        img_dirs = [acq["prj"], "derivatives",  type, acq["sub"], acq["ses"], "anat"]
        if acq["CoilString"] == "Simulation":
            img_flnm = "_".join([acq["sub"], acq["ses"], f"t2-{int(acq['T2']):3}", f"te-{int(acq['EchoTime']):3}", type +".nii.gz"])
        else:
            img_flnm = "_".join([acq["sub"], acq["ses"], f"te-{int(acq['EchoTime']*1000):3}", type +".nii.gz"])
    else: 
        img_dirs = [acq["prj"], "derivatives",  type, acq["sub"], acq["ses"], "anat"]
        img_flnm = "_".join([acq["sub"], acq["ses"], acq["run"], "T2w", type +".nii.gz"])
    
    mk_bids_dir(bids_path,*img_dirs)
    img_path = os.path.join(bids_path, *img_dirs)
    return os.path.join(img_path,img_flnm)

def run_resample_volume(metadata,high_res,bids_path,resamp_dirname,in_dirname='anat', denoising=False):
    print(" ===== Resampling =====")
    # Sanity check missing
    # - check key present in metadata
    # - high_res > low_res

    # group by orientation
    #for (orient), sub_metadata in metadata.groupby(["ImageOrientationPatientSTR"]):
    for _, acq in metadata.iterrows():
        print(acq['run'])

        #match = re.search(r'run-(\d{2,4})', acq['run'])
        #if int(match.group(1))> 2606 and int(match.group(1))< 2620:
            
        img_path = get_img_path(bids_path,acq,in_dirname)
        img_low = sitk.ReadImage(img_path)
        img_high = resample_volume(img_low,[high_res,high_res,high_res])

        if denoising:
            img_high = run_denoising(img_high)

        # save image
        resamp_path = get_img_path(bids_path,acq, resamp_dirname)
        sitk.WriteImage(img_high, resamp_path)
        print(f"Image saved in : {resamp_path}")

def resample_volume(volume: sitk.Image,
                    new_spacing: list = [1, 1, 1],
                    interpolator = sitk.sitkLinear):
    """Resamples image to a new voxel spacing.
       Results in an image with different dimensions.

    Args:
        volume (sitk.Image): Image to resample
        interpolator (_type_, optional): Interpolator for the voxel sampling. Defaults to sitk.sitkLinear.
        new_spacing (list, optional): New voxel size. Defaults to [1, 1, 1].

    Returns:
        sitk.Image: Image with new voxel spacing
    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())

def reconstruct_vol_trilinear(imgs: dict, fixed_type="ax"):
    
    # register imgs
    registered_imgs = {}
    fixed_img = imgs[fixed_type]
    moving_types = [string for string in ["ax", "cor", "sag"] if string != fixed_type]
    registered_imgs[fixed_type] = fixed_img
    for moving_type in moving_types:
        print(f"Registration: fixed img - {fixed_type}, moving img - {moving_type}")
        registered_imgs[moving_type] = registration_elastix(fixed_img, imgs[moving_type])

    print("Interpolation: type: trilinear")
    
    # Get spacing, origin, and direction from the fixed image
    reference_spacing = fixed_img.GetSpacing()
    reference_origin = fixed_img.GetOrigin()
    reference_direction = fixed_img.GetDirection()

    # Create a grid for the fixed image volume (axial, coronal, sagittal)
    fixed_img_array = sitk.GetArrayFromImage(fixed_img)
    img_shape = fixed_img_array.shape
    z = np.linspace(reference_origin[2], reference_origin[2] + reference_spacing[2] * (img_shape[0] - 1), img_shape[0])
    y = np.linspace(reference_origin[1], reference_origin[1] + reference_spacing[1] * (img_shape[1] - 1), img_shape[1])
    x = np.linspace(reference_origin[0], reference_origin[0] + reference_spacing[0] * (img_shape[2] - 1), img_shape[2])

    # Generate the grid points corresponding to the fixed image shape
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    points = np.array([Z.ravel(), Y.ravel(), X.ravel()]).T

    # Stack all the registered images into a single array
    stacked_imgs = {}
    for img_type in registered_imgs:
        # Extract the registered image and stack it for interpolation
        stacked_imgs[img_type] = sitk.GetArrayFromImage(registered_imgs[img_type])

    # Perform interpolation for each volume (axial, coronal, sagittal)
    interpolated_arrays = []
    for img_type in stacked_imgs:
        interpolator = RegularGridInterpolator((z, y, x), stacked_imgs[img_type], method='linear')
        interpolated_values = interpolator(points)
        interpolated_arrays.append(interpolated_values)

    # Average the three interpolated arrays element-wise
    averaged_interpolated_values = np.mean(interpolated_arrays, axis=0)

    # Reshape the averaged result to match the fixed image shape
    final_volume = averaged_interpolated_values.reshape(fixed_img_array.shape)

    # Convert the final averaged and reshaped volume back to a SimpleITK image
    final_img = sitk.GetImageFromArray(final_volume)
    final_img.SetSpacing(reference_spacing)
    final_img.SetOrigin(reference_origin)
    final_img.SetDirection(reference_direction)

    return final_img

def reconstruct_vol_avg(imgs: dict, fixed_type = "ax"):
    
    # register imgs
    registered_imgs = {}
    fixed_img = imgs[fixed_type]
    moving_types = [string for string in ["ax","cor","sag"] if string != fixed_type]
    registered_imgs[fixed_type] = imgs[fixed_type]
    for moving_type in moving_types:
        print(f"Registration: fixed img - {fixed_type}, moving img - {moving_type}")
        registered_imgs[moving_type] = registration_elastix(imgs[fixed_type],imgs[moving_type])

    print("Interpolation: type: avg - balanced")
    # Combine volumes using weighted average (1/3, 1/3, 1/3)
    final_volume = sum( (1./3.) * sitk.GetArrayFromImage(img) for img in registered_imgs.values())
    final_img = sitk.GetImageFromArray(final_volume)
    final_img.SetSpacing(registered_imgs[fixed_type].GetSpacing())
    final_img.SetOrigin(registered_imgs[fixed_type].GetOrigin())
    final_img.SetDirection(registered_imgs[fixed_type].GetDirection())    

    return final_img

def registration_elastix(fixed_image, moving_image):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elastixImageFilter.Execute()
    return elastixImageFilter.GetResultImage()

def registration_itk(fixed_image, moving_image):


    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_image, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method = sitk.ImageRegistrationMethod()

    # Set Metric As Correlation
    registration_method.SetMetricAsCorrelation()
    #registration_method.SetMetricAsMeanSquares()
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=24)
    
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1) # 0.1
    
    # mask
    moving_mask = build_mask(moving_image)
    fixed_mask = build_mask(fixed_image)
    registration_method.SetMetricMovingMask(moving_mask)
    registration_method.SetMetricFixedMask(fixed_mask)
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1, 
        numberOfIterations=100, 
        minStep=1e-6, 
        gradientMagnitudeTolerance=1e-6
    )
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    #registration_method.SetOptimizerAsGradientDescent(
    #    learningRate=1.0, numberOfIterations=100
    #)
    
    final_transform = sitk.Euler3DTransform(initial_transform)
    registration_method.SetInitialTransform(final_transform)
    
    #registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    #registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    
    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
    )

    registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    return  registered_image

def build_mask(img):
    
    threshold = 1.0 # for sufficient signal
        
    # Get information from the original image
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
        
    # Now, you can set a new voxel array for the 'new_image' (assuming 'new_voxel_array' is boolean)
    mask_array = np.zeros_like(sitk.GetArrayFromImage(img))
    mask_array = mask_array.astype('uint8')
    img_array = sitk.GetArrayFromImage(img)

    # Iterate over slices
    for i in range(mask_array.shape[2]):
        # Read the binary image
        BW = img_array[:,:,i] > threshold
        BW = binary_fill_holes(BW)
        BW = binary_dilation(BW, structure=np.ones((5, 5)))
        BW = binary_erosion(BW, structure=np.ones((5, 5)))
    
        # Store the result in the 4D array
        mask_array[:, :, i] = BW
    mask = sitk.GetImageFromArray(mask_array)
    mask.SetSpacing(spacing)
    mask.SetOrigin(origin)
    mask.SetDirection(direction)
    
    return mask

def run_biasfield_correction(metadata, bids_path, n4_dirname,mask_dirname):
    # to clean with structure bids etc....

    for (prj,sub, ses), sub_metadata in metadata.groupby(["prj","sub","ses"]):
        mask_dir = os.path.join(bids_path, prj, 'derivatives',mask_dirname,sub,ses,'anat')
        input_dir = os.path.join(bids_path, prj,sub,ses,'anat')
        pre_dir = os.path.join(bids_path,prj,'derivatives',n4_dirname,sub,ses,'anat')
        mk_bids_dir(pre_dir)

        for _, acq in sub_metadata.iterrows():
            mask_img = sitk.ReadImage(mask_dir + f'/{sub}_{ses}_{acq.run}_T2w_mask.nii.gz')
            low_img = sitk.ReadImage(input_dir + f'/{sub}_{ses}_{acq.run}_T2w.nii.gz')
            low_img = sitk.Cast(low_img, sitk.sitkFloat32)
            
            if acq["ImageOrientationPatientSTR"] == 'cor':
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetBiasFieldFullWidthAtHalfMaximum(0.25)
                corrected_img = corrector.Execute(low_img, mask_img)
                corrected_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(corrected_img) * 0.25)
                corrected_img.CopyInformation(low_img)

            else:
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetBiasFieldFullWidthAtHalfMaximum(0.5)
                corrected_img = corrector.Execute(low_img, mask_img)
                corrected_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(corrected_img) * 0.25)
                corrected_img.CopyInformation(low_img)

            corrected_img.CopyInformation(low_img)

            """ # Apply histogram matching
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(40)
            matcher.SetNumberOfMatchPoints(8)
            matcher.ThresholdAtMeanIntensityOn()
            corrected_img = matcher.Execute(corrected_img, low_img) """
            
            # save image
            pre_path = get_img_path(bids_path, acq, n4_dirname)
            sitk.WriteImage(corrected_img,pre_path)
            print(f"Image saved in : {pre_path}")

def run_biasfield_correction2(metadata, bids_path, n4_dirname,mask_dirname):
    # to clean with structure bids etc....

    for (prj,sub, ses), sub_metadata in metadata.groupby(["prj","sub","ses"]):

        mask_dir = os.path.join(bids_path, prj, 'derivatives',mask_dirname,sub,ses,'anat')
        input_dir = os.path.join(bids_path, prj,sub,ses,'anat')
        pre_dir = os.path.join(bids_path,prj,'derivatives',n4_dirname,sub,ses,'anat')
        mk_bids_dir(pre_dir)

        for _, acq in sub_metadata.iterrows():
            if acq["EchoTime"] == 0.255:

                mask_img = sitk.ReadImage(mask_dir + f'/{sub}_{ses}_{acq.run}_T2w_mask.nii.gz')
                low_img = sitk.ReadImage(input_dir + f'/{sub}_{ses}_{acq.run}_T2w.nii.gz')
                low_img = sitk.Cast(low_img, sitk.sitkFloat32)
                corrected_img = low_img
                
                if acq["ImageOrientationPatientSTR"] == 'cor':
                    print("N4 Bias Field Correction: COR")
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.25)
                    corrected_img = corrector.Execute(corrected_img, mask_img)
                    log_bias_field_cor = corrector.GetLogBiasFieldAsImage(low_img)
                elif acq["ImageOrientationPatientSTR"] == 'ax':
                    print("N4 Bias Field Correction: AX")
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.5)
                    corrected_img = corrector.Execute(corrected_img, mask_img)
                    log_bias_field_ax = corrector.GetLogBiasFieldAsImage(low_img)
                else:
                    print("N4 Bias Field Correction: SAG")
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.5)
                    corrected_img = corrector.Execute(corrected_img, mask_img)
                    log_bias_field_sag = corrector.GetLogBiasFieldAsImage(low_img)

        for _, acq in sub_metadata.iterrows():
            mask_img = sitk.ReadImage(mask_dir + f'/{sub}_{ses}_{acq.run}_T2w_mask.nii.gz')
            low_img = sitk.ReadImage(input_dir + f'/{sub}_{ses}_{acq.run}_T2w.nii.gz')
            low_img = sitk.Cast(low_img, sitk.sitkFloat32)
            
            if acq["ImageOrientationPatientSTR"] == 'cor':
                corrected_img = low_img / sitk.Exp(log_bias_field_cor)
            elif acq["ImageOrientationPatientSTR"] == 'ax':
                corrected_img = low_img / sitk.Exp(log_bias_field_ax)
            else:
                corrected_img = low_img / sitk.Exp(log_bias_field_sag)

            corrected_img.CopyInformation(low_img)

            """ # Apply histogram matching
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(100)
            matcher.SetNumberOfMatchPoints(7)
            matcher.ThresholdAtMeanIntensityOn()
            corrected_img = matcher.Execute(corrected_img, low_img) """

            # save image
            pre_path = get_img_path(bids_path, acq, n4_dirname)
            sitk.WriteImage(corrected_img,pre_path)
            print(f"Image saved in : {pre_path}")

def run_reconstruct_volume(metadata, bids_path, recon_dirname,resamp_dirname,denoising=False,orient_fix_type='ax'):
    """ADD DESCRIPTION"""
    # iterate sub_ses
    for (prj,sub,ses), sub_ses_metadata in metadata.groupby(["prj","sub","ses"]):
        i=0
        # iterate echo time
        for (echotime), sub_metadata in sub_ses_metadata.groupby(["EchoTime"]):
            # iterate orientation
            for _, acq in sub_metadata.iterrows():

                print(acq['EchoTime'])
                imgs = {acq["ImageOrientationPatientSTR"]: sitk.ReadImage(get_img_path(bids_path, acq,resamp_dirname)) for _, acq in sub_metadata.iterrows()}
                print(imgs.keys())

                if len(imgs) == 3:
                    print(f"===== Registration + Reconstruction: TE {int(acq['EchoTime']*1000):3} ms =====")
                    #recon_img = reconstruct_vol_avg(imgs, fixed_type=orient_fix_type)
                    recon_img = reconstruct_vol_trilinear(imgs, fixed_type=orient_fix_type)
                    # register wrt to first echotime
                    if i==0:
                        print(echotime,sub,ses)
                        fixed_recon = recon_img
                    else:
                        recon_img = registration_elastix(fixed_recon,recon_img)
                    i+=1
                        
                    if denoising:
                        recon_img = run_denoising(recon_img)

                    # save image
                    recon_path = get_img_path(bids_path, acq, recon_dirname)
                    sitk.WriteImage(recon_img,recon_path)
                    print(f"Image saved in : {recon_path}")

def run_denoising(recon_img):
    print("############# DENOISING #############")
    data = sitk.GetArrayFromImage(recon_img)
    # Preprocess each 2D slice of the 3D image
    denoised_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        denoised_data[i] = restoration.denoise_tv_chambolle(data[i])

    # Convert denoised data back to SimpleITK image
    denoised_img = sitk.GetImageFromArray(denoised_data)
    denoised_img.CopyInformation(recon_img)

    return denoised_img

def check_all_orientations_available(imgs):
    dict = {"ax": None, "cor": None, "sag": None}

    for img in imgs:
        if img.orient == "trueax":
            dict["ax"] = img
        elif img.orient == "trueCor":
            dict["cor"] = img
        elif img.orient == "trueSag":
            dict["sag"] = img

    for orient, img in dict.items():
        if img is None:
            raise ValueError(f"No object with {orient} orientation found")

    return dict

def gen_synthseg_sh(metadata, bids_path, recon_dirname, labels_dirname):    
    # export and source freesurfer software
    fsl_sh = """#!/bin/bash
    export FREESURFER_HOME=/usr/local/freesurfer/7.4.1/
    source /usr/local/freesurfer/7.4.1/SetUpFreeSurfer.sh
    source /usr/local/freesurfer/7.4.1/FreeSurferEnv.sh
    """
    # Use parallel processing: careful 2 cores already use 80% of the memory
    fsl_sh += f"parallel -j 2 :::"

    # build line for each dir within metadata dataframe
    for (prj,sub, ses), _ in metadata.groupby(["prj","sub","ses"]):

        input_dir = os.path.join(bids_path,prj,'derivatives',recon_dirname, sub,ses,'anat')
        mk_bids_dir(bids_path,prj,'derivatives', labels_dirname, sub, ses,'anat')
        output_dir = os.path.join(bids_path, prj, 'derivatives', labels_dirname, sub, ses, 'anat')

        # cmd line
        # bug found in Billot Synthseg fast ... does not seem to work at 1.5 T for some reason
        # even after registering the brain volumes to the low-field data...
        #cmd=f"\"mri_synthseg --i {input_dir} --o {output_dir} --fast --threads 4 --cpu\""
        cmd=f"\"mri_synthseg --i {input_dir} --o {output_dir} --robust --threads 4 --cpu\""
        fsl_sh += ' ' + cmd

    # Write the script to a file
    mk_bids_dir(bids_path,'code')
    script_filename = os.path.join(bids_path, 'code','mri_synthseg.sh')
    
    with open(script_filename, 'w') as script_file:
        script_file.write(fsl_sh)

    print(f"Shell script '{script_filename}' has been created.")

    return script_filename

def run_synthseg_sh(synthseg_sh_path):
    # Run Synthseg using shell script generated
    try:    
        # Run the shell script
        subprocess.run(['bash', synthseg_sh_path], check=True)
        print(f"Shell script '{synthseg_sh_path}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing shell script '{synthseg_sh_path}': {e}")

def resamp_labels_masks(root_path,grouped_df,
                        labels_dir, labels_resamp_dir, masks_dir,
                        anat_file_id ="_T2w",
                        labels_file_id="_T2w_synthseg",
                        labels_resamp_file_id="_T2w_synthseg_resamp"):

    for (sub, ses), sub_df in grouped_df:
 
        for index, row in sub_df.iterrows():

            T2w = sitk.ReadImage(os.path.join(root_path, sub,ses,'anat',row['anat_filenam']))
            labels = sitk.ReadImage(os.path.join(root_path, labels_dir, sub, ses,'anat', row['anat_filenam'].replace(anat_file_id,labels_file_id)))
            
            # Get the spacing and size of the input image
            input_spacing = T2w.GetSpacing()
            input_size = T2w.GetSize()
            
            # Create a resampling filter
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(input_size)
            resampler.SetOutputSpacing(input_spacing)
            resampler.SetOutputOrigin(labels.GetOrigin())
            resampler.SetOutputDirection(labels.GetDirection())
            resampler.SetTransform(sitk.Transform())
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            
            # Resample
            labels_resamp = resampler.Execute(labels)
            
            # Save label map as nii.gz file
            sitk.WriteImage(labels_resamp, os.path.join(root_path, labels_resamp_dir, sub, ses,'anat',row['anat_filenam'].replace(anat_file_id,labels_resamp_file_id)))

            # Build mask and Save as nii.gz file
            mask = sitk.BinaryThreshold(labels_resamp, lowerThreshold=1, upperThreshold=float(sitk.GetArrayFromImage(labels_resamp).max()), insideValue=1, outsideValue=0)
            sitk.WriteImage(mask, os.path.join(root_path, masks_dir, sub, ses,'anat',row['anat_filenam'].replace(anat_file_id,"_mask")))

def downsample_labels(metadata, bids_path, high_labels_dirname, low_labels_dirname):
    for (prj, sub, ses), metadata_sub in metadata.groupby(["prj", "sub", "ses"]):
        for _, acq in metadata_sub.iterrows():
            high_labels_path = get_img_path(bids_path, acq, high_labels_dirname)
            high_labels = sitk.ReadImage(high_labels_path)
            
            img_path = get_img_path(bids_path, acq)
            low_labels_path = get_img_path(bids_path, acq, low_labels_dirname)
            img_low = sitk.ReadImage(img_path)
            
            # Ensure the high_labels and img_low are in the same physical space
            if not sitk.GetArrayFromImage(high_labels).shape == sitk.GetArrayFromImage(img_low).shape:
                # Define resampling parameters to match low-resolution image
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(img_low)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor interpolation for label maps
                resampler.SetOutputSpacing(img_low.GetSpacing())
                resampler.SetOutputOrigin(img_low.GetOrigin())
                resampler.SetOutputDirection(img_low.GetDirection())
                resampler.SetSize(img_low.GetSize())
                
                # Resample label map to low-resolution space
                labels = resampler.Execute(high_labels)
            else:
                labels = high_labels

            sitk.WriteImage(labels, low_labels_path)

def downsample_labels_old(metadata, bids_path, high_labels_dirname, low_labels_dirname):

    for (prj,sub, ses), metadata_sub in metadata.groupby(["prj","sub","ses"]):

        for _, acq in metadata_sub.iterrows():
            
            high_labels_path = get_img_path(bids_path, acq, high_labels_dirname)
            high_labels = sitk.ReadImage(high_labels_path)
            
            img_path = get_img_path(bids_path,acq)
            low_labels_path = get_img_path(bids_path, acq, low_labels_dirname)
            print(img_path)
            print(high_labels_path)
            print(low_labels_path)
            img_low = sitk.ReadImage(img_path)
            
            ## RESAMPLE
            #labels = sitk.Resample(high_labels, img_low)

            # Define resampling parameters to match low-resolution image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img_low)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor interpolation for label maps
            # Resample label map to low-resolution space
            labels= resampler.Execute(high_labels)

            # register labels to img_low
            # PROVIDE HERE 
            registration_method = sitk.ImageRegistrationMethod


            #labels = sitk.BinaryDilate(labels,2)
            #labels = sitk.BinaryMorphologicalClosing(labels, 1)
            #labels = sitk.BinaryMorphologicalOpening(labels,1)
            sitk.WriteImage(labels, low_labels_path)

def downsample_masks(metadata, bids_path, high_masks_dirname,low_masks_dirname):

    for (prj,sub, ses), metadata_sub in metadata.groupby(["prj","sub","ses"]):
        for _, acq in metadata_sub.iterrows():

            high_mask_path = get_img_path(bids_path, acq, high_masks_dirname)
            recon_dir = os.path.join(bids_path, prj,'derivatives',high_masks_dirname,sub,ses,'anat')
            recon_flnms = glob.glob(f"{recon_dir}/*.nii.gz")
            high_mask_path = recon_flnms[0]
            high_mask = sitk.ReadImage(high_mask_path)
            
            img_path = get_img_path(bids_path,acq)
            low_mask_path = get_img_path(bids_path, acq, low_masks_dirname).replace('masks.nii','mask.nii')
            print(img_path)
            print(high_mask_path)
            print(low_mask_path)
            img_low = sitk.ReadImage(img_path)
            mask = sitk.Resample(high_mask, img_low)
            mask = sitk.BinaryDilate(mask,2)
            mask = sitk.BinaryMorphologicalClosing(mask, 1)
            mask = sitk.BinaryMorphologicalOpening(mask,1)
            sitk.WriteImage(mask, low_mask_path)

def build_phantom_masks(metadata,bids_path, recon_dir, masks_dir,low=False):

    for (prj,sub, ses), _ in metadata.groupby(["prj","sub","ses"]):
        
        if low:
           input_dir = os.path.join(bids_path, prj,sub,ses,'anat')
        else:
            input_dir = os.path.join(bids_path, prj,'derivatives',recon_dir,sub,ses,'anat')

        mk_bids_dir(bids_path, prj,'derivatives',masks_dir,sub,ses,'anat')
        output_dir = os.path.join(bids_path, prj,'derivatives',masks_dir,sub,ses,'anat')

        recon_img_flnms = glob.glob(f"{input_dir}/*.nii.gz")

        for recon_img_flnm in recon_img_flnms:
            recon_img = sitk.ReadImage(recon_img_flnm)

            mask = sitk.BinaryThreshold(recon_img, lowerThreshold=100, upperThreshold=float(sitk.GetArrayFromImage(recon_img).max()), insideValue=1, outsideValue=0)
            # Create a BinaryFillholeImageFilter
            fill_holes_filter = sitk.BinaryFillholeImageFilter()
            mask = fill_holes_filter.Execute(mask)

            # Step 3: Define the structuring element
            #structuring_element = sitk.BinaryBall(10)
            # Step 4: Perform closing operation
            mask = sitk.BinaryMorphologicalClosing(mask, [15,15,15])
            mask = sitk.BinaryDilate(mask, 10)
            #mask = sitk.BinaryMorphologicalOpening(mask,[15,15,15])

            if low:
                sitk.WriteImage(mask, os.path.join(output_dir,os.path.basename(recon_img_flnm).replace("T2w","T2w_mask")))
            else:
                sitk.WriteImage(mask, os.path.join(output_dir,os.path.basename(recon_img_flnm).replace(recon_dir,recon_dir +"_mask")))

def build_phantom_labels(metadata,bids_path, recon_dir, labels_dir,seeds,low=False):

    for (prj,sub, ses), _ in metadata.groupby(["prj","sub","ses"]):
        
        if low:
            input_dir = os.path.join(bids_path, prj,sub,ses,'anat')
        else:
            input_dir = os.path.join(bids_path, prj,'derivatives',recon_dir,sub,ses,'anat')

        mk_bids_dir(bids_path, prj,'derivatives',labels_dir,sub,ses,'anat')
        output_dir = os.path.join(bids_path, prj,'derivatives',labels_dir,sub,ses,'anat')

        img_flnms = glob.glob(f"{input_dir}/*.nii.gz")
        print(input_dir)

        for img_flnm in img_flnms:
            print(img_flnm)
            img = sitk.ReadImage(img_flnm)

            # Initialize label map
            output_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
            output_img.CopyInformation(img)  # Copy metadata from the original image

            labels = range(1, len(seeds) + 1)  # Unique label for each seed

            # Perform binary dilation and intensity threshold for each seed
            for seed, label in zip(seeds, labels):
                # Create a binary image representing the seed
                seed_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
                seed_img.CopyInformation(img)
                seed_img[seed] = 1

                # Perform binary dilation
                region_grown_image = sitk.BinaryDilate(seed_img, 4)

                # Set the label value to the segmented region
                region_grown_image *= label

                # Convert output_img to array
                region_grown_array = sitk.GetArrayFromImage(region_grown_image)

                # Create mask array
                mask_array = np.zeros_like(region_grown_array, dtype=np.uint8)
                x, y, z = seed
                mask_array[z-3:z+4, y-3:y+4, x-3:x+4] = 1  # Set seed and neighboring voxels to 1

                # Apply mask to output array
                region_grown_array *= mask_array

                # Convert back to SimpleITK image
                region_grown_image = sitk.GetImageFromArray(region_grown_array)
                region_grown_image.CopyInformation(img)

                # Create a BinaryFillholeImageFilter
                fill_holes_filter = sitk.BinaryFillholeImageFilter()
                region_grown_image = fill_holes_filter.Execute(region_grown_image)

                # Accumulate the segmented region into the output image
                output_img = sitk.Maximum(output_img, region_grown_image)

            sitk.WriteImage(output_img, os.path.join(output_dir,os.path.basename(img_flnm).replace("T2w","T2w_labels")))

def build_phantom_labels_ref(metadata, bids_path, recon_dir, labels_dir, seeds, low=True):
    for (prj, sub, ses), _ in metadata.groupby(["prj", "sub", "ses"]):
        if low:
            input_dir = os.path.join(bids_path, prj, sub, ses, 'anat')
        else:
            input_dir = os.path.join(bids_path, prj, 'derivatives', recon_dir, sub, ses, 'anat')

        mk_bids_dir(bids_path, prj, 'derivatives', labels_dir, sub, ses, 'anat')
        output_dir = os.path.join(bids_path, prj, 'derivatives', labels_dir, sub, ses, 'anat')

        img_flnms = glob.glob(f"{input_dir}/*.nii.gz")
        print(input_dir)

        for img_flnm in img_flnms:
            print(img_flnm)
            img = sitk.ReadImage(img_flnm)

            # Ensure it's a single slice image
            assert img.GetSize()[2] == 1, "The image is not a single-slice image."

            # Initialize label map
            output_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
            output_img.CopyInformation(img)
            print("Output image shape:", sitk.GetArrayFromImage(output_img).shape)

            labels = range(1, len(seeds) + 1)  # Unique label for each seed

            # List to store individual segmented regions
            segmented_regions = []

            # Perform binary dilation and intensity threshold for each seed
            for seed, label in zip(seeds, labels):
                print(f"Processing seed: {seed} with label: {label}")

                # Check if the seed coordinates are within the image bounds
                if all(0 <= s < d for s, d in zip(seed, img.GetSize())):
                    print("Seed is within bounds.")
                    
                    # Create a binary image representing the seed
                    seed_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
                    print("Seed image shape:", sitk.GetArrayFromImage(seed_img).shape)
                    
                    seed_img.CopyInformation(img)
                    
                    try:
                        seed_img[seed] = 1
                        print(f"Seed {seed} set successfully.")
                    except IndexError as e:
                        print(f"Error setting seed {seed}: {e}")
                        continue

                    # Perform region growing with variance threshold
                    region_grown_image = sitk.Image(img.GetSize(), sitk.sitkUInt8)
                    region_grown_image.CopyInformation(img)
                    region_grown_image[seed] = 1
                    queue = [seed]
                    visited = set([tuple(seed)])  # Convert seed to tuple

                    while queue:
                        current = queue.pop(0)
                        neighbors = get_neighbors(current, img)
                        
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                # Check variance condition
                                if np.abs(img.GetPixel(current) - img.GetPixel(neighbor)) < 100:
                                    region_grown_image[neighbor] = 1
                                    queue.append(neighbor)
                                    visited.add(neighbor)

                    # Set the label value to the segmented region
                    region_grown_image *= label

                    # Store the segmented region for erosion
                    segmented_regions.append(region_grown_image)

                else:
                    print(f"Seed {seed} is out of bounds for image size {img.GetSize()}")

            # Merge all segmented regions into the final output image
            for region in segmented_regions:
                output_img = sitk.Maximum(output_img, region)

            # Perform erosion on the merged output image
            erode_filter = sitk.BinaryErodeImageFilter()
            erode_filter.SetKernelType(sitk.sitkBall)
            erode_filter.SetKernelRadius([2, 2, 0])  # 2-voxel erosion in 2D
            
            eroded_output_img = erode_filter.Execute(output_img)

            # Write the eroded output image
            output_filename = os.path.join(output_dir, os.path.basename(img_flnm).replace("T2w", "T2w_labels_eroded"))
            print(output_filename)
            sitk.WriteImage(output_img, output_filename)

def get_neighbors(point, image):
    """ Get 8-connected neighbors of a point in a 2D image. """
    x, y, z = point
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny, nz = x + dx, y + dy, z
            if 0 <= nx < image.GetWidth() and 0 <= ny < image.GetHeight():
                neighbors.append((nx, ny, nz))
    return neighbors

def build_phantom_labels_ref2(metadata, bids_path, recon_dir, labels_dir, seeds, low=True):
    for (prj, sub, ses), _ in metadata.groupby(["prj", "sub", "ses"]):
        if low:
            input_dir = os.path.join(bids_path, prj, sub, ses, 'anat')
        else:
            input_dir = os.path.join(bids_path, prj, 'derivatives', recon_dir, sub, ses, 'anat')

        mk_bids_dir(bids_path, prj, 'derivatives', labels_dir, sub, ses, 'anat')
        output_dir = os.path.join(bids_path, prj, 'derivatives', labels_dir, sub, ses, 'anat')

        img_flnms = glob.glob(f"{input_dir}/*.nii.gz")
        print(input_dir)

        for img_flnm in img_flnms:
            print(img_flnm)
            img = sitk.ReadImage(img_flnm)

            # Ensure it's a single slice image
            assert img.GetSize()[2] == 1, "The image is not a single-slice image."

            # Initialize label map
            output_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
            output_img.CopyInformation(img)
            print("Output image shape:", sitk.GetArrayFromImage(output_img).shape)

            labels = range(1, len(seeds) + 1)  # Unique label for each seed

            # Perform binary dilation and intensity threshold for each seed
            for seed, label in zip(seeds, labels):
                print(f"Processing seed: {seed} with label: {label}")

                # Check if the seed coordinates are within the image bounds
                if all(0 <= s < d for s, d in zip(seed, img.GetSize())):
                    print("Seed is within bounds.")
                    
                    # Create a binary image representing the seed
                    seed_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
                    print("Seed image shape:", sitk.GetArrayFromImage(seed_img).shape)
                    
                    seed_img.CopyInformation(img)
                    
                    try:
                        seed_img[seed] = 1
                        print(f"Seed {seed} set successfully.")
                    except IndexError as e:
                        print(f"Error setting seed {seed}: {e}")
                        continue

                    # Perform binary dilation
                    region_grown_image = sitk.BinaryDilate(seed_img, 4)

                    # Set the label value to the segmented region
                    region_grown_image *= label

                    # Convert output_img to array
                    region_grown_array = sitk.GetArrayFromImage(region_grown_image)

                    # Convert back to SimpleITK image
                    region_grown_image = sitk.GetImageFromArray(region_grown_array)
                    region_grown_image.CopyInformation(img)

                    # Create a BinaryFillholeImageFilter
                    fill_holes_filter = sitk.BinaryFillholeImageFilter()
                    region_grown_image = fill_holes_filter.Execute(region_grown_image)

                    # Accumulate the segmented region into the output image
                    output_img = sitk.Maximum(output_img, region_grown_image)
                    print(np.sum(sitk.GetArrayFromImage(output_img)))
                else:
                    print(f"Seed {seed} is out of bounds for image size {img.GetSize()}")

            output_filename = os.path.join(output_dir, os.path.basename(img_flnm).replace(".nii", "_T2w_labels.nii"))
            print(output_filename)
            sitk.WriteImage(output_img, output_filename)

def build_phantom_labels_v2(metadata,bids_path, recon_dir, labels_dir,seeds,low=True):

    for (prj,sub, ses), _ in metadata.groupby(["prj","sub","ses"]):
        
        if low:
            input_dir = os.path.join(bids_path, prj,sub,ses,'anat')
        else:
            input_dir = os.path.join(bids_path, prj,'derivatives',recon_dir,sub,ses,'anat')

        mk_bids_dir(bids_path, prj,'derivatives',labels_dir,sub,ses,'anat')
        output_dir = os.path.join(bids_path, prj,'derivatives',labels_dir,sub,ses,'anat')

        img_flnms = sorted(glob.glob(f"{input_dir}/*.nii.gz"))
        print(input_dir)

        for img_flnm in img_flnms:
            print(img_flnm)
            img = sitk.ReadImage(img_flnm)

            # Initialize label map
            output_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
            output_img.CopyInformation(img)  # Copy metadata from the original image
            print(sitk.GetArrayFromImage(output_img).shape)
            labels = range(1, len(seeds) + 1)  # Unique label for each seed

            # Perform binary dilation and intensity threshold for each seed
            for seed, label in zip(seeds, labels):
                # Create a binary image representing the seed
                seed_img = sitk.Image(img.GetSize(), sitk.sitkUInt8)
                
                seed_img.CopyInformation(img)
                #print(sitk.GetArrayFromImage(seed_img).shape)
                seed_img[seed] = 1

                # Perform binary dilation
                region_grown_image = sitk.BinaryDilate(seed_img, 6)

                # Set the label value to the segmented region
                region_grown_image *= label

                # Convert output_img to array
                region_grown_array = sitk.GetArrayFromImage(region_grown_image)

                """ # Create mask array
                mask_array = np.zeros_like(region_grown_array, dtype=np.uint8)
                x, y, z = seed
                mask_array[z:z+1, y-3:y+4, x-3:x+4] = 1  # Set seed and neighboring voxels to 1
                mask_array[z+1:z+2, y-2:y+3, x-2:x+3] = 1  # Set seed and neighboring voxels to 1
                mask_array[z-1:z, y-2:y+3, x-2:x+3] = 1  # Set seed and neighboring voxels to 1
                """
                # Apply mask to output array
                #region_grown_array *= mask_array

                # Convert back to SimpleITK image
                region_grown_image = sitk.GetImageFromArray(region_grown_array)
                region_grown_image.CopyInformation(img)

                # Create a BinaryFillholeImageFilter
                fill_holes_filter = sitk.BinaryFillholeImageFilter()
                region_grown_image = fill_holes_filter.Execute(region_grown_image)

                # Accumulate the segmented region into the output image
                output_img = sitk.Maximum(output_img, region_grown_image)
                print(np.sum(sitk.GetArrayFromImage(output_img)))
            print(os.path.join(output_dir,os.path.basename(img_flnm).replace("T2w","T2w_labels")))
            sitk.WriteImage(output_img, os.path.join(output_dir,os.path.basename(img_flnm).replace("T2w","T2w_labels")))

def build_mask_from_labels(metadata, bids_path, labels_dir, masks_dir):

    for (prj,sub, ses), _ in metadata.groupby(["prj","sub","ses"]):
        
        input_dir = os.path.join(bids_path, prj,'derivatives',labels_dir,sub,ses,'anat')
        mk_bids_dir(bids_path, prj,'derivatives',masks_dir,sub,ses,'anat')
        output_dir = os.path.join(bids_path, prj,'derivatives',masks_dir,sub,ses,'anat')

        lbl_imgs = glob.glob(f"{input_dir}/*.nii.gz")

        for lbl_img_flnm in lbl_imgs:

            lbl_img = sitk.ReadImage(lbl_img_flnm)
            
            # Build mask and Save as nii.gz file
            mask = sitk.BinaryThreshold(lbl_img, lowerThreshold=1, upperThreshold=float(sitk.GetArrayFromImage(lbl_img).max()), insideValue=1, outsideValue=0)
            sitk.WriteImage(mask, os.path.join(output_dir,os.path.basename(lbl_img_flnm).replace("synthseg","mask")))

def extract_brain(metadata, bids_path, recon_dirname, mask_dirname, bet_dirname):
    for (prj,sub, ses), _ in metadata.groupby(["prj","sub","ses"]):
        
        mask_dir = os.path.join(bids_path, prj,'derivatives',mask_dirname,sub,ses,'anat')
        recon_dir = os.path.join(bids_path, prj,'derivatives',recon_dirname,sub,ses,'anat')

        mk_bids_dir(bids_path, prj,'derivatives',bet_dirname,sub,ses,'anat')
        bet_dir = os.path.join(bids_path, prj,'derivatives',bet_dirname,sub,ses,'anat')

        recon_imgs = glob.glob(f"{recon_dir}/*.nii.gz")
        mask_imgs = glob.glob(f"{mask_dir}/*.nii.gz")
        
        for recon_flnm, mask_flnm in zip(recon_imgs,mask_imgs):

            recon = sitk.ReadImage(recon_flnm)
            mask = sitk.ReadImage(mask_flnm)

             # Apply the mask to the T2w image
            bet = sitk.Mask(recon, mask)
            
            # Save the extracted image
            sitk.WriteImage(bet, os.path.join(bet_dir,os.path.basename(recon_flnm).replace(recon_dirname + ".nii",bet_dirname + ".nii")))

def convert_synthseg_to_feta(metadata, bids_path, synthseg_dir, feta_dir):

     for (prj,sub, ses), _ in metadata.groupby(["prj","sub","ses"]):
         
        input_dir = os.path.join(bids_path, prj,'derivatives',synthseg_dir,sub,ses,'anat')
        mk_bids_dir(bids_path, prj,'derivatives',feta_dir,sub,ses,'anat')
        output_dir = os.path.join(bids_path, prj,'derivatives',feta_dir,sub,ses,'anat')
        
        lbl_imgs = glob.glob(f"{input_dir}/*.nii.gz")

        for lbl_img_flnm in lbl_imgs:

            feta_img = sitk.ReadImage(lbl_img_flnm)
            lbl_img = sitk.ReadImage(lbl_img_flnm)
            synthseg = sitk.GetArrayFromImage(lbl_img)
            
            feta = np.zeros_like(synthseg)

            feta[synthseg == 24] = 1
            feta[(synthseg == 3) | (synthseg == 42)] = 2
            feta[(synthseg == 2) | (synthseg == 41)] = 3
            feta[(synthseg == 4) | (synthseg == 5) | (synthseg == 14) | (synthseg == 15) | (synthseg == 43) | (synthseg == 44)] = 4
            feta[(synthseg == 7) | (synthseg == 8) | (synthseg == 46) | (synthseg == 47)] = 5
            feta[(synthseg == 10) | (synthseg == 11) | (synthseg == 12) | (synthseg == 13) | 
                 (synthseg == 17) | (synthseg == 18) | (synthseg == 26) | (synthseg == 28) | 
                 (synthseg == 49) | (synthseg == 50) | (synthseg == 51) | (synthseg == 52) | 
                 (synthseg == 53) | (synthseg == 54) | (synthseg == 58) | (synthseg == 60)] = 6
            feta[synthseg == 16] = 7
                
            feta_img = sitk.GetImageFromArray(feta)
            feta_img.CopyInformation(lbl_img)

            # Build mask and Save as nii.gz file
            sitk.WriteImage(feta_img, os.path.join(output_dir,os.path.basename(lbl_img_flnm).replace("synthseg","feta")))

def build_jhu_ho_labels(metadata,bids_path,bet_dirname,mni_dirname,jhu_dirname,ho_dirname,low_field=False):
    for (prj,sub, ses), _ in metadata.groupby(["prj","sub","ses"]):
        
        jhu_dir = os.path.join(bids_path, prj,'derivatives',jhu_dirname,sub,ses,'anat')
        ho_dir = os.path.join(bids_path, prj,'derivatives',ho_dirname,sub,ses,'anat')
        bet_dir = os.path.join(bids_path, prj,'derivatives',bet_dirname,sub,ses,'anat')
        mni_dir = os.path.join(bids_path, prj,'derivatives',mni_dirname,sub,ses,'anat')

        mk_bids_dir(bids_path, prj,'derivatives',jhu_dirname,sub,ses,'anat')
        mk_bids_dir(bids_path, prj,'derivatives',ho_dirname,sub,ses,'anat')
        mk_bids_dir(bids_path, prj,'derivatives',mni_dirname,sub,ses,'anat')

        # build command

        # registder MNI152 standard to Volunteer brain -> not robust te written raw in string
        if low_field:
            cmd1 = f'flirt -in $FSLDIR/data/standard/MNI152_T1_1mm_brain -out {mni_dir}/{sub}_{ses}_{mni_dirname}.nii.gz -ref {bet_dir}/{sub}_{ses}_te-114_{bet_dirname}.nii.gz  -omat {mni_dir}/{sub}_{ses}_{mni_dirname}_omat.mat'
        else:
            cmd1 = f'flirt -in $FSLDIR/data/standard/MNI152_T1_1mm_brain -out {mni_dir}/{sub}_{ses}_{mni_dirname}.nii.gz -ref {bet_dir}/{sub}_{ses}_te-115_{bet_dirname}.nii.gz  -omat {mni_dir}/{sub}_{ses}_{mni_dirname}_omat.mat'

        # apply omat to JHU and HO atlases to get labels
        cmd2 = f'flirt -in $FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz -ref {mni_dir}/{sub}_{ses}_{mni_dirname}.nii.gz -applyxfm -init {mni_dir}/{sub}_{ses}_{mni_dirname}_omat.mat -out {jhu_dir}/{sub}_{ses}_{jhu_dirname}.nii.gz -interp nearestneighbour'
        cmd3 = f'flirt -in $FSLDIR/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr50-1mm.nii.gz -ref {mni_dir}/{sub}_{ses}_{mni_dirname}.nii.gz -applyxfm -init {mni_dir}/{sub}_{ses}_{mni_dirname}_omat.mat -out {ho_dir}/{sub}_{ses}_{ho_dirname}.nii.gz -interp nearestneighbour'

        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)
        subprocess.run(cmd3, shell=True)

def register_high_to_low_field(metadata,bids_path,recon_dirname):

    for (prj,sub,ses,echotime), sub_metadata in metadata.groupby(["prj","sub","ses","EchoTime"]):
        for _, acq in sub_metadata.iterrows():
            if not ((echotime == 299 and sub == 'sub-003') or (echotime == 299 and sub == 'sub-004')):
                moving_high_path = get_img_path(bids_path, acq,recon_dirname)
                fixed_low_path = re.sub(r'ses-\d{2}', 'ses-01', moving_high_path)
                fixed_low_path = re.sub(r'te-\d+', 'te-114', fixed_low_path)
                fixed_recon = sitk.ReadImage(fixed_low_path)
                recon_img = sitk.ReadImage(moving_high_path)
                recon_img = registration_elastix(fixed_recon,recon_img)
                sitk.WriteImage(recon_img,moving_high_path)
                print(f"Image saved in : {moving_high_path}")


