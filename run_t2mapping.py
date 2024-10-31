from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e
import time
import SimpleITK as sitk
from functools import partial

from utils.t2map_utils import *
from utils.metadata_utils import *
import argparse

# Set functions
def set_phantom_gt(low_field):
    id = ["T2-3", "T2-5", "T2-7", "T2-9", "T2-11"]
    if low_field:
        gt = [594,284,167,80,40]
    else:
        gt = [428,186,90,44,19]
    return gt,id

def set_fit_params(args):
    """ Set fitting parameters including initial guess, bounds, solver and options for convergence
        Given fitting can follow different noise distribution, fitting parameters might change and are handle here.
        In case prior knowledge is not used regarding M0 minimal bound, that bounds must be handled voxel wise 
        and is therefore done in the fit_voxel function instead.
    """
    # LOW FIELD
    if args.gaussian and args.lf and not args.norm:
        fit = 'gaussian'
        fit_params =    {   "initial_guess": [650, 165],
                            "param_bounds": [(600,10000),(10,600)],
                            "solver": "L-BFGS-B",
                            "options": {
                                "ftol": 1e-6,
                                "maxls": 50,
                                "disp": False
                            }
                        }
    elif args.gaussian_rician and args.lf and not args.norm:
        fit = 'gaussian_rician'
        fit_params =    {   "initial_guess": [650, 110, 40],
                            "param_bounds": [(550,10000),(10,600),(2,1000)],
                            "solver": "L-BFGS-B",
                            "options": {
                                "gtol": 1e-2,
                                "ftol": 1e-2,
                                "maxls": 50,
                                "disp": False
                            }
                        }
    elif args.rician and args.lf and not args.norm:
        fit = 'rician'
        fit_params =    {   "initial_guess": [650, 110, 40],
                            "param_bounds": [(550,900),(10,600),(2,1000)],
                            "solver": "L-BFGS-B",
                            "options": {
                                "gtol": 1e-2,
                                "ftol": 1e-2,
                                "maxls": 50,
                                "disp": False
                            }
                        }
    # HIGH-FIELD
    elif args.gaussian and args.hf and not args.norm:
        fit = 'gaussian'
        fit_params =    {   "initial_guess": [890, 165],
                            "param_bounds": [(850,30000),(10,600)],
                            "solver": "L-BFGS-B",
                            "options": {
                                "ftol": 1e-6,
                                "maxls": 50,
                                "disp": False
                            }
                        }
    elif args.gaussian_rician and args.hf and not args.norm:
        fit = 'gaussian_rician'
        fit_params =    {   "initial_guess": [890, 110, 40],
                            "param_bounds": [(850,30000),(30,600),(2,1000)],
                            "solver": "L-BFGS-B",
                            "options": {
                                "gtol": 1e-2,
                                "ftol": 1e-2,
                                "maxls": 50,
                                "disp": False
                            }
                        }
    elif args.rician and args.hf and not args.norm:
        fit = 'rician'
        fit_params =    {   "initial_guess": [17, 40, 0.15],
                            "param_bounds": [(850,30000),(30,600),(7,200)],
                            "solver": "L-BFGS-B",
                            "options": {
                                "gtol": 1e-2,
                                "ftol": 1e-2,
                                "maxls": 50,
                                "disp": False
                            }
                        }
    else:
        print("Error: Normalization is set to true though no parameters where defined yet. Please modify set_fit_params to manage.")
        exit(1)

    return fit, fit_params

def set_ada_path(bids_path,prj):
    ada_path = os.path.join(bids_path,prj,'ada/convergence_analysis')
    if not os.path.exists(ada_path):
        os.mkdir(ada_path)
    return ada_path
# *********************************************************************
# Fit voxel 
def fit_voxel(voxel, fit, fit_params, TEeffs, reshaped_t2w,prior,norm):
    """ Voxel-wise T2 mapping fitting
            Sub functions are defined including models, objectives and callback
            Scipy minimize is used to fit. Fitting can be done under different noise assumption: gaussian, gaussian_rician, rician
            If no prior knoledge is used on M0 -> minimal bound is set to T2w signal intensity at minimal TE.
            The function return the fitted params and convergence information
        This function is called using Pool process to parallelized voxels fitting. Partial function tools is used to pass some fixed arguments
    """
    # Models ******************************************************************
    def gauss_model(t, k, t2):
        """Mono-exponential T2 relaxation decay assuming gaussian noise distribution"""
        return k * np.exp(-t / t2)

    def gauss_rician_model(t,k,t2,sigma):
        """ Mono-exponential T2 relaxation decay assuming gaussian-rician noise distribution
            Gaussian-Rician objective is based on this paper:
            Gudbjartsson H, Patz S. The Rician distribution of noisy MRI data. Magn Reson Med. 1995 Dec;34(6):910-4. doi: 10.1002/mrm.1910340618.
        """
        return ( k**2 * np.exp(-2*t/t2) + sigma**2 )**(1/2)

    # Objectives ***************************************************************
    def gauss_obj(params, TEeffs, signal):
        """Least-squares objective function under gaussian noise asssumption"""
        k, t2 = params
        model = gauss_model(TEeffs, k, t2)
        residuals = signal - model

        return np.sum(residuals**2) / len(signal)

    def gauss_rician_obj(params, TEeffs, signal):
        """Least-squares objective function under gaussian-rician noise asssumption"""
        k, t2, sigma = params
        model = gauss_rician_model(TEeffs, k, t2,sigma)
        residuals = signal - model

        return np.sum(residuals**2) / len(signal)

    def rician_obj(params,TEeffs,signal):
        """ negative log-lilelihood of rician distribution
            This objective is based on this paper: 
            C. E. Hajj, S. Moussaoui, G. Collewet and M. Musse, 
            Multi-Exponential Transverse Relaxation Times Estimation From Magnetic Resonance Images Under Rician Noise and Spatial Regularization, 
            in IEEE Transactions on Image Processing, vol. 29, pp. 6721-6733, 2020, doi: 10.1109/TIP.2020.2993114. 
        """
        k, t2, sigma = params
        model = gauss_model(TEeffs, k, t2)
        x = (model * signal) / (sigma**2)

        ll = np.sum(
            ( np.log(signal) - np.log(sigma**2)) -
            (signal**2 + model**2) / (2 * sigma**2) +
            (np.abs(x) + np.log(i0e(x)))
        )

        if np.isnan(ll):
            print(f"NaN detected in objective function for params: {params}")

        return -ll

    # Callbacks ****************************************************************
    def callback(xk, TEeffs, signal):
        """ Callback of the fitting process when rician or gauss-rician ojbjective is used.
            The callback records the objective function values, the norm of the gradient (if jacobian accessible) and the step size
            This information can be used for convergence plots.
        """
        # Get the current value of the objective function
        if fit == 'rician':
            f_val = rician_obj(xk, TEeffs, signal)
        else:
            f_val = gauss_rician_obj(xk, TEeffs, signal)
        # Gradient norm is omitted since `result.jac` is not accessible
        grad_norm = None

        # Calculate the step size (if you can derive it from xk or the previous xk)
        if callback.prev_xk is not None:
            step_size = np.linalg.norm(xk - callback.prev_xk)
        else:
            step_size = np.nan
            
        # Update previous xk
        callback.prev_xk = xk
        
        # Store the metrics (you can append them to a list or print them)
        callback.iteration_info.append({
            'f_val': f_val,
            'grad_norm': grad_norm,  #
            'step_size': step_size,
        })

    def callback_gauss(xk, TEeffs, signal):
        """ Callback of the fitting process when gauss objective is used.
            The callback records the objective function values, the norm of the gradient (if jacobian accessible) and the step size
            This information can be used for convergence plots.
        """
        # Get the current value of the objective function
        f_val = gauss_obj(xk, TEeffs, signal)
        
        # Gradient norm is omitted since `result.jac` is not accessible
        grad_norm = None
        
        # Calculate the step size (if you can derive it from xk or the previous xk)
        if callback_gauss.prev_xk is not None:
            step_size = np.linalg.norm(xk - callback_gauss.prev_xk)
        else:
            step_size = np.nan
            
        # Update previous xk
        callback_gauss.prev_xk = xk
        
        # Store the metrics (you can append them to a list or print them)
        callback_gauss.iteration_info.append({
            'f_val': f_val,
            'grad_norm': grad_norm,  # Will be None in this case
            'step_size': step_size,
        })
    
    # Normalize signal if required (I don't use normalization but this is an option that might be useful eventually for further research)
    if norm:
        fitted_t2w = reshaped_t2w[voxel, :] / np.max(reshaped_t2w[voxel, :])
    else:
        fitted_t2w = reshaped_t2w[voxel, :]

    # set fit parameters
    if not prior:
        fit_params['param_bounds'][0] = (reshaped_t2w[voxel, 0],10000)

    # Initialize callback *********************************************************
    if fit != 'gaussian':
        callback.prev_xk = None
        callback.iteration_info = []
        # Create the partial callback with TEeffs and signal
        partial_callback = partial(callback, TEeffs=TEeffs, signal=np.array(fitted_t2w))
    else:
        callback_gauss.prev_xk = None
        callback_gauss.iteration_info = []
        # Create the partial callback with TEeffs and signal
        partial_callback = partial(callback_gauss, TEeffs=TEeffs, signal=np.array(fitted_t2w))
    
    # Fit *********************************************************
    if fit == 'gaussian':
        result = minimize(gauss_obj, 
                          fit_params['initial_guess'], 
                          args=(TEeffs, np.array(fitted_t2w)),
                          method = fit_params['solver'], 
                          bounds=fit_params['param_bounds'],
                          options=fit_params['options'],
                          jac=False,
                          callback=partial_callback)
    elif fit == 'gaussian_rician':
        result = minimize(gauss_rician_obj, 
                          fit_params['initial_guess'], 
                          args=(TEeffs, np.array(fitted_t2w)),
                          method = fit_params['solver'], 
                          bounds=fit_params['param_bounds'],
                          options=fit_params['options'],
                          jac=False,  # Assume you have the gradient
                          callback=partial_callback)
    elif fit == 'rician': 
        result = minimize(rician_obj, 
                          fit_params['initial_guess'], 
                          args=(TEeffs, np.array(fitted_t2w)),
                          method = fit_params['solver'], 
                          bounds=fit_params['param_bounds'],
                          options=fit_params['options'],
                          jac=False,  # Assume you have the gradient
                          callback=partial_callback)
    
    # Out Results *********************************************************
    if result.success:
        params = result.x
        convergence_flag = result.success
        num_iterations = result.nit
        final_error = result.fun
        if fit != 'gaussian':
            iteration_info = callback.iteration_info
        else:
            iteration_info = callback_gauss.iteration_info
    else:
        params = result.x
        #print("=====================================")
        print(f"FAIL : Optimization failed for voxel {voxel}: {result.message}")
        print('Objective function value at optimum:', result.fun)
        print('params', result.x)
        convergence_flag = result.success
        num_iterations = result.nit
        final_error = result.fun
        if fit != 'gaussian':
            iteration_info = callback.iteration_info
        else:
            iteration_info = callback_gauss.iteration_info
        
    return params, convergence_flag, num_iterations, final_error, iteration_info

##############################################
# Final T2 fitting function
def set_in_vivo_jmri(sim_id):

    fit_params =    {   "initial_guess": [630, 165],
                        "param_bounds": [(600,900),(10,600)],
                        "solver": "L-BFGS-B",
                        "options": {
                            "ftol": 1e-6,
                            "maxls": 50,
                            "disp": False
                        }
                    }

    dfs = 0

    return dfs
##############################################

def process_t2maps(metadata, bids_path, TEs, fit, fit_params, phantom, low_field, prior, fast, norm, sim):
    """ PROCESS_T2MAPS manages manages and launches the voxel-wise T2 fit and save T2 maps as nifti imgs.
        Args in:
        metadata :  panda dataframe of data to process
        bids_path:  path to the root of the bids data. ../qMRI/
        TEs:        List of the TEs to process default is [114,202,299] at low-field, and [115,202,299] at high-field
        fit:        fit type str: gaussian - gaussian_rician - rician
        fit params: dictionary of the parameters required to fit using scipy.minimize (list, initial guess, bounds, solver, ftol)
        phantom:    Boolean if data to process is in vitro or in_vivo
        low_field:  Boolean low_field vs high_field
        prior:      Boolean - if set to True, M0 bounds are restricted. If set to False, minimal M0 bounds is set to T2w signal intensity at minimal TE
        fast:       Boolean for in_vitro only, if True, only voxels inside ROI is analyzed (no full maps are output)
        norm:       Boolean if True, will normalized T2w signal intensities
        sim:        Str ID of the T2 fit 
    """

    # get only metadata for TEs selected
    tes_in_seconds = [x / 1000 for x in TEs]
    metadata = metadata[metadata['EchoTime'].isin(tes_in_seconds)]

    # process data by project if data from different projects are processed at once
    for prj, prj_metadata in metadata.groupby("prj"):
        ada_path =set_ada_path(bids_path,prj)

        # iterate through each subject session and process data
        for (sub,ses), sub_metadata in prj_metadata.groupby(["sub","ses"]):
            
            # Initialization of arrays for subsequent t2 fit
            t2w = []
            mask = []
            TEeffs = []

            for echotime, acq in sub_metadata.groupby("EchoTime"):
                # get metadata of current sub, ses at a selected echotime.
                # you should get 3 imgs (ax, cor, sag). Use only first img to read recon img and mask
                
                TEeffs.append(echotime*1000) 
                recon_flnm = get_img_path(bids_path, acq.iloc[0], recon_dirname).replace(' ', '')
                mask_flnm = get_img_path(bids_path, acq.iloc[0], mask_dirname).replace(' ', '')
                if phantom:
                    label_flnm = get_img_path(bids_path, acq.iloc[0], phantom_labels_dirname).replace(' ', '')
                recon_img = sitk.ReadImage(recon_flnm)
                mask_img = sitk.ReadImage(mask_flnm)
                mask.append(sitk.GetArrayFromImage(mask_img))
                t2w.append(sitk.GetArrayFromImage(recon_img))

            # In vitro - load label img too
            if phantom:
                label = sitk.GetArrayFromImage(sitk.ReadImage(label_flnm))

            mask = np.stack(mask, axis=-1)
            mask = np.sum(mask,axis=3) > 0
            t2w = np.stack(t2w, axis=-1)                    
            TEeffs = np.array(TEeffs)
            
            # check if TEs selected were acquired for specific subject and session, if no skip fit
            if not np.array_equal(TEeffs, TEs):
                print(f"Warning: one or more TEs selected to fit is missing for {sub}_{ses}. T2 fit is skipped.")
            else:       
                
                # IN VITRO - quick analysis at ROI ***************************
                if phantom and fast:
                    # for ultra-fast analysis only you can decide to only check manually values at one slice ^^
                    #z_indices = np.arange(mask.shape[2])
                    #x_indices = np.arange(mask.shape[0])
                    #y_indices = np.arange(mask.shape[1])
                    #label[(x_indices < 107) | (x_indices > 107), :, :] = 0 
                    mask[label == 0] = 0
                
                #**************************************************
                # Printing information prior fit launch
                print(f"T2 Mapping: {prj}_{sub}_{ses}")
                print(f"Dimensions of the simulated t2w images: {t2w.shape} (x,y,slice,necho)")
                print(f"Mask Dimension: {mask.shape} -  Number of voxels inside mask: {int(np.sum(mask))}")
                print(f"TEeffs: {TEeffs}")

                #**************************************************
                # reshape for computation time
                reshaped_t2w = np.reshape(t2w, (-1, TEeffs.size)).astype(np.float32)
                reshaped_mask = np.reshape(mask, (-1, 1))

                # Initialize an array to store the fitting parameters for each voxel
                t2_map = np.zeros_like(reshaped_t2w[..., 0])
                k_map = np.zeros_like(reshaped_t2w[..., 0])
                sigma_map = np.zeros_like(reshaped_t2w[..., 0])
                res_map = np.zeros_like(reshaped_t2w[..., 0])
                
                # get indices inside mask
                mask_indices, _ = np.where(reshaped_mask)
                
                # NOISE ESTIMATION OF BACKGROUND VOXELS
                #if phantom:
                #    estimate_in_vitro_noise(reshaped_t2w, reshaped_mask)

                #********** Partial Function Definition *****************************************************
                # Launch Fitting
                starttime = time.time()
                partial_fit_voxel = partial(fit_voxel, 
                                                fit=fit,
                                                fit_params=fit_params, 
                                                TEeffs=TEeffs, 
                                                reshaped_t2w=reshaped_t2w,
                                                prior=prior,
                                                norm=norm)
                
                # Fit on multiple workers
                print(f"Fitting using {fit} model ... ")
                print("Expected computation time for a full map: ~2-3minutes on CPU)")

                with Pool(processes=20) as pool:
                    all_results  = pool.map(partial_fit_voxel, mask_indices)
                
                print(f"... done. Time to fit: {round(time.time()-starttime, 4)} sec")

                #*********************************************************************************************
                # Process results
                results = np.array([result[0] for result in all_results])
                convergence_flags = [result[1] for result in all_results]
                num_iterations_array = [result[2] for result in all_results]
                final_errors_array = [result[3] for result in all_results]
                iteration_infos = [result[4] for result in all_results]

                if fit == 'gaussian':
                    t2_map[mask_indices], k_map[mask_indices] = results[:, 1].astype(np.float32), results[:, 0].astype(np.float32)
                else:
                    t2_map[mask_indices], k_map[mask_indices], sigma_map[mask_indices] = results[:, 1].astype(np.float32), results[:, 0].astype(np.float32), results[:, 2].astype(np.float32)

                # build residual maps
                res_map = compute_residuals(reshaped_t2w,TEeffs,fit,norm,k_map,t2_map,sigma_map,res_map,mask_indices,mask)

                #*********************************************************************************************
                # Convergence study
                plot_convergence_20_random_voxels_colored_by_t2(ada_path,iteration_infos,t2_map,mask_indices,sub,ses,sim,fit)
                #plot_gradnorm_20_random_voxels_colored_by_t2(ada_path,iteration_infos, t2_map, mask_indices,sub,ses,sim)
                plot_step_size_convergence_20_random_voxels_colored_by_t2(ada_path,iteration_infos, t2_map, mask_indices,sub,ses,sim)
                plot_scatter_iterations_vs_loss_colored_by_t2(ada_path,num_iterations_array,final_errors_array,t2_map,mask_indices,sub,ses,sim)
                # ********************************************************************************************
                # Reshape map and Build and save nifti image
                t2_map = np.reshape(t2_map, (t2w.shape[0], t2w.shape[1], t2w.shape[2]))
                k_map = np.reshape(k_map, (t2w.shape[0], t2w.shape[1], t2w.shape[2]))
                sigma_map = np.reshape(sigma_map, (t2w.shape[0], t2w.shape[1], t2w.shape[2]))
                save_nifti_maps(t2_map,k_map, sigma_map,res_map,t2map_dirname,recon_img,bids_path,acq,sim, fit)
                
                # build csv for phantom analysis
                if phantom:
                    id,gt = set_phantom_gt(low_field)
                    save_phantom_csv(t2_map,k_map,sigma_map,label,id,gt,bids_path,acq,t2map_dirname,sim,fit)

                #*********************************************************************************************

def parse_arguments():
    parser = argparse.ArgumentParser(description="T2 Mapping Parser",formatter_class=argparse.RawTextHelpFormatter)
    
    # Required path argument
    parser.add_argument('--path', type=str, required=True, help="Path to general directory ../qMRI/")
    parser.add_argument('--csv', type=str, nargs='+', required=True, 
                        help=(  "Can be either:\n"
                                "  (1) Name of one or more metadata CSV log files (e.g., YYYYMMDDHH_MMSS0000.csv)\n"
                                "  (2) Name of project to process all CSV log files in that project (e.g., prj-00X)"))
    
    # Optional mutually exclusive arguments: --in_vivo or --in_vitro --in_vitro_fast
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--in_vivo', action='store_true', help="Process in vivo data")
    group.add_argument('--in_vitro', action='store_true', help="Process NIST Phantom data and generate full map")
    group.add_argument('--in_vitro_fast', action='store_true', help="Process NIST Phantom data only at ROI")
    
    # Optional mutually exclusive arguments: --gaussian --gaussian_rician -- rician
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gaussian', action='store_true', help="T2 fit under gaussian noise assumption")
    group.add_argument('--gaussian_rician', action='store_true', help="T2 fit under gaussian-rician noise assumption")
    group.add_argument('--rician', action='store_true', help="T2 fit under rician noise assumption")

    # Specify TEs to fit
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--lf', action='store_true', help="Process low-field 0.55 T data")
    group2.add_argument('--hf', action='store_true', help="Process high-field 1.5 T data")
    parser.add_argument('--sim', type=str, required=True, help="T2 fitting ID (can be a description or a simple int)")

    # Optional arguments
    parser.add_argument('--TEs', nargs='+', type=int, help=("List of TEs to fit\n"
                                                                "   default low-field: [114,202,299]\n"
                                                                "   default high-field: [115,202,299]"))
    parser.add_argument('--no_prior', action='store_true', default=False, help="If set, will not restrict M0 bounds")
    parser.add_argument('--norm', action='store_true', default=False, help="If set, will normalize T2w data")

    return parser.parse_args()

###############     MAIN      ################
##############################################
def main():

    args = parse_arguments()

    # Check if the provided path exists
    if not os.path.exists(args.path):
        print(f"Error: The specified path does not exist: {args.path}")
        exit(1)

    bids_path = os.path.join(args.path,'projects/')
    csv_path = os.path.join(args.path,'dicom/logs/')
    
    # Handle the chosen parsed action
    if args.lf:
        low_field = True
    elif args.hf:
        low_field = False

    if args.TEs is None and args.lf:
        TEs = [114,202,299]
    elif args.TEs is None and args.hf:
        TEs = [115,202,299]
    else:
        TEs = args.TEs

    if args.in_vivo:
        phantom = False
        fast = False
    elif args.in_vitro:
        phantom = True
        fast = False
    elif args.in_vitro_fast:
        phantom = True
        fast = True

    if args.norm:
        print("Warning: Fitting using normalization is not optimal !")
        norm = True
    else:
        norm = False

    if args.no_prior:
        prior = False
    else:
        prior = True

    sim_id = args.sim

    # set fit parameters given parsed arguments
    fit,fit_params = set_fit_params(args)

    # load metadata dataframe
    metadata = set_metadata(csv_path,args.csv,low_field)

    process_t2maps(metadata, bids_path, TEs, fit, fit_params, phantom, low_field, prior, fast, norm, sim_id)

if __name__ == '__main__':
    main()
