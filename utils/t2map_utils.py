import os
import pydicom
import numpy as np
import subprocess
import pandas as pd
import json
import shlex
from pathlib import Path
from scipy.interpolate import griddata
import pickle
import matplotlib.pyplot as plt

from dcm_utils import *
from qmri_utils import *

##############################################
# Save in_vivo/in_vitro maps + in_vitro csv with statistical mean, std for each ROI
def save_nifti_maps(t2_map,k_map, sigma_map,res_map,t2map_dirname,recon_img,bids_path,acq,sim,analysis):
    for map, param_str in zip([t2_map, k_map, sigma_map,res_map], ['t2','k','sigma','res']):
        map_img = sitk.GetImageFromArray(map)
        map_img.SetSpacing(recon_img.GetSpacing())
        map_img.SetOrigin(recon_img.GetOrigin())
        map_img.SetDirection(recon_img.GetDirection()) 

        map_path = get_img_path(bids_path, acq.iloc[0], t2map_dirname)
        map_path = map_path.replace("t2map.nii.gz", "sim-" + str(sim)+ f"_{param_str}map_ada-{analysis}.nii.gz")
    
        sitk.WriteImage(map_img, map_path)  
    print(f"T2 map saved as nifti file in {t2map_dirname}")
def save_phantom_csv(t2_map,k_map,sigma_map,label,id,gt,bids_path,acq,t2map_dirname,sim,analysis):
    # Compute mean T2 per ROI
    n_roi = 5
    ut2 = np.zeros(n_roi)
    uk = np.zeros(n_roi)
    usigma = np.zeros(n_roi)
    stdt2 = np.zeros(n_roi)
    stdk = np.zeros(n_roi)
    stdsigma = np.zeros(n_roi)
    for i in range(n_roi): 
        ut2[i] = np.nanmean(t2_map[label==i+1])
        uk[i] = np.nanmean(k_map[label==i+1])
        usigma[i] = np.nanmean(sigma_map[label==i+1])
        stdt2[i] = np.nanstd(t2_map[label==i+1])
        stdk[i] = np.nanstd(k_map[label==i+1])
        stdsigma[i] = np.nanstd(sigma_map[label==i+1])

    # Create DataFrame
    df = pd.DataFrame({
        'id': id,
        'trueT2': gt,
        'meanT2': ut2,
        'stdT2': stdt2,
        'meanK': uk,
        'stdK': stdk,
        'meanC': usigma,
        'stdC': stdsigma
    })
    df_path = get_img_path(bids_path, acq.iloc[0], t2map_dirname).replace("t2map.nii.gz", f"sim-{sim}_ROI_data_ada-{analysis}.csv")
    df.to_csv(df_path, index=False) 
##############################################
# Compute residuals given T2w signals intensity measured and fitted
def compute_residuals(reshaped_t2w,TEeffs,fit,norm,k_map,t2_map, sigma_map, res_map,mask_indices,mask):
    # Calculate the predicted signal using the mono-exponential decay model
    predicted_signal = np.zeros_like(reshaped_t2w)
    if fit=='gaussian':
        for i, te in enumerate(TEeffs):
            predicted_signal[:, i] = k_map * np.exp(-te / t2_map)

    else:
        for i, te in enumerate(TEeffs):
            predicted_signal[:, i] = (k_map**2 * np.exp(-2*te / t2_map) + sigma_map**2)**(1/2)

    # Compute the residuals: difference between measured and predicted signals
    if norm:
        # Compute the maximum value in each row
        row_maxes = np.max(reshaped_t2w , axis=1, keepdims=True)

        # Normalize each row by its maximum value
        reshaped_t2w = reshaped_t2w  / row_maxes

    residuals = reshaped_t2w - predicted_signal

    # Compute the residual map as the sum of squared residuals across TEs
    res_map[mask_indices] = np.sum(residuals[mask_indices], axis=1)/len(TEeffs)

    # Reshape the residual map to the original spatial dimensions
    res_map = res_map.reshape(mask.shape[:3])

    return res_map
##############################################
# Compute in vitro noise to estimate sigma bounds
def estimate_in_vitro_noise(reshaped_t2w, reshaped_mask):
    outside_mask = reshaped_t2w[reshaped_mask[:, 0] == 0]
    reshaped_t2w_114 = reshaped_t2w[:, 0]  # This is the data at the first TE
    reshaped_mask_squeezed = np.squeeze(reshaped_mask)
    print(np.shape(reshaped_t2w_114))
    print(np.shape(reshaped_mask_squeezed))
    print(f"Number of 1s in the mask: {np.sum(reshaped_mask_squeezed == 1)}")
    print(f"Number of 0s in the mask: {np.sum(reshaped_mask_squeezed == 0)}")

    # Select values where mask is 0 (outside mask)
    masked_values = reshaped_t2w_114[reshaped_mask_squeezed == 0]
    # Calculate mean and standard deviation of outside-mask values
    print("Mean of outside mask values:", masked_values.mean())
    print("Std of outside mask values:", masked_values.std())

    # Reshape outside mask data back to original dimensions for further analysis if needed
    outside_mask_reshaped = outside_mask.reshape(-1, reshaped_t2w.shape[1])
    std_outside_mask = np.std(outside_mask_reshaped, axis=0)
    print("Std across echo times for outside mask:", std_outside_mask)
    mean_outside_mask = np.mean(outside_mask_reshaped, axis=0)
    print("Mean across echo times for outside mask:", mean_outside_mask)
##############################################
# Convergence Study Plots
def plot_convergence_20_random_voxels_colored_by_t2(ada_path,iteration_infos,t2_map,mask_indices,sub,ses,i,analysis):
    import matplotlib.pyplot as plt
    import random
    import matplotlib.cm as cm

    # Define the path to save the figures
    fig_path = ada_path
    print(fig_path)

    # Randomly select 20 voxels from iteration_infos
    num_voxels = len(iteration_infos)
    selected_voxels_indices = random.sample(range(num_voxels), 50)
    selected_iteration_infos = [iteration_infos[i] for i in selected_voxels_indices]
    selected_t2_values = [t2_map[mask_indices[i]] for i in selected_voxels_indices]

    # Normalize the T2 values for colormap
    norm = plt.Normalize(vmin=min(selected_t2_values), vmax=max(selected_t2_values))
    cmap = cm.jet

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the objective function value (loss) for each selected voxel
    for idx, (info, t2_value) in enumerate(zip(selected_iteration_infos, selected_t2_values)):
        objective_values = [entry['f_val'] for entry in info]
        color = cmap(norm(t2_value))
        ax.plot(range(len(objective_values)), objective_values, label=f'Voxel {selected_voxels_indices[idx]}', color=color)

    # Add colorbar to indicate T2 value range
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)  # Associate the colorbar with the plot's axis
    cbar.set_label('T2 Value')
    #ax.set_ylim([0,1])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Function Value (Loss)')
    ax.set_title('Convergence of 20 Random Voxels Colored by T2 Value')
    ax.grid(True)
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'convergence_20_random_voxels_colored_by_t2_{sub}_{ses}_sim-{i}_{analysis}.png'))
    plt.close()

    print(f"Convergence plots for 20 random voxels colored by T2 value saved to {fig_path}")
def plot_gradnorm_20_random_voxels_colored_by_t2(ada_path,iteration_infos, t2_map, mask_indices, sub,ses,i):
    import matplotlib.pyplot as plt
    import random
    import matplotlib.cm as cm
    import os

    # Define the path to save the figures
    fig_path = ada_path  # Replace with your actual directory path

    # Randomly select 20 voxels from iteration_infos
    num_voxels = len(iteration_infos)
    selected_voxels_indices = random.sample(range(num_voxels), 50)
    selected_iteration_infos = [iteration_infos[i] for i in selected_voxels_indices]
    selected_t2_values = [t2_map[mask_indices[i]] for i in selected_voxels_indices]

    # Normalize the T2 values for colormap
    norm = plt.Normalize(vmin=min(selected_t2_values), vmax=max(selected_t2_values))
    cmap = cm.jet

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the gradient norm for each selected voxel
    for idx, (info, t2_value) in enumerate(zip(selected_iteration_infos, selected_t2_values)):
        grad_norm_values = [entry['grad_norm'] for entry in info]
        color = cmap(norm(t2_value))
        ax.plot(range(len(grad_norm_values)), grad_norm_values, label=f'Voxel {selected_voxels_indices[idx]}', color=color)

    # Add colorbar to indicate T2 value range
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)  # Associate the colorbar with the plot's axis
    cbar.set_label('T2 Value')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Convergence of 20 Random Voxels Colored by T2 Value')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'gradnorm_20_random_voxels_colored_by_t2_{sub}_{ses}_sim-{i}.png'))
    plt.close()

    print(f"Convergence plots for 20 random voxels colored by T2 value saved to {fig_path}")
def plot_step_size_convergence_20_random_voxels_colored_by_t2(ada_path,iteration_infos, t2_map, mask_indices,sub,ses,i):
    import matplotlib.pyplot as plt
    import random
    import matplotlib.cm as cm
    import os
    
    # Define the path to save the figures
    fig_path = ada_path 

    # Randomly select 20 voxels from iteration_infos
    num_voxels = len(iteration_infos)
    selected_voxels_indices = random.sample(range(num_voxels), 20)
    selected_iteration_infos = [iteration_infos[i] for i in selected_voxels_indices]
    selected_t2_values = [t2_map[mask_indices[i]] for i in selected_voxels_indices]

    # Normalize the T2 values for colormap
    norm = plt.Normalize(vmin=min(selected_t2_values), vmax=max(selected_t2_values))
    cmap = cm.jet

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the step size for each selected voxel
    for idx, (info, t2_value) in enumerate(zip(selected_iteration_infos, selected_t2_values)):
        step_sizes = [entry['step_size'] for entry in info]
        color = cmap(norm(t2_value))
        ax.plot(range(len(step_sizes)), step_sizes, label=f'Voxel {selected_voxels_indices[idx]}', color=color)

    # Add colorbar to indicate T2 value range
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)  # Associate the colorbar with the plot's axis
    cbar.set_label('T2 Value')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Step Size')
    ax.set_title('Step Size Convergence of 20 Random Voxels Colored by T2 Value')
    ax.grid(True)
    plt.yscale('log')  # Log scale for step size

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'step_size_convergence_20_random_voxels_colored_by_t2_{sub}_{ses}_sim-{i}.png'))
    plt.close()

    print(f"Step size convergence plots for 20 random voxels colored by T2 value saved to {fig_path}")
def plot_scatter_iterations_vs_loss_colored_by_t2(ada_path,num_iterations_array,final_errors_array,t2_map,mask_indices,sub,ses,i):

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import matplotlib.cm as cm

    # Define the path to save the figures
    fig_path = ada_path # Replace with your actual directory path
    os.makedirs(fig_path, exist_ok=True)

    # Extract final values from the results
    final_num_iterations = np.array(num_iterations_array)
    final_errors = np.array(final_errors_array)
    t2_values = t2_map[mask_indices]

    # Normalize the T2 values for colormap
    norm = plt.Normalize(vmin=np.min(t2_values), vmax=np.max(t2_values))
    cmap = cm.jet

    # Create a figure and axis for the scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot with colors based on T2 values
    sc = ax.scatter(final_num_iterations, final_errors, c=t2_values, cmap=cmap, norm=norm)

    # Add colorbar to indicate T2 value range
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('T2 Value')

    # Set labels and title
    #ax.set_ylim([20,60])
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Final Loss Function Value')
    ax.set_title('Final Number of Iterations vs Final Loss Value (Colored by T2 Value)')
    ax.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'scatter_iterations_vs_loss_colored_by_t2_{sub}_{ses}_sim-{i}.png'))
    plt.close()

    print(f"Scatter plot of iterations vs loss colored by T2 value saved to {fig_path}")
#############################################
## Obsolete
def set_A(ada_path,t2w, label, gt, TEeffs):
    # Initialize lists for points and Amean values
    points = []
    Ameans = []
    T2s = []

    # Iterate through each echo time (TE)
    for j, te in enumerate(TEeffs):
        # Extract the T2-weighted image for the current TE
        t2w_te = t2w[:, :, :, j]
        for i in range(1, len(gt)):
            # Calculate the mean signal for the current ROI
            Smean = np.mean(t2w_te[label == i])
            # Calculate A using the given formula
            Amean = Smean / np.exp(-te / gt[i])

            # Debugging output
            #print(f"GT: {gt[i]}, TE: {te}, Smean: {Smean}, Amean: {Amean}")

            # Store points and Amean for interpolation
            points.append([te, Smean])
            Ameans.append(Amean)
            T2s.append(gt[i])

    # Convert lists to numpy arrays
    points = np.array(points)
    Ameans = np.array(Ameans)
    T2s = np.array(T2s)



    # Determine the min and max of the TEeffs values
    TE_min = min(TEeffs)
    TE_max = max(TEeffs)
    
    # Create a grid of TE values with a step size (delta) of 1
    TE_linspace = np.arange(TE_min, TE_max + 1, 1)
    
    # Determine the min and max of the Smean values from the points
    Smean_min = 0 #np.round(min(points[:, 1]))
    Smean_max = np.round(max(points[:, 1])) + 500

    # Create a grid of Smean values with a step size (delta) of 1
    Smean_linspace = np.arange(Smean_min, Smean_max + 1, 1)
    
    # Create the meshgrid using TE_linspace and Smean_linspace
    te_grid, Smean_grid = np.meshgrid(TE_linspace, Smean_linspace)
    Amean_interpolated = griddata(points, Ameans, (te_grid, Smean_grid), method='nearest')
    T2_interpolated = griddata(points,T2s, (te_grid, Smean_grid), method='nearest')

    Amean_dict = {}
    t2_dict = {}
    for i in range(te_grid.shape[0]):
        for j in range(te_grid.shape[1]):
            te = te_grid[i, j]
            smean = Smean_grid[i, j]
            Amean = Amean_interpolated[i, j]
            Amean_dict[(te, smean)] = Amean
            t2 = T2_interpolated[i,j]
            t2_dict[(te, smean)] = t2
   
    # Assuming Amean_dict is your dictionary
    A_save_path = os.path.join(ada_path, 'Amean_interpolated.pkl')
    T2_save_path = os.path.join(ada_path, 'T2_interpolated.pkl')

    # Save the dictionary to a file using pickle
    with open(A_save_path, 'wb') as f:
        pickle.dump(Amean_dict, f)
    print(f"Interpolated Amean data saved to {A_save_path}")

    # Save the dictionary to a file using pickle
    with open(T2_save_path, 'wb') as f:
        pickle.dump(t2_dict, f)
    print(f"Interpolated Amean data saved to {T2_save_path}")


    # Create a scatter plot of the original Amean values
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=Ameans, cmap='viridis', s=100, edgecolor='k', alpha=0.75, vmin=0, vmax=7000)
    
    # Add a color bar to show the values of Amean
    plt.colorbar(scatter, label='Amean')
    plt.xlabel('TE (Echo Time)')
    plt.ylabel('Smean (Signal Mean)')
    plt.title('Scatter Plot of Computed Amean Values')
    plt.grid(True)
    
    # Save the scatter plot to a file
    plot_save_path = os.path.join(ada_path, 'Ameans_ses-1.png')
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Scatter plot saved to {plot_save_path}")

    # Create a scatter plot of the original Amean values
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=T2s, cmap='viridis', s=100, edgecolor='k', alpha=0.75, vmin=20, vmax=300)
    
    # Add a color bar to show the values of Amean
    plt.colorbar(scatter, label='T2')
    plt.xlabel('TE (Echo Time)')
    plt.ylabel('Smean (Signal Mean)')
    plt.title('Scatter Plot of Known T2 Values')
    plt.grid(True)
    
    # Save the scatter plot to a file
    plot_save_path = os.path.join(ada_path, 'T2s_ses-1.png')
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Scatter plot saved to {plot_save_path}")

    return Amean_dict, t2_dict
