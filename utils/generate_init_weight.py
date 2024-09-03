import numpy as np
import scipy.io as sio
from utils import VCA
import os
from matplotlib import pyplot as plt

def generate_init_weight(data_name, is_show=True):
    """
    Generate the initial weight for the model
    weight: B x (m + 2)  
    VCA endmembers, water spectrum and target prior spectrum
    """
    data_path = 'dataset/' + data_name + '/data.mat'
    dataset = sio.loadmat(data_path)
    data = dataset['data']
    x_dim, y_dim, num_bands = data.shape
    target_prior = dataset['target']
    
    target_prior_spectrum = target_prior.squeeze()
    endmembers = 4
    data = np.reshape(data, (-1, num_bands)).T      # num of bands * num of pixels
    weight, IdxOfE, Xpca = VCA.VCA(data, endmembers)    # weight: num of bands * num of endmembers

    init_weight = np.column_stack((weight, target_prior_spectrum))

    save_path = fr'init_weight_VCA/{data_name}.npy'
    np.save(save_path, init_weight)

    if is_show:
        if not os.path.exists('init_weight_VCA/VCA_show'):
            os.makedirs('init_weight_VCA/VCA_show')
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(projection='3d')
        nonendmembers = np.delete(np.arange(Xpca.shape[1]), IdxOfE)
        ax.scatter(Xpca[0,nonendmembers], Xpca[1,nonendmembers], Xpca[2,nonendmembers], s=5, c='b')
        ax.scatter(Xpca[0,IdxOfE], Xpca[1,IdxOfE], Xpca[2,IdxOfE], s=40, c='r', zorder=1)
        plt.title('Gulfport Data Projected to 3D - Endmembers in Red')
        fig.savefig(fr'init_weight_VCA/VCA_show/{data_name}_3d_endmember.png')
        plt.close(fig)

        wavelengths = range(num_bands)
        fig2 = plt.figure()
        for i in range(init_weight.shape[1]):
            plt.plot(wavelengths, init_weight[:, i], label=f"endmembers {i+1}")
        plt.title('Total Initial Endmembers')
        plt.xlabel('Wavelength-Index')
        plt.ylabel('Reflectance')
        plt.legend()
        fig2.savefig(fr'init_weight_VCA/VCA_show/{data_name}_endmember_priori.png')
        plt.close(fig2)
    

    