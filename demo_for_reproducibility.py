import scipy.io as sio
import numpy as np
from main import HyperspecAE
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utils import threshold

def get_AUC(result_map, gt, data_name, is_show):
    path = fr'dataset/{data_name}/data.mat'
    try:
        nr, nc, _ = gt.shape
    except:
        print(gt.shape)
    # normalize
    try:
        min_val = np.min(result_map)
        max_val = np.max(result_map)
        range_val = max_val - min_val
        if range_val == 0:
            print("Warning: min and max values are equal. Division by zero.")
            print(min_val, max_val)   
            return 0
        else:
            result_norm = (result_map - min_val) / range_val
    except Exception as e:
        print("Unexpected error:", e)
        print(result_map.min(), result_map.max())
    gt = gt.flatten()
    result_norm = result_norm.flatten()
    FPR, TPR, threshold = roc_curve(gt, result_norm)
    test_auc = auc(FPR, TPR)
    if is_show:
        plt.title('ROC curve AUC = %0.4f' % test_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.plot(FPR, TPR, color='orange', linewidth=3)
        plt.plot([0, 1], [0, 1], color='m', linestyle='--') 
        plt.savefig('./ROC_curve_{}.png'.format(data_name))
        plt.clf()
        plt.close()

    return test_auc


def test_demo_for_reproducibility(data_name, pth_path, endmembers=4, activation='LeakyReLU', is_nonlinear=True):
    data = sio.loadmat(f'dataset/{data_name}/data.mat')['data']
    gt = sio.loadmat(f'dataset/{data_name}/data.mat')['gt']
    num_bands = data.shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hsi = torch.tensor(data.reshape(-1, num_bands)).float().to('cuda')
    
    model = HyperspecAE(data_name=data_name, num_bands=num_bands, endmembers=4, activation=activation, is_nonlinear=is_nonlinear).to(device)
    model.load_state_dict(torch.load(pth_path), strict=False)
    model.eval()
    with torch.no_grad():
        enc_out, _ = model(hsi)
        detect_result = enc_out.detach().cpu().squeeze().numpy().T[-1].reshape(data.shape[0], data.shape[1])
        auc = get_AUC(detect_result, gt, data_name, is_show=True)
        print(f'AUC: {auc}')
        
        after_threshold = threshold.threshold(detect_result, 0.1)
        plt.imshow(after_threshold)
        plt.axis('off')
        plt.savefig(f'./detect_result_{data_name}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()


if __name__ == '__main__':
    # dataset name: 'River_scene2', weights path: './NUN-UTD_River_scene2_best.pth'
    test_demo_for_reproducibility('simulated_data', 'result/simulated_data/NUN-UTD_best.pth')

