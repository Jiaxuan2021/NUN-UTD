import scipy.io as sio
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils.train_objectives import SAD, SID
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.cal_AUC_ROC import get_epoch_AUC
from utils.underwater_dataset import RiverDataset
from utils.generate_init_weight import generate_init_weight
import random
import math
from utils import threshold
from sklearn.metrics import roc_curve, auc
from config import get_args


def seed_torch(seed): 
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def open_file(path):
    """
    Open mat file
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    assert ext == '.mat', 'Only support .mat file'
    return sio.loadmat(path)

class GaussianDroupout(nn.Module): 
    """
    Multiplied by a Gaussian sequence, 
    each forward propagation will be slightly different, 
    introducing a certain randomness
    """
    def __init__(self, alpha=1.0): # alpha is the variance of Gaussian distribution
        super(GaussianDroupout, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        N(1, alpha)
        """
        if self.training:
            # Gaussian noise
            noise = torch.randn(x.size()) * self.alpha + 1   # mean=1, std=self.alpha
            if x.is_cuda:
                noise = noise.cuda()
                return x * noise
            else:
                raise RuntimeError('GaussianDroupout is only supported on CUDA while training')
        else:
            return x
        
class ASC(nn.Module):
    """
    Abundances sum to one and constraint
    """
    def __init__ (self):
        super(ASC, self).__init__()
    
    def forward(self, input):
        # ANC ASC
        constrained = F.softmax(input, dim=1)

        return constrained
    
class HyperspecAE(nn.Module):
    """
    Autoencoder for underwater hyperspectral unmixing and target detection
    """
    def __init__(self, data_name, num_bands, endmembers, activation: str='LeakyReLU', is_nonlinear: bool=True):   
        super(HyperspecAE, self).__init__()
        self.num_bands = num_bands
        self.endmembers = endmembers
        self.is_nonlinear = is_nonlinear
        self.data_name = data_name

        self.gauss = GaussianDroupout()
        self.asc = ASC()

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()

        self.encoder = nn.Sequential(OrderedDict([
                                            ('h1', nn.Linear(num_bands, 18*endmembers)),
                                            ('activation_01', self.activation),
                                            ('hidden_1', nn.Linear(18*endmembers, 9*endmembers)),
                                            ('activation1', self.activation),
                                            ('hidden_2', nn.Linear(9*endmembers, 6*endmembers)),
                                            ('activation2', self.activation),
                                            ('hidden_3', nn.Linear(6*endmembers, 3*endmembers)),
                                            ('activation3', self.activation),
                                            ('hidden_4', nn.Linear(3*endmembers, endmembers + 1)),
                                            ('activation4', self.activation),
                                            ('batch_norm', nn.BatchNorm1d(endmembers + 1)),
                                            ('soft_thresholding', nn.Softplus(threshold=5)),   
                                            ('ASC_ANC', self.asc)
        ]))
        self.middle = nn.Sequential(OrderedDict([
                                            ('GaussianDroupout', self.gauss)
        ]))
        self.decoder_linear = nn.Sequential(OrderedDict([
                                            ('Linear1', nn.Linear(endmembers+1, num_bands, bias=False)),   # without bias and activation
        ]))
        self.nonlinear = nn.Sequential(OrderedDict([
                                            ('Linear2', nn.Linear(num_bands, 9*endmembers)),        
                                            ('activation6', nn.Sigmoid()),
                                            ('Linear3', nn.Linear(9*endmembers, 18*endmembers)),
                                            ('activation7', nn.Sigmoid()),
                                            ('Linear4', nn.Linear(18*endmembers, num_bands)),
                                            ('activation8', nn.Sigmoid()),
                                            ('Linear5', nn.Linear(num_bands, num_bands))
            ]))

    def init_decoder_linear(self):
        """"
        Initialize the decoder linear layer with water spectrum and target prior spectrum
        data : DATA MATRIX  B WAVELENGTHS x N PIXELS
        VCA output -- weight:  B x M+1 MATRIX OF ESTIMATED ENDMEMBERS
        load VCA initialization npy file
        """
        if not os.path.exists(fr'init_weight_VCA/{self.data_name}.npy'):
            print("The initial weight does not exist, generating the initial weight, this may take a while...")
            generate_init_weight(self.data_name)
        init_weight = np.load(fr'init_weight_VCA/{self.data_name}.npy')
        self.decoder_linear.Linear1.weight = nn.Parameter(torch.from_numpy(init_weight).float())

    def forward(self, spectral):
        encoded = self.encoder(spectral)
        temp = self.middle(encoded)
        linear_out = self.decoder_linear(temp)
        if self.is_nonlinear:
            decoded = self.nonlinear(linear_out)
            decoded = linear_out + decoded
            return encoded, decoded
        else:
            return encoded, linear_out
        
def normalize(data):
    data = (data - torch.min(data).item()) / (torch.max(data).item() - torch.min(data).item() + 1e-8)
    return data

def train(args, seed):
    seed_torch(seed)
    matData = open_file(fr'dataset/{args.data_name}/data.mat')
    data = matData['data']
    prior = matData['target']
    x_dims, y_dims, num_bands = data.shape
    dataset = RiverDataset(data, args.data_name, is_pseudo=False)

    best_auc = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.pseudo_data:
        pseudo_dataset = RiverDataset(data, args.data_name, is_pseudo=True)
        pseudo_batch_size = math.ceil(pseudo_dataset.data.shape[0] / math.ceil(x_dims * y_dims / args.batch_size))-1
        pseudo_train_loader = DataLoader(dataset=pseudo_dataset, batch_size=pseudo_batch_size, shuffle=False)

    model = HyperspecAE(data_name=args.data_name, num_bands=args.num_bands, endmembers=args.endmembers, activation=args.activation, is_nonlinear=args.is_nonlinear).to(device)
    # Initialize linear decoder weight with VCA
    model.init_decoder_linear()
    decoder_linear1_params = list(map(id, model.decoder_linear.Linear1.parameters()))
    other_params = filter(lambda p: id(p) not in decoder_linear1_params, model.parameters())
    optimizer = torch.optim.Adam([
                                {'params': other_params}, 
                                {'params': model.decoder_linear.Linear1.parameters(), 
                                 'lr': args.learning_rate*0.1,    # fine-tune
                                 'weight_decay': 0} 
    ], lr=args.learning_rate, weight_decay=args.weight_decay) 
    model.to(device)
    Loss = []
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(1, args.epochs+1)):
        model.train()
        iterator = iter(train_loader)
        if args.pseudo_data:
            iterator_pseudo = iter(pseudo_train_loader)
        epoch_loss = 0
        for i in range(len(iterator)):
            batch = next(iterator)
            batch = batch.to(device)

            optimizer.zero_grad()

            if args.pseudo_data:
                batch_pseudo = next(iterator_pseudo)
                batch_pseudo = batch_pseudo.to(device)
                prior_batch = torch.tensor(np.tile(prior, (batch_pseudo.shape[0], 1))).to('cuda')
                pseudo_abundance = torch.tensor(np.tile(np.random.rand(batch_pseudo.shape[0])*0.4+0.2, (args.num_bands, 1)).T).to('cuda')   # 0.2 - 0.6  alpha
                pseudo_label = torch.ones_like(pseudo_abundance[:, 0])
                pseudo_batch = pseudo_abundance * prior_batch + (1 - pseudo_abundance) * batch_pseudo   # target prior spectrum with water spectrum
                pseudo_batch = model.nonlinear(pseudo_batch.float())
                pseudo_enc_out, _ = model(pseudo_batch)
                pseudo_data_loss = criterion(pseudo_enc_out[:, args.endmembers], pseudo_label-0.05)  # soft label
            
            enc_out, dec_out = model(batch.float())

            if args.objective == 'SAD':
                reconstruction_loss = torch.sum(SAD()(dec_out, batch.float())) / batch.shape[0]
            elif args.objective == 'SID':
                reconstruction_loss = torch.sum(SID()(dec_out, batch.float())) / batch.shape[0]
            elif args.objective == 'MSE':
                reconstruction_loss = args.l11 * nn.MSELoss()(dec_out, batch.float())  
            else:   # The downward trend of the two is different and it is not recommended to mix
                reconstruction_loss = args.l11 * nn.MSELoss()(dec_out, batch.float()) + torch.sum(SAD()(dec_out, batch.float())) / batch.shape[0]     
            # The inverse of the variance of pixel-corresponding endmember abundances is used as a regularization term.
            abundance_regularization = torch.log10(1 / (torch.var(enc_out, dim=1, unbiased=False) + 1e-8)).sum()

            if args.pseudo_data:
                loss = reconstruction_loss + args.l1 * abundance_regularization + args.l2 * pseudo_data_loss
            else:
                loss = reconstruction_loss + args.l1 * abundance_regularization
            loss.backward()
            optimizer.step()

            linear_layer = model.decoder_linear
            for name, param in linear_layer.named_parameters():
                if name == 'Linear1.weight':
                    # Constraining weights to be greater than 0.
                    param.data.clamp_(0, 100000)   # _max take a larger value

            with torch.no_grad():
                epoch_loss += loss.item()

        with torch.no_grad():
            epoch_loss /= len(iterator)
            Loss.append(epoch_loss)

        model.eval()
        if not os.path.exists(fr'{args.save_path}/{args.data_name}'):
            os.makedirs(fr'{args.save_path}/{args.data_name}')
        with torch.no_grad():
            enc_out = torch.zeros((x_dims*y_dims, args.endmembers + 1)).to(device)
            batch_index = 0
            for batch_ in train_loader:
                batch_ = batch_.to(device=device)
                en_temp, _ = model(batch_.float())
                start_index = batch_index * args.batch_size
                end_index = start_index + batch_.size(0)
                enc_out[start_index:end_index] = en_temp
                batch_index += 1

            epoch_detect_result = enc_out.detach().cpu().squeeze().numpy().T[args.endmembers].reshape(x_dims, y_dims)
            epoch_auc = get_epoch_AUC(epoch_detect_result, seed, is_show=False)
            if epoch > 3:  # At least iterate 3 epoch
                if epoch_auc > best_auc:
                    best_auc = epoch_auc
                    torch.save(model.state_dict(), '{}/{}/NUN-UTD_best_{}.pth'.format(args.save_path, args.data_name, seed))
        tqdm.write('Epoch [{}/{}], Loss: {:.8f}, Best AUC: {:.4f}'.format(epoch, args.epochs, epoch_loss, best_auc))
    
    plt.plot(range(1, args.epochs+1), Loss, label='Loss')
    plt.xlabel('Epoch')
    plt.title('Loss Curve')
    plt.savefig(f'{args.save_path}/{args.data_name}/loss_curve_{seed}.png')
    plt.clf()
    plt.close()

def get_target_abundance(args, seed):  # test stage
    seed_torch(seed)
    dataset = open_file(fr'dataset/{args.data_name}/data.mat')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HyperspecAE(data_name=args.data_name, num_bands=args.num_bands, endmembers=args.endmembers, activation=args.activation, is_nonlinear=args.is_nonlinear).to(device)
    model.load_state_dict(torch.load(fr'{args.save_path}/{args.data_name}/NUN-UTD_best_'+str(seed)+'.pth'), strict=False)

    data = dataset['data']
    gt = dataset['gt']
    hsi = torch.tensor(data.reshape(-1, args.num_bands)).float().to('cuda')
    model.eval()
    with torch.no_grad():
        enc_out, _ = model(hsi)
        detect_result = enc_out.detach().cpu().squeeze().numpy().T[-1].reshape(data.shape[0], data.shape[1])
        auc = get_epoch_AUC(detect_result, seed, is_show=True)
        np.save(f'{args.save_path}/{args.data_name}/detect_result_{seed}_{auc:.4f}.npy', detect_result)

        after_threshold = threshold.threshold(detect_result, 0.1)
        plt.imshow(after_threshold)
        plt.axis('off')
        plt.savefig(f'{args.save_path}/{args.data_name}/detect_result_{seed}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

def test(seed):
    args = get_args()
    get_target_abundance(args, seed)

def main(seed):
    args = get_args()
    print(args)  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    train(args, seed)

if __name__ == '__main__':
    seed_list = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    args = get_args()
    for idx, seed in enumerate(seed_list):
        print("[%d / %d Random seed (%d) start training]" %(idx+1, len(seed_list), seed_list[idx]))
        main(seed)
        print("[%d / %d Random seed (%d) training completed]" %(idx+1, len(seed_list), seed_list[idx]))
        test(seed)
        torch.cuda.empty_cache()
