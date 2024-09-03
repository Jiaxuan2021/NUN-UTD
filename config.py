import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Underwater target detection based on autoencoder unmixing for Hyperspectral Image NUN-UTD')
    parser.add_argument('--data_name', type=str, default='ningxiang', help='The name of the dataset in dataset folder')    # Change this to the name of the dataset folder
    parser.add_argument('--num_bands', type=int, default=270, help='The number of bands')    # River_scene: 270, simulated_data: 150
    parser.add_argument('--endmembers', type=int, default=4, help='Number of endmembers')
    parser.add_argument('--epochs', type=int, default=250, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--l1', type=float, default=1e-4, help='abundance regularization weight in loss')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--l2', type=float, default=1.5, help='pseudo data loss weight in loss')
    parser.add_argument('--l11', type=float, default=10, help='MSE loss weight in loss')
    parser.add_argument('--is_nonlinear', type=bool, default=True, help='use nonlinear unmixing or not')
    parser.add_argument('--pseudo_data', type=bool, default=True, help='use pseudo data or not')
    parser.add_argument('--objective', type=str, default='MSE', help='MSE, SAD, SID, MSE_SAD')
    parser.add_argument('--save_path', type=str, default='result', help='The path to save the temp result')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='The activation function')
    parser.add_argument('--device', type=str, default='0', help='The device to run the model')
    return parser.parse_args()