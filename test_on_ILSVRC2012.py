import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
from utils.vit_wrapper import  vit_img_wrap
from utils.ILSVRC2012_data_utils import ILSVRC2012
import torch.optim as optim
from utils.learning import *
from utils.directional_diffusion_model import *
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
from utils.log_config import setup_logger

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)


def test(diffusion_model, test_loader, test_embed):

    if not torch.is_tensor(test_embed):
        test_embed = torch.tensor(test_embed).to(torch.float)

    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = 0.
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader),total=len(test_loader),desc=f'evaluating diff',ncols=100):
            [images, target, indicies] = data_batch[:3]
            target = target.to(device)
            fp_embed = test_embed[indicies, :].to(device)
            label_t_0 = diffusion_model.ddim_sample(x_batch=images,y_input=0,fp_x=fp_embed,last=True,stochastic=False).detach().cpu()
            correct = cnt_agree(label_t_0.detach().cpu(), target.cpu())
            correct_cnt += correct

    acc = 100 * correct_cnt / test_embed.shape[0]
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=300, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--num_workers", default=16, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=1, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=1024, help="feature_dim", type=int)
    parser.add_argument("--k", default=50, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet50_l', help="which encoder for diffusion", type=str)
    parser.add_argument("--gpu_devices", default=[0, 1, 2, 3], type=int, nargs='+', help="")
    parser.add_argument("--device", default=None, help="which cuda to use", type=str)
    parser.add_argument("--log_name",default='ILSVRC2012.log',help="create your logs name",type=str)
    args = parser.parse_args()
    logger = setup_logger(args)

    if args.device is None:
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    n_class = 50

    # load datasets WebVsison
    ILSVRC2012_dir = os.path.join(os.getcwd(), 'data/ILSVRC2012')
    print('data_dir', ILSVRC2012_dir)
    val_dataset = ILSVRC2012(data_root=ILSVRC2012_dir)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=args.num_workers)

    # initialize diffusion model
    fp_encoder = vit_img_wrap('ViT-L/14', device, center=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    fp_dim = fp_encoder.dim
    model_path = './model/DLD_Webvision.pt'
    base_model = DirectionalConditionalModel(n_steps = 1000, y_dim = n_class, fp_dim  = fp_dim, feature_dim = args.feature_dim, guidance = True, num_models = args.num_models, objective = args.objective, encoder_type=args.diff_encoder).to(args.device)

    diffusion_model = DirectionalDiffusion(model = base_model, fp_encoder=fp_encoder, num_models=args.num_models, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim,  device=args.device, feature_dim=args.feature_dim, encoder_type=args.diff_encoder, objective= args.objective, sampling_timesteps = args.ddim_n_step, condition = True, convert_to_ddim = False, sum_scale = 1., ddim_sampling_eta=0., beta_schedule = 'cosine')
    state_dict = torch.load(model_path, map_location=torch.device(device))
    diffusion_model.load_diffusion_net(state_dict)
    diffusion_model.fp_encoder.eval()

    # DataParallel wrapper
    if args.device is None:
        print('using DataParallel')
        diffusion_model.model = nn.DataParallel(diffusion_model.model).to(device)
        diffusion_model.fp_encoder = nn.DataParallel(fp_encoder).to(device)
    else:
        print('using single gpu')
        diffusion_model.to(device)

    print('pre-computing fp embeddings for validation data for ILSVRC2012')
    val_embed_dir = os.path.join(ILSVRC2012_dir, f'fp_embed_val_ILSVRC2012.npy')
    val_embed = prepare_fp_x(diffusion_model.fp_encoder, val_dataset, val_embed_dir, device=device,
                             fp_dim=fp_dim, batch_size=200)

    max_accuracy = test(diffusion_model, val_loader, val_embed)
    print('test ILSVRC2012 accuracy:', max_accuracy)
    logger.info(f"test ILSVRC2012 accuracy: {max_accuracy:.2f}%")
