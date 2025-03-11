import numpy as np
import argparse
from tqdm import tqdm
from utils.ema import EMA
from utils.vit_wrapper import  vit_img_wrap
from utils.pre_correction import *
from utils.webvision_data_utils import *
import torch.optim as optim
from utils.learning import *
from utils.directional_diffusion_model import *
import torch.multiprocessing
from utils.log_config import setup_logger
torch.multiprocessing.set_sharing_strategy('file_system')
torch.manual_seed(111)
torch.cuda.manual_seed(111)
np.random.seed(111)
random.seed(111)

def train(diffusion_model, val_loader, device, model_path, args, data_dir):
    """
    Train the diffusion model with the given datasets and arguments.

    Parameters:
    - diffusion_model: The diffusion model to be trained.
    - val_loader: DataLoader for the validation set.
    - device: Device to be used for training (CPU or GPU).
    - model_path: Directory to save the trained model.
    - args: Command line arguments containing training parameters.
    - data_dir: Directory where the data is stored.
    """
    print(f'Use loss weights: {args.loss_w}, Use Single label: {args.to_single_label}, Use One view: {args.one_view}')

    k = args.k
    n_epochs = args.nepoch
    n_class = 50
    num_models = diffusion_model.num_models
    n_epochs = args.nepoch
    batch_size = args.batch_size
    warmup_epochs = args.warmup_epochs

    val_embed = np.load(os.path.join(data_dir, 'fp_embed_val_webvision.npy'))
    weak_embed = torch.tensor(np.load(os.path.join(data_dir, 'fp_embed_train_webvision_weak.npy'))).to(device)
    strong_embed = torch.tensor(np.load(os.path.join(data_dir, 'fp_embed_train_webvision_strong.npy'))).to(device)
    train_labels = torch.tensor(np.load(os.path.join(data_dir, 'train_labels_webvision.npy'))).to(device)

    # Optimizer settings for dual networks
    if diffusion_model.num_models == 1:
        optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
        ema_helper = EMA(mu=0.999)
        ema_helper.register(diffusion_model.model)
    elif diffusion_model.num_models == 2:
        optimizer_res = optim.Adam(diffusion_model.model0.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
        optimizer_noise = optim.Adam(diffusion_model.model1.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
        # Initialize EMA helper and register model parameters to smooth model parameter updates during training to improve model stability and performance.
        ema_helper_res = EMA(mu=0.999)
        ema_helper_noise = EMA(mu=0.999)
        ema_helper_res.register(diffusion_model.model0)
        ema_helper_noise.register(diffusion_model.model1)

    diffusion_loss = nn.MSELoss(reduction='none')

    max_accuracy = 0

    print('Directional Diffusion training start')
    for epoch in range(n_epochs):
        train_dataset = WebVision_WS(data_root=data_dir, split='train', balance=True, randomize=True, cls_size=500, transform='train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=init_fn, drop_last=True)

        if diffusion_model.num_models == 1:
            diffusion_model.model.train()
        else:
            diffusion_model.model0.train()
            diffusion_model.model1.train()

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch_w, x_batch_s, y_batch, data_indices] = data_batch[:4]
                x_batch_w = x_batch_w.to(device)
                x_batch_s = x_batch_s.to(device)
                y_noisy = y_batch.to(device)

                # Use precomputed feature embeddings
                fp_embd_w = weak_embed[data_indices, :].to(device)
                fp_embd_s = strong_embed[data_indices, :].to(device)


                # pre-correct labels based on two views
                y_label_batch_w, y_label_batch_s, loss_weights_w, loss_weights_s, y_label_batch_n, gamma_batch = precorrect_labels_in_two_view(fp_embd_w=fp_embd_w, fp_embd_s=fp_embd_s, y_noisy=y_noisy, weak_embed=weak_embed, strong_embed=strong_embed, noisy_labels=train_labels, k=k, n_class=n_class, use_cosine_similarity=args.use_cos, to_single_label=args.to_single_label)

                if args.one_view:
                    x_batch = x_batch_w
                else:
                    x_batch = (1 - gamma_batch.view(-1, 1, 1, 1)) * x_batch_w + gamma_batch.view(-1, 1, 1, 1) * x_batch_s

                # Check if the labels are one-hot encoded, if not, convert them to vectors
                if len(y_label_batch_w.shape) == 1:
                    y_one_hot_batch_w = cast_label_to_one_hot_and_prototype(y_label_batch_w.to(torch.int64), n_class=n_class)
                    y_one_hot_batch_s = cast_label_to_one_hot_and_prototype(y_label_batch_s.to(torch.int64), n_class=n_class)
                else:
                    y_one_hot_batch_w = y_label_batch_w
                    y_one_hot_batch_s = y_label_batch_s

                y_0_batch_w = y_one_hot_batch_w.to(device)
                y_0_batch_s = y_one_hot_batch_s.to(device)
                y_zeros = torch.zeros_like(y_0_batch_w)
                y_n_batch = y_label_batch_n.to(device)

                #Adjust learning rate
                if diffusion_model.num_models == 1:
                    adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=n_epochs,
                    lr_input=1e-3)
                else:
                    adjust_learning_rate(optimizer_res, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=n_epochs, lr_input=1e-3)
                    adjust_learning_rate(optimizer_noise, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=n_epochs, lr_input=1e-3)

                # Sampling t for symmetric
                n = x_batch.size(0)

                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1, )).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                ## Sampling t for random
                # t = torch.randint(0, diffusion_model.num_timesteps, (n,), device=device).long()


                print(y_zeros.device, y_0_batch_w.device, x_batch.device, t.device, fp_embd_w.device)
                output, e = diffusion_model.forward_t(y_zeros, y_0_batch_w, x_batch, t, fp_embd_w)

                if diffusion_model.objective == 'pred_res_noise':
                    # Calculate L_res and L_noise losses
                    L_res = diffusion_loss(output[0], e[0])
                    L_noise = diffusion_loss(output[1], e[1])

                    # Apply optional loss weighting
                    weighted_L_res = torch.matmul(loss_weights_w, L_res) if args.loss_w else L_res
                    weighted_L_noise = torch.matmul(loss_weights_w, L_noise) if args.loss_w else L_noise

                    # Mean loss calculation
                    l_res_loss = torch.mean(weighted_L_res)
                    l_noise_loss = torch.mean(weighted_L_noise)

                    if diffusion_model.num_models == 2:
                        # Update model0 using l_res_loss
                        optimizer_res.zero_grad()
                        l_res_loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(diffusion_model.model0.parameters(), 1.0)
                        optimizer_res.step()
                        ema_helper_res.update(diffusion_model.model0)

                        # Update model1 using l_noise_loss
                        optimizer_noise.zero_grad()
                        l_noise_loss.backward()
                        torch.nn.utils.clip_grad_norm_(diffusion_model.model1.parameters(), 1.0)
                        optimizer_noise.step()
                        ema_helper_noise.update(diffusion_model.model1)

                        # Update progress bar
                        pbar.set_postfix({'res_loss': l_res_loss.item(), 'noise_loss': l_noise_loss.item()})

                    elif diffusion_model.num_models == 1:  # For single model, combine losses
                        loss = 0.1 * l_res_loss + 0.9 * l_noise_loss
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                        optimizer.step()
                        ema_helper.update(diffusion_model.model)
                        pbar.set_postfix({'loss': loss.item()})

                else:  # For objectives other than 'pred_res_noise'
                    # Calculate loss without separation
                    L_noise = diffusion_loss(output, e)
                    weighted_L_noise = torch.matmul(loss_weights_w, L_noise) if args.loss_w else L_noise
                    loss = torch.mean(weighted_L_noise)

                    # Perform single optimizer step for combined objective
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                    optimizer.step()
                    ema_helper.update(diffusion_model.model)
                    pbar.set_postfix({'loss': loss.item()})

        # Every epoch, perform validation, if the validation accuracy of the current epoch exceeds the previous highest accuracy, evaluate the model on the test set, and save the current best model parameters.
        if epoch >= warmup_epochs:
            test_acc = test(diffusion_model, val_loader, val_embed)
            logger.info(f"epoch: {epoch}, test accuracy: {test_acc:.2f}%")
            if test_acc > max_accuracy:
                # Save diffusion model
                print('Improved! Evaluate on testing set...')
                if diffusion_model.num_models == 1:
                    states = {
                        'model': diffusion_model.model.state_dict(),
                        'fp_encoder': diffusion_model.fp_encoder.state_dict(),
                    }
                else:
                    states = {
                        'model0': diffusion_model.model0.state_dict(),
                        'model1': diffusion_model.model1.state_dict(),
                        'fp_encoder': diffusion_model.fp_encoder.state_dict(),
                    }
                torch.save(states, model_path)
                message = (f"Model saved, update best accuracy at Epoch {epoch}, test acc: {test_acc}")
                logger.info(message)
                max_accuracy = max(max_accuracy, test_acc)

def test(diffusion_model, test_loader, test_embed):

    if not torch.is_tensor(test_embed):
        test_embed = torch.tensor(test_embed).to(torch.float)

    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = 0.
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'evaluating diff', ncols=100):
            [images, target, indicies] = data_batch[:3]
            target = target.to(device)
            fp_embed = test_embed[indicies, :].to(device)
            label_t_0 = diffusion_model.ddim_sample(x_batch = images, y_input = 0, fp_x=fp_embed, last=True, stochastic=False).detach().cpu()
            correct = cnt_agree(label_t_0.detach().cpu(), target.cpu())
            correct_cnt += correct

    acc = 100 * correct_cnt / test_embed.shape[0]
    return acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument("--nepoch", default=300, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--num_workers", default=16, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=1, help="warmup_epochs", type=int)
    parser.add_argument("--gpu_devices", default=[0, 1, 2, 3], type=int, nargs='+', help="")
    parser.add_argument("--device", default=None, help="which cuda to use", type=str)
    # Diffusion model hyperparameters
    parser.add_argument("--num_models", default=2, help="number of models", type=int)
    parser.add_argument("--feature_dim", default=1024, help="feature_dim", type=int)
    parser.add_argument("--k", default=50, help="k neighbors for knn", type=int)
    parser.add_argument("--loss_w", default=True, help="use weights for loss", action='store_false')
    parser.add_argument("--to_single_label", default=False, help="use single_label for label sampling", action='store_true')
    parser.add_argument("--one_view", default=False, help="use single view", action='store_true')
    parser.add_argument("--use_cos", default=True, help="use cos", action='store_false')
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet50_l', help="which encoder for diffusion", type=str)
    parser.add_argument("--objective", default='pred_res_noise', help="which type for diffusion (pred_res, pred_noise, pred_res_noise...)", type=str)
    # Storage path
    parser.add_argument("--log_name", default='WebVision.log', help="create your logs name", type=str)
    args = parser.parse_args()
    logger = setup_logger(args)

    if args.device is None:
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    n_class = 50

    # Load datasets WebVision
    webvision_dir = os.path.join(os.getcwd(), 'data/WebVision')
    print('data_dir', webvision_dir)

    train_dataset = WebVision_WS(data_root=webvision_dir, split='train', balance=False, randomize=False, cls_size=500, transform='train')
    train_labels = torch.tensor(train_dataset.targets).to(torch.long)
    np.save(os.path.join(webvision_dir, f'train_labels_webvision.npy'), train_labels)
    val_dataset = WebVision(data_root=webvision_dir, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=args.num_workers)

    # Initialize diffusion model
    fp_encoder = vit_img_wrap('ViT-L/14', device, center=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    fp_dim = fp_encoder.dim
    # Initialize the diffusion model
    model_path = './model/DLD_Webvision.pt'
    base_model = DirectionalConditionalModel(n_steps = 1000, y_dim = n_class, fp_dim  = fp_dim, feature_dim = args.feature_dim, guidance = True, num_models = args.num_models, objective = args.objective, encoder_type=args.diff_encoder).to(args.device)

    diffusion_model = DirectionalDiffusion(model = base_model, fp_encoder=fp_encoder, num_models=args.num_models, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim,  device=args.device, feature_dim=args.feature_dim, encoder_type=args.diff_encoder, objective= args.objective, sampling_timesteps = args.ddim_n_step, condition = True, convert_to_ddim = False, sum_scale = 1., ddim_sampling_eta=0., beta_schedule = 'cosine')
    diffusion_model.fp_encoder.eval()

    # Load model
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)

    # DataParallel wrapper
    if args.device is None:
        print('using DataParallel')
        if diffusion_model.num_models == 1:
            diffusion_model.model = nn.DataParallel(diffusion_model).to(device)
            diffusion_model.fp_encoder = nn.DataParallel(fp_encoder).to(device)
        else:
            diffusion_model.model0 = nn.DataParallel(diffusion_model).to(device)
            diffusion_model.fp_encoder = nn.DataParallel(fp_encoder).to(device)
    else:
        print('using single gpu')
        diffusion_model.to(device)

    # Pre-compute for fp embeddings on training data
    print('pre-computing fp embeddings for training data')
    train_embed_dir = os.path.join(webvision_dir, f'fp_embed_train_webvision')
    weak_embed, strong_embed = prepare_2_fp_x(diffusion_model.fp_encoder, train_dataset, train_embed_dir, device=device, fp_dim=fp_dim, batch_size=200)
    # For validation data
    print('pre-computing fp embeddings for validation data for webvision')
    val_embed_dir = os.path.join(webvision_dir, f'fp_embed_val_webvision.npy')
    val_embed = prepare_fp_x(diffusion_model.fp_encoder, val_dataset, val_embed_dir, device=device, fp_dim=fp_dim, batch_size=200)

    # Train the diffusion model
    train(diffusion_model, val_loader, device, model_path, args, data_dir=webvision_dir)
