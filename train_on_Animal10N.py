import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
import torch.optim as optim
from utils.learning import *
from utils.vit_wrapper import  vit_img_wrap
from utils.animal_data_utils import Animal10N_dataset
import torch.optim as optim
from utils.learning import *
from utils.pre_correction import *
from utils.ws_augmentation import *
from utils.directional_diffusion_model import * 
from utils.log_config import setup_logger
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)

# Main training function (diffusion model, training, validation, test sets, model save path, command line arguments, encoder)
def train(diffusion_model, train_dataset, test_dataset, model_path, args, vit_fp, fp_dim):
    """
    Train the diffusion model with the given datasets and arguments.

    Parameters:
    - diffusion_model: The diffusion model to be trained.
    - train_dataset: The dataset used for training.
    - test_dataset: The dataset used for testing.
    - model_path: Path to save the trained model.
    - args: Command line arguments containing training parameters.
    - vit_fp: Whether to use precomputed feature embeddings.
    - fp_dim: Dimension of the feature embeddings.
    """
    # Extract configurations from the model and command line arguments, including training device, number of classes, total training epochs, k value in KNN, and warmup epochs.
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    num_models = diffusion_model.num_models
    n_epochs = args.nepoch
    batch_size = args.batch_size
    k = args.k
    warmup_epochs = args.warmup_epochs
    data_dir = os.path.join(os.getcwd(), 'data/Animal10N')
    noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)

    # Extract features
    print('data_dir:', data_dir)
    train_embed_dir = os.path.join(data_dir, f'fp_embed_train_animal')
    print('Doing pre-computing fp embeddings for weak and strong dataset')
    weak_embed, strong_embed = prepare_2_fp_x(fp_encoder, train_dataset, save_dir=train_embed_dir, device=device, fp_dim=fp_dim)
    weak_embed = weak_embed.to(device)
    strong_embed = strong_embed.to(device)
    
    # For testing data
    print('pre-computing fp embeddings for testing data')
    test_embed_dir = os.path.join(data_dir, f'fp_embed_test_animal.npy')
    test_embed = prepare_fp_x(diffusion_model.fp_encoder, test_dataset, test_embed_dir, device=device, fp_dim=fp_dim)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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

    # Train in a loop and record the highest accuracy to save the model
    max_accuracy = 0.0
    print('Directional Diffusion training start')
    for epoch in range(n_epochs):
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

                if vit_fp:
                    # Use precomputed feature embeddings
                    fp_embd_w = weak_embed[data_indices, :].to(device)
                    fp_embd_s = strong_embed[data_indices, :].to(device)
                else:
                    # Compute feature embeddings in real-time
                    fp_embd_w = diffusion_model.fp_encoder(x_batch_w.to(device))
                    fp_embd_s = diffusion_model.fp_encoder(x_batch_s.to(device))

                # Perform label sampling based on two views
                y_label_batch_w, y_label_batch_s, loss_weights_w, loss_weights_s, y_label_batch_n, gamma_batch = precorrect_labels_in_two_view(fp_embd_w=fp_embd_w, fp_embd_s=fp_embd_s, y_noisy=y_noisy, weak_embed=weak_embed, strong_embed=strong_embed, noisy_labels=noisy_labels, device=device, k=k, n_class=n_class, use_cosine_similarity=args.use_cos, to_single_label=args.to_single_label)

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

                # Single view training, only train weakly augmented images with precise label sampling
                
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
            test_acc = test(diffusion_model, test_loader, test_embed)
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
    """
    Test the diffusion model with the given test loader and embeddings.

    Parameters:
    - diffusion_model: The diffusion model to be tested.
    - test_loader: DataLoader for the test set.
    - test_embed: Precomputed feature embeddings for the test set.

    Returns:
    - acc: The accuracy of the model on the test set.
    """
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
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    # Training parameters
    parser.add_argument("--nepoch", default=200, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=200, help="batch_size", type=int)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    # Diffusion model hyperparameters
    parser.add_argument("--num_models", default=2, help="number of models", type=int)
    parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    parser.add_argument("--k", default=50, help="k neighbors for knn or cos", type=int)
    parser.add_argument("--loss_w", default=True, help="use weights for loss", action='store_false')
    parser.add_argument("--to_single_label", default=False, help="use single_label for label sampling", action='store_true')
    parser.add_argument("--one_view", default=False, help="use single view", action='store_true')
    parser.add_argument("--use_cos", default=True, help="use cos", action='store_false')
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet34', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)
    parser.add_argument("--objective", default='pred_res_noise', help="which type for diffusion (pred_res, pred_noise, pred_res_noise...)", type=str)
    # Large model hyperparameters
    parser.add_argument("--fp_encoder", default='ViT', help="which encoder for fp (SimCLR, Vit or ResNet)", type=str)
    parser.add_argument("--ViT_type", default='ViT-L/14', help="which encoder for Vit", type=str)
    # Storage path
    parser.add_argument("--gpu_devices", default=[0, 1, 2, 3], type=int, nargs='+', help="")
    parser.add_argument("--device", default=None, help="which cuda to use", type=str)
    parser.add_argument("--log_name", default='Animal10N.log', help="create your logs name", type=str)
    args = parser.parse_args()
    logger = setup_logger(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device is None:
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    n_class = 10

    # Prepare dataset directories
    data_dir = os.path.join(os.getcwd(), 'data/Animal10N')
    print('data_dir', data_dir)

    ANIMAL_MEAN = (0.6959, 0.6537, 0.6371)
    ANIMAL_STD = (0.3113, 0.3192, 0.3214)
    # Load datasets, where Animal10N_train is doubly augmented data
    train_dataset = Animal10N_dataset(root_dir=data_dir, mode='train')
    test_dataset = Animal10N_dataset(root_dir=data_dir, mode='test')

    # Load fp feature extractor
    fp_encoder = vit_img_wrap(args.ViT_type, args.device, center=ANIMAL_MEAN, std=ANIMAL_STD)
    fp_dim = fp_encoder.dim

    # Initialize the diffusion model
    model_path = './model/DLD_Animal10N.pt'
    base_model = DirectionalConditionalModel(n_steps = 1000, y_dim = n_class, fp_dim  = fp_dim, feature_dim = args.feature_dim, guidance = True, num_models = args.num_models, objective = args.objective, encoder_type=args.diff_encoder).to(args.device)

    diffusion_model = DirectionalDiffusion(model = base_model, fp_encoder=fp_encoder, num_models=args.num_models, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim,  device=args.device, feature_dim=args.feature_dim, encoder_type=args.diff_encoder, objective= args.objective, sampling_timesteps = args.ddim_n_step, condition = True, convert_to_ddim = False, sum_scale = 1., ddim_sampling_eta=0., beta_schedule = 'cosine')

    diffusion_model.fp_encoder.eval()

    # Load model weights
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)

    # DataParallel wrapper
    if args.device is None:
        print('using DataParallel')
        diffusion_model.model = nn.DataParallel(diffusion_model.model).to(device)
        diffusion_model.diffusion_encoder = nn.DataParallel(diffusion_model.diffusion_encoder).to(device)
        diffusion_model.fp_encoder = nn.DataParallel(fp_encoder).to(device)
    else:
        print('using single gpu')
        diffusion_model.to(device)

    # Train the diffusion model
    print(f'Training DLD using fp encoder: {args.fp_encoder} on: Animal-10N.')
    print(f'Model saving dir: {model_path}')
    train(diffusion_model, train_dataset=train_dataset, test_dataset=test_dataset, model_path=model_path, args=args, vit_fp=True, fp_dim=fp_dim)





