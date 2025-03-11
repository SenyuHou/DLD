from collections import namedtuple
from functools import partial
import random
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.ResNet_for_32 as resnet_s
import utils.ResNet_for_224 as resnet_l

ModelPrediction = namedtuple(
    'ModelPrediction', ['pred_yd', 'pred_noise', 'pred_y0'])
# helpers functions

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t.to(a.device))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    assert (alphas.sum()-1).abs() < 1e-6

    return alphas*sum_scale

def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    """
    Define a function to set the beta schedule, supporting multiple scheduling strategies.

    Parameters:
    - schedule: The type of schedule to use ('linear', 'const', 'quad', 'jsd', 'sigmoid', 'cosine', 'cosine_anneal').
    - num_timesteps: The number of timesteps for the schedule.
    - start: The starting value of beta.
    - end: The ending value of beta.

    Returns:
    - betas: The beta values for each timestep.
    """
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) /
                 (math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return torch.tensor(betas, dtype=torch.float32)

class ConditionalLinear(nn.Module):
    """
    A conditional linear layer that combines input features with embedded time steps.
    """
    def __init__(self, num_in, num_out, n_steps):
        """
        Initialize the ConditionalLinear layer.

        Parameters:
        - num_in: Number of input features.
        - num_out: Number of output features.
        - n_steps: Number of time steps for embedding.
        """
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out

class ConditionalModel(nn.Module):
    """
    A conditional model that uses a U-Net structure with conditional linear layers,
    with an internal diffusion encoder to process input x.
    """
    def __init__(self, n_steps, y_dim=10, fp_dim=128, feature_dim=None, guidance=True, num_models=2, encoder_type='resnet34',objective='pred_res_noise'):
        """
        Initialize the ConditionalModel.

        Parameters:
        - n_steps: Number of time steps for embedding.
        - y_dim: Dimension of the labels.
        - fp_dim: Dimension of the feature embeddings.
        - feature_dim: Dimension of the features.
        - guidance: Whether to use guidance.
        - encoder_type: Type of diffusion encoder to use (e.g., 'resnet50').
        """
        super(ConditionalModel, self).__init__()
        n_steps = n_steps + 1
        self.y_dim = y_dim
        self.guidance = guidance
        self.num_models = num_models
        self.feature_dim = feature_dim
        self.encoder_type = encoder_type
        self.objective = objective

        # Instantiate diffusion encoder within ConditionalModel
        self.diffusion_encoder = self._get_encoder(encoder_type, feature_dim)
        self.norm = nn.BatchNorm1d(feature_dim)

        # U-Net structure
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim + fp_dim, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)

        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        
        # Output layer(s)
        if num_models == 1:
            # Single output layer for both res and noise (doubling the output channels)
            if self.objective == 'pred_res_noise':
                self.lin4 = nn.Linear(feature_dim, y_dim * 2)
            elif self.objective == 'pred_noise':
                self.lin4 = nn.Linear(feature_dim, y_dim)
        else:
            self.lin4 = nn.Linear(feature_dim, y_dim)

    def _get_encoder(self, encoder_type, feature_dim):
        """
        Helper function to instantiate the encoder based on the encoder_type.
        """
        if encoder_type == 'resnet34':
            return resnet_s.resnet34(num_input_channels=3, num_classes=feature_dim)
        elif encoder_type == 'resnet18':
            return resnet_s.resnet18(num_input_channels=3, num_classes=feature_dim)
        elif encoder_type == 'resnet50':
            return resnet_s.resnet50(num_input_channels=3, num_classes=feature_dim)
        elif encoder_type == 'resnet18_l':
            return resnet_l.resnet18(num_classes=feature_dim, pretrained=True)
        elif encoder_type == 'resnet34_l':
            return resnet_l.resnet34(num_classes=feature_dim, pretrained=True)
        elif encoder_type == 'resnet50_l':
            return resnet_l.resnet50(num_classes=feature_dim, pretrained=True)
        else:
            raise ValueError("ResNet type should be one of [18, 34, 50]")

    def forward(self, x, y, t, fp_x=None):
        """
        Forward pass for the ConditionalModel.

        Parameters:
        - x: Raw input features (images).
        - y: Labels.
        - t: Time step.
        - fp_x: Optional feature embeddings.

        Returns:
        - The output of the model.
        """
        # Process input x through diffusion_encoder to get embedded features
        x_embed = self.diffusion_encoder(x)
        x_embed = self.norm(x_embed)

        if self.guidance:
            y = torch.cat([y, fp_x], dim=-1)

        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x_embed * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        
        if self.num_models == 1:
            if self.objective == 'pred_res_noise':
                # Return two outputs: residual and noise
                combined_output = self.lin4(y)
                pred_res, pred_noise = torch.split(combined_output, self.y_dim, dim=-1)
                return pred_res, pred_noise
            elif self.objective == 'pred_noise':
                return self.lin4(y)
        elif self.num_models == 2:
            # Return a single output (residual or noise)
            return self.lin4(y)
        
class DirectionalConditionalModel(nn.Module):
    """
    A ResConditionalModel that wraps around one or two ConditionalModel(s).
    Supports different objectives such as predicting residuals or noise.
    """
    def __init__(self, n_steps, y_dim=10, fp_dim=128, feature_dim=None, guidance=True, num_models=2, encoder_type = 'resnet34', objective='pred_res_noise'):
        """
        Initialize the ResConditionalModel.

        Parameters:
        - n_steps: Number of time steps for embedding.
        - y_dim: Dimension of the labels.
        - fp_dim: Dimension of the feature embeddings.
        - feature_dim: Dimension of the features.
        - guidance: Whether to use guidance.
        - num_models: Number of ConditionalModels to use (1 or 2).
        - objective: Defines the task ('pred_res_noise', 'pred_noise', 'pred_res').
        """
        super().__init__()
        self.n_steps = n_steps
        self.y_dim = y_dim
        self.fp_dim = fp_dim
        self.feature = feature_dim
        self.guidance = guidance
        self.num_models = num_models
        self.objective = objective
        self.encoder_type = encoder_type

        # Define one or two ConditionalModel instances
        if self.num_models == 1:
            self.model = ConditionalModel(n_steps, y_dim, fp_dim, feature_dim, guidance, num_models, encoder_type,objective)
        
        elif self.num_models == 2:
            self.model0 = ConditionalModel(n_steps, y_dim, fp_dim, feature_dim, guidance, num_models, encoder_type,objective)
            self.model1 = ConditionalModel(n_steps, y_dim, fp_dim, feature_dim, guidance, num_models, encoder_type,objective)

        else:
            print("num_models must be 1 or 2 !")

    def forward(self, x, y, t, fp_x=None):
        """
        Forward pass for the ResConditionalModel.

        Parameters:
        - x_embed: Resnet Embedded input features.
        - y: Labels.
        - t: Time step (tuple of two time steps if using two models).
        - fp_x: Optional feature embeddings.

        Returns:
        - Outputs of one or both models based on the objective.
        """
        if self.num_models == 2:
            t0 = t1 = t
            # Depending on the objective, return the appropriate output(s)
            if self.objective == 'pred_res_noise':
                output_res = self.model0(x, y, t0, fp_x)
                output_noise = self.model1(x, y, t1, fp_x)
                return output_res, output_noise
            elif self.objective == 'pred_res':
                output_res = self.model0(x, y, t0, fp_x)
                return output_res, 0
            elif self.objective == 'pred_noise':
                output_noise = self.model1(x, y, t1, fp_x)
                return 0, output_noise
        else:
            # Single model case
            if self.objective == 'pred_res_noise':
                return self.model(x, y, t, fp_x)
            elif self.objective == 'pred_res':
                return self.model(x, y, t, fp_x)
            elif self.objective == 'pred_noise':
                return self.model(x, y, t, fp_x)
            
class DirectionalDiffusion(nn.Module):
    """
    A diffusion model that uses a conditional model and a feature encoder.
    """
    def __init__(self, 
                 model,
                 fp_encoder,
                 num_models=2, 
                 num_timesteps=1000, 
                 n_class=10, 
                 fp_dim=512, 
                 device='cuda', 
                 feature_dim=2048, 
                 objective='pred_res_noise',
                 encoder_type='resnet34',
                 ddim_sampling_eta=0.,
                 condition=True,
                 sum_scale=1.,
                 sampling_timesteps=10,
                 convert_to_ddim=False,
                 beta_schedule='cosine'):
        
        super().__init__()
        self.model = model
        self.device = device
        self.num_models = num_models
        self.num_timesteps = num_timesteps
        self.n_class = n_class
        self.y_dim = n_class
        self.feature_dim = feature_dim
        self.objective = objective
        self.condition = condition
        self.fp_dim = fp_dim
        self.encoder_type = encoder_type
        self.fp_encoder = fp_encoder.eval()
        self.sum_scale = sum_scale
        self.sampling_timesteps = sampling_timesteps 
        self.convert_to_ddim = convert_to_ddim
        self.beta_schedule = beta_schedule

        if self.num_models == 2:
            self.model0 = self.model.model0
            self.model1 = self.model.model1
        else:
            self.model =  self.model.model

        if self.convert_to_ddim:
            beta_schedule = "squaredcos_cap_v2"
            beta_start = 0.0001
            beta_end = 0.02
            timesteps = 1000
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas = alphas.float().to(self.device)
            betas = betas.float().to(self.device)
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = alphas[1]
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = betas2[1]

        else:
            alphas = gen_coefficients(self.num_timesteps, schedule="average", ratio=1)
            betas2 = gen_coefficients(self.num_timesteps, schedule="average", sum_scale=self.sum_scale, ratio=1)
            alphas = alphas.float().to(self.device)
            betas2 = betas2.float().to(self.device)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)

        
        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2 * betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        
        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def q_sample(self, y_0, y_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (y_0 + extract(self.alphas_cumsum, t, y_0.shape) * y_res +
            extract(self.betas_cumsum, t, y_0.shape) * noise
        )

    def predict_noise_from_res(self, y_t, t, y_input, y_pred_res):
        return (
            (y_t-y_input-(extract(self.alphas_cumsum, t, y_t.shape)-1)
             * y_pred_res)/extract(self.betas_cumsum, t, y_t.shape)
        )
    
    def predict_start_from_yinput_noise(self, y_t, t, y_input, noise):
        return (
            (y_t-extract(self.alphas_cumsum, t, y_t.shape)*y_input -
             extract(self.betas_cumsum, t, y_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, y_t.shape)
        )

    def predict_start_from_res_noise(self, y_t, t, y_res, noise):
        return (
            y_t-extract(self.alphas_cumsum, t, y_t.shape) * y_res -
            extract(self.betas_cumsum, t, y_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, y_res, noise, y_t, t):
        return (y_t-extract(self.alphas, t, y_t.shape) * y_res -
                (extract(self.betas2, t, y_t.shape)/extract(self.betas_cumsum, t, y_t.shape)) * noise)
    
    def q_posterior(self, y_pred_res, y_0, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_t +
            extract(self.posterior_mean_coef2, t, y_t.shape) *y_pred_res +
            extract(self.posterior_mean_coef3, t, y_t.shape) * y_0
        )
        posterior_variance = extract(self.posterior_variance, t, y_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, y_input, y, x_batch, fp_x, t, clip_denoised=True):
        
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else identity

        y_in = y_input

        model_output = self.model(x_batch, y, t, fp_x)

        if self.objective == 'pred_res_noise':            
            pred_res = model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            pred_y0 = self.predict_start_from_res_noise(
                y, t, pred_res, pred_noise)
            pred_y0 = maybe_clip(pred_y0)
        
        elif self.objective == "pred_noise":
            pred_noise = model_output
            pred_y0 = self.predict_start_from_yinput_noise(
                y, t, y_in, pred_noise)
            pred_y0 = maybe_clip(pred_y0)
            pred_res = y_in - pred_y0
            pred_res = maybe_clip(pred_res)

        return ModelPrediction(pred_res, pred_noise, pred_y0)
     
    @torch.no_grad()
    def ddim_sample(self, x_batch, y_input=0, fp_x=None, last=True, stochastic=False):

        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        x_batch = x_batch.to(self.device)

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        y_T = stochastic * torch.randn([x_batch.shape[0], self.n_class], device=device)
        if isinstance(y_input, int):
            y_input = torch.full((x_batch.shape[0], self.n_class), y_input, dtype=torch.float32, device=device)

        pred_y0 = None

        if not last:
            y_list = []

        y_t = y_T
        for time, time_next in time_pairs:
            time_cond = torch.full(
                (x_batch.shape[0],), time, device=device, dtype=torch.long)
            
            if fp_x is None:
                fp_x = self.fp_encoder(x_batch)

            preds = self.model_predictions(y_input, y_t, x_batch, fp_x, time_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            pred_y0 = preds.pred_y0

            if time_next < 0:
                y_t = pred_y0
                if not last:
                    y_list.append(y_t)
                continue

            alpha_cumsum = self.alphas_cumsum[time] #alpha_t_bar
            alpha_cumsum_next = self.alphas_cumsum[time_next] #alpha_t-1_bar
            alpha = alpha_cumsum-alpha_cumsum_next #alpha_t-1

            betas2_cumsum = self.betas2_cumsum[time] #betas2_t_bar
            betas2_cumsum_next = self.betas2_cumsum[time_next] #betas2_t-1_bar
            betas2 = betas2_cumsum-betas2_cumsum_next #betas2_t
            betas = betas2.sqrt() #betas_t
            betas_cumsum = self.betas_cumsum[time] #betas_t_bar
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            noise = torch.randn_like(y_t) if eta != 0 else 0

            if objective == "pred_res_noise":
                y_t = y_t - alpha * pred_res - \
                    (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
                    pred_noise + sigma2.sqrt()*noise
                
            if not last:
                y_list.append(y_t)    

        if not last:
            return y_list
        else:
            return y_t
    
    def forward_t(self, y_input, y_0, x_batch, t, fp_x,  noise=None):

        noise = default(noise, lambda: torch.randn_like(y_0))

        y_start = y_0
        y_in = y_input
        y_res = y_in - y_start

        y_t_batch = self.q_sample(y_0 = y_0, y_res = y_res, t = t, noise=noise)

        x_batch = x_batch.to(self.device)

        model_out = self.model(x_batch, y_t_batch, t, fp_x)

        target = []
        if self.objective == 'pred_res_noise':
            target.append(y_res)
            target.append(noise)
        
        elif self.objective == "pred_noise":
            target = noise

        return model_out, target    
    
    def load_diffusion_net(self, net_state_dicts):
        """
        Load the state dictionaries for the diffusion model components.

        Parameters:
        - net_state_dicts: Dictionary of state dictionaries for the models, encoders, and optional feature encoder.
        """
        self.model0.load_state_dict(net_state_dicts['model0'])
        self.model1.load_state_dict(net_state_dicts['model1'])

        if 'fp_encoder' in net_state_dicts:
            self.fp_encoder.load_state_dict(net_state_dicts['fp_encoder'])