import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.models.diffusion.utils import extract, sum_except_batch, log_categorical
from src.models.diffusion.utils import log_add_exp, sliced_logsumexp, index_to_log_onehot

DEFAULT_NUM_TIMESTEPS = 1000

def sample_timesteps(num_timesteps, batch_size):
    timesteps = torch.randint(0, num_timesteps, (batch_size,)).long()
    return timesteps

def linear_beta_scheduler(num_timesteps, start=0.001, end=0.2):
    scale = DEFAULT_NUM_TIMESTEPS / num_timesteps
    beta_start = scale * start
    beta_end = scale * end
    beta = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    return beta

def multinomial_kl_div(log_dist_1, log_dist_2):
    kl_div = (log_dist_1.exp() * (log_dist_1 - log_dist_2)).sum(dim=1)
    return kl_div

def ohe_to_categories(ohe, K):
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i + 1]].argmax(dim=1))
    return torch.stack(res, dim=1)

class BaseDDPM(nn.Module):
    """Base class for Diffusion models."""
    def __init__(self):
        super().__init__()

    def build(self, beta, num_classes, num_numerical_features, denoise_fn):
        """"Precompute parameters given beta and move to device."""
        alpha = 1. - beta
        cumprod_alpha = np.cumprod(alpha, axis=0)
        cumprod_alpha_prev = np.concatenate([np.ones((1,)), cumprod_alpha[:-1]], axis=0)
        
        # Precomputed parameters
        params = {}
        params['alpha'] = alpha
        params['beta'] = beta
        params['cumprod_alpha'] = cumprod_alpha
        params['cumprod_alpha_prev'] = cumprod_alpha_prev
        params['sqrt_cumprod_alpha'] = np.sqrt(cumprod_alpha)
        params['one_min_cumprod_alpha'] = 1. - cumprod_alpha
        params['gauss_posterior_mean_coef1'] = beta * np.sqrt(cumprod_alpha_prev) / (1.0 - cumprod_alpha)
        params['gauss_posterior_mean_coef2'] = (1 - cumprod_alpha_prev) * np.sqrt(alpha) / (1.0 - cumprod_alpha)
        params['gauss_posterior_var'] = beta * (1.0 - cumprod_alpha_prev) / (1.0 - cumprod_alpha)
        params['sqrt_one_min_cumprod_alpha'] = np.sqrt(1.0 - cumprod_alpha)
        params["log_alpha"] = np.log(alpha)
        params["log_one_min_alpha"] = np.log(1. - alpha)
        params["log_cumprod_alpha"] = np.log(cumprod_alpha)
        params["log_one_min_cumprod_alpha"] = np.log(1. - cumprod_alpha)
        for key, value in params.items():
            params[key] = torch.from_numpy(value).float()
            self.register_buffer(key, params[key])
        
        # Denoising function
        self.num_timesteps = len(beta)
        self.denoise_fn = denoise_fn
        
        # Relevant for tabular data
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        )
        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets))

class TabDDPM(BaseDDPM):
    """Container for Tabular Diffusion models."""
    def __init__(self, beta, num_classes, num_numerical_features, denoise_fn):
        super().__init__()
        self.build(beta, num_classes, num_numerical_features, denoise_fn)

    def gauss_q_prior_mean_var(self, x_0, t):
        mean = extract(self.sqrt_cumprod_alpha, t, x_0.shape) * x_0
        var = extract(self.one_min_cumprod_alpha, t, x_0.shape)
        return mean, var
    
    def gauss_q_sample(self, x_0, t, eps):
        mean, var = self.gauss_q_prior_mean_var(x_0, t)
        std = torch.sqrt(var)
        x_t = mean + std * eps
        return x_t
    
    def gauss_q_posterior_mean_var(self, x_0, x_t, t):
        posterior_mean_part1 = extract(self.gauss_posterior_mean_coef1, t, x_0.shape) * x_0
        posterior_mean_part2 = extract(self.gauss_posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_mean = posterior_mean_part1 + posterior_mean_part2
        posterior_var = extract(self.gauss_posterior_var, t, x_t.shape)
        return posterior_mean, posterior_var
    
    def gauss_x_0_from_eps(self, x_t, t, eps):
        x_0 = x_t - extract(self.sqrt_one_min_cumprod_alpha, t, eps.shape) * eps
        x_0 = x_0 / extract(self.sqrt_cumprod_alpha, t, eps.shape)
        return x_0
    
    def gauss_p_estimated_mean_var(self, x_t, t, eps):
        x_0 = self.gauss_x_0_from_eps(x_t, t, eps)
        mean, var = self.gauss_q_posterior_mean_var(x_0, x_t, t)
        return mean, var
    
    def gauss_p_sample(self, x_t, t, eps):
        mean, var = self.gauss_p_estimated_mean_var(x_t, t, eps)
        std = torch.sqrt(var)
        x_0 = mean + std * torch.randn_like(x_t)
        return x_0
    
    def gauss_loss(self, eps_real, eps_pred):
        loss = (eps_real - eps_pred) ** 2
        loss = loss.mean()
        return loss
    
    def multinomial_q_one_timestep(self, log_x_t, t):
        log_alpha = extract(self.log_alpha, t, log_x_t.shape)
        log_one_min_alpha = extract(self.log_one_min_alpha, t, log_x_t.shape)
        log_dist = log_add_exp(
            log_x_t + log_alpha, 
            log_one_min_alpha - torch.log(self.num_classes_expanded)
        )
        return log_dist
    
    def multinomial_q_prior_dist(self, log_x_0, t):
        log_cumprod_alpha = extract(self.log_cumprod_alpha, t, log_x_0.shape)
        log_one_min_cumprod_alpha = extract(self.log_one_min_cumprod_alpha, t, log_x_0.shape)
        log_dist = log_add_exp(
            log_x_0 + log_cumprod_alpha, 
            log_one_min_cumprod_alpha - torch.log(self.num_classes_expanded)
        )
        return log_dist
    
    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def multinomial_q_sample(self, log_x_0, t):
        log_dist = self.multinomial_q_prior_dist(log_x_0, t)
        log_sample = self.log_sample_categorical(log_dist)
        return log_sample
    
    def multinomial_q_posterior_dist(self, log_x_0, log_x_t, t):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.multinomial_q_prior_dist(log_x_0, t_minus_1)
        num_axes = (1,) * (len(log_x_0.size()) - 1)
        t_broadcast = t.to(log_x_0.device).view(-1, *num_axes) * torch.ones_like(log_x_0)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_0, log_EV_qxtmin_x0.to(torch.float32))
        unnormed_logprobs = log_EV_qxtmin_x0 + self.multinomial_q_one_timestep(log_x_t, t)
        log_dist = unnormed_logprobs - sliced_logsumexp(unnormed_logprobs, self.offsets)
        return log_dist
    
    def multinomial_x_0_from_pred(self, pred, log_x_t, t):
        log_x_0 = torch.empty_like(pred)
        for ix in self.slices_for_classes:
            log_x_0[:, ix] = F.log_softmax(pred[:, ix], dim=1)
        return log_x_0
    
    def multinomial_p_estimated_dist(self, pred, log_x_t, t):
        log_x_0 = self.multinomial_x_0_from_pred(pred, log_x_t, t)
        log_dist = self.multinomial_q_posterior_dist(log_x_0, log_x_t, t)
        return log_dist
    
    def multinomial_p_sample(self, pred, log_x_t, t):
        log_dist = self.multinomial_p_estimated_dist(pred, log_x_t, t)
        log_sample = self.log_sample_categorical(log_dist)
        return log_sample
    
    def multinomial_reconstruction_loss(self, log_x_0):
        batch_size = log_x_0.size(0)
        device = log_x_0.device
        ones = torch.ones(batch_size, device=device).long()

        log_qxT_prob = self.multinomial_q_prior_dist(log_x_0, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = multinomial_kl_div(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)
    
    def multinomial_diffusion_loss(self, pred, log_x_0, log_x_t, t):
        log_dist_true = self.multinomial_q_posterior_dist(log_x_0, log_x_t, t)
        log_dist_est = self.multinomial_p_estimated_dist(pred, log_x_t, t)
        kl_div = multinomial_kl_div(log_dist_true, log_dist_est)
        kl_div = sum_except_batch(kl_div)
        decoder_nll = -log_categorical(log_x_0, log_dist_est)
        decoder_nll = sum_except_batch(decoder_nll)
        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl_div
        return loss
    
    def multinomial_loss(self, pred, log_x_0, log_x_t, t):
        diffusion_loss = self.multinomial_diffusion_loss(pred, log_x_0, log_x_t, t)
        reconstruction_loss = self.multinomial_reconstruction_loss(log_x_0)
        loss = diffusion_loss * self.num_timesteps + reconstruction_loss
        loss = loss.mean()
        return loss
    
    # def mixed_loss(self, x_0, t, y):
    #     x_num_0 = x_0[:, :self.num_numerical_features]
    #     x_cat_0 = x_0[:, self.num_numerical_features:]

    #     # sample x_t
    #     noise = torch.randn_like(x_num_0)
    #     x_num_t = self.gaussian_q_sample(x_num_0, t, noise)
    #     x_cat_t = self.multinomial_q_sample(x_cat_0, t)
    #     x_t = torch.cat([x_num_t, x_cat_t], dim=1)

    #     # denoise x_t
    #     pred = self.denoise_fn(x_t, t, y)
    #     pred_num = pred[:, :self.num_numerical_features]
    #     pred_cat = pred[:, self.num_numerical_features:]
        
    #     # compute loss
    #     loss_num = self.gaussian_loss(noise, pred_num)
    #     loss_cat = self.multinomial_loss(pred_cat)  # TODO: change this
    #     return loss_num.mean() + loss_cat.mean()
    def forward(self, x_0, y):
        batch_size = x_0.shape[0]
        timesteps = sample_timesteps(self.num_timesteps, batch_size).to(x_0.device)
        x_num_0 = x_0[:, :self.num_numerical_features]
        x_cat_0 = x_0[:, self.num_numerical_features:]
        x_num_t = x_num_0
        log_x_cat_t = x_cat_0
        if x_num_0.shape[1] > 0:
            noise = torch.randn_like(x_num_0)
            x_num_t = self.gauss_q_sample(x_num_0, timesteps, noise)
        if x_cat_0.shape[1] > 0:
            log_x_cat_0 = index_to_log_onehot(x_cat_0.long(), self.num_classes)
            log_x_cat_t = self.multinomial_q_sample(log_x_cat_0, timesteps)
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)
        pred = self.denoise_fn(x_in, timesteps, y)
        pred_num = pred[:, :self.num_numerical_features]
        pred_cat = pred[:, self.num_numerical_features:]
        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat_0.shape[1] > 0:
            loss_multi = self.multinomial_loss(pred_cat, log_x_cat_0, log_x_cat_t, timesteps)
        if x_num_0.shape[1] > 0:
            loss_gauss = self.gauss_loss(pred_num, noise)
        loss = loss_gauss.mean() + loss_multi.mean()
        return loss

    @torch.no_grad()
    def sample(self, num_samples, y_dist):
        batch_size = num_samples
        y = torch.multinomial(
            y_dist,
            num_samples=batch_size,
            replacement=True
        ).long()
        device = self.log_alpha.device
        z_norm = torch.randn((batch_size, self.num_numerical_features), device=device)
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((batch_size, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((batch_size, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            model_in = torch.cat([z_norm, log_z], dim=1).float()
            model_out = self.denoise_fn(model_in, t, y)
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gauss_p_sample(model_out_num, t, z_norm)
            if has_cat:
                log_z = self.multinomial_p_sample(model_out_cat, log_z, t)
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, y
    
    def sample_all(self, num_samples, batch_size, y_dist):
        all_y = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, y = self.sample(batch_size, y_dist)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            all_samples.append(sample)
            all_y.append(y.cpu())
            num_generated += sample.shape[0]
        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]
        return x_gen, y_gen

if __name__ == "__main__":
    ddpm = TabDDPM(beta=np.linspace(0.1, 0.9, 10), 
                   num_classes=np.array([2, 3, 4]), num_numerical_features=2, 
                   denoise_fn=None)
    print(ddpm.num_classes_expanded)