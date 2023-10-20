from collections import defaultdict
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

DEFAULT_CONFIG = dict(signed=False,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.01, #changed from 0.1 to 0.01
                      optim='adam',
                      restarts=1,
                      max_iterations=100,
                      total_variation=0, #changed from 1e-1 to 0
                      bn_stat=0, #changed from 1e-1 to 0
                      image_norm=0, #changed from 1e-1 to 0
                      z_norm=0,
                      group_lazy=0, #changed from 1e-1 to 0
                      init='randn',
                      lr_decay=True,

                      dataset='CIFAR10',

                      generative_model='',
                      gen_dataset='',
                      giml=False, 
                      gias=False,
                      gias_lr=0.1,
                      gias_iterations=0,
                      )

construct_group_mean_at = 500
save_interval=100

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


def total_variation(x): #(1,48,76)
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


class BNStatisticsHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        # r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
        #     module.running_mean.data - mean, 2)
        mean_var = [mean, var]

        self.mean_var = mean_var
        # must have no output

    def close(self):
        self.hook.remove()
        
def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn.startswith('compressed'):
                ratio = float(cost_fn[10:])
                k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
                k = max(k, 1)

                trial_flatten = trial_gradient[i].flatten()
                trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
                trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
                trial_compressed = trial_flatten * trial_mask

                input_flatten = input_gradient[i].flatten()
                input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
                input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
                input_compressed = input_flatten * input_mask
                costs += ((trial_compressed - input_compressed).pow(2)).sum() * weights[i]
            elif cost_fn.startswith('sim_cmpr'):
                ratio = float(cost_fn[8:])
                k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
                k = max(k, 1)
                
                input_flatten = input_gradient[i].flatten()
                input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
                input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
                input_compressed = input_flatten * input_mask

                trial_flatten = trial_gradient[i].flatten()
                # trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
                # trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
                trial_compressed = trial_flatten * input_mask

                
                costs -= (trial_compressed * input_compressed).sum() * weights[i]
                pnorm[0] += trial_compressed.pow(2).sum() * weights[i]
                pnorm[1] += input_compressed.pow(2).sum() * weights[i]

            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                input_gradient[i].flatten(),
                                                                0, 1e-10) * weights[i]
        if cost_fn.startswith('sim'):
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)

class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, G=None, bn_prior=((0.0, 1.0))):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.setup = dict(device=self.device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        #BN Statistics
        self.bn_layers = []
        if self.config['bn_stat'] > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn_layers.append(BNStatisticsHook(module))
        self.bn_prior = bn_prior
        
        #Group Regularizer
        self.do_group_mean = False
        self.group_mean = None
        
        self.loss_fn = torch.nn.BCELoss()
        self.iDLG = True


        self.G = None
        self.generative_model_name = self.config['generative_model']
        self.initial_z = None

    def set_initial_z(self, z):
        self.initial_z = z

    def init_dummy_z(self, num_images):
        dummy_z = self.initial_z.clone().unsqueeze(0) \
            .expand(num_images, self.initial_z.shape[0], self.initial_z.shape[1]) \
            .to(self.device).requires_grad_(True)
        return dummy_z


    def gen_dummy_data(self, G, generative_model_name, dummy_z):
        running_device = dummy_z.device
        if generative_model_name.startswith('stylegan2-ada'):
            # @click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
            dummy_data = G(dummy_z, noise_mode='random')
        elif generative_model_name.startswith('stylegan2'):
            dummy_data = G(dummy_z)
            if self.config['gen_dataset'].startswith('I'):
                kernel_size = 512 // self.image_size
            else:
                kernel_size = 1024 // self.image_size
            dummy_data = torch.nn.functional.avg_pool2d(dummy_data, kernel_size)

        elif generative_model_name in ['stylegan2-ada-z']:
            dummy_data = G(dummy_z, None, truncation_psi=0.5, truncation_cutoff=8)
        elif generative_model_name in ['DCGAN', 'DCGAN-untrained']:
            dummy_data = G(dummy_z)
        
        dm, ds = self.mean_std
        dummy_data = (dummy_data + 1) / 2
        dummy_data = (dummy_data - dm.to(running_device)) / ds.to(running_device)
        return dummy_data

    def count_trainable_params(self, G=None, z=None , x=None):
        n_z, n_G, n_x = 0,0,0
        if G:
            n_z = torch.numel(z) if z.requires_grad else 0
            print(f"z: {n_z}")
            n_G += sum(layer.numel() for layer in G.parameters() if layer.requires_grad)
            print(f"G: {n_G}")
        else:
            n_x = torch.numel(x) if x.requires_grad else 0
            print(f"x: {n_x}")
        self.n_trainable = n_z + n_G + n_x

    def reconstruct(self, input_data, labels, data_shape=(48, 76), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()

        if torch.is_tensor(input_data[0]):
            input_data = [input_data] #[(48,76)]
        
        stats = defaultdict(list)
        x = self._init_images(data_shape) #(#restarts, 1, 48, 76)
        scores = torch.zeros(self.config['restarts'])

        try:
            # labels = [None for _ in range(self.config['restarts'])]
            dummy_z = [None for _ in range(self.config['restarts'])]
            optimizer = [None for _ in range(self.config['restarts'])]
            scheduler = [None for _ in range(self.config['restarts'])]
            _x = [None for _ in range(self.config['restarts'])]
            max_iterations = self.config['max_iterations']

            for trial in range(self.config['restarts']):
                _x[trial] = x[trial]
                
                _x[trial].requires_grad = True
                if self.config['optim'] == 'adam':
                    optimizer[trial] = torch.optim.Adam([_x[trial]], lr=self.config['lr'])
                elif self.config['optim'] == 'sgd':  # actually gd
                    optimizer[trial] = torch.optim.SGD([_x[trial]], lr=0.01, momentum=0.9, nesterov=True)
                elif self.config['optim'] == 'LBFGS':
                    optimizer[trial] = torch.optim.LBFGS([_x[trial]])
                else:
                    raise ValueError()

                if self.config['lr_decay']:
                    scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial],
                                                                        milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                                    max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
            dm, ds = self.mean_std

            print("Start original space search")
            self.count_trainable_params(x=_x[0])
            print(f"Total number of trainable parameters: {self.n_trainable}")
            
            for iteration in range(max_iterations):
                for trial in range(self.config['restarts']):
                    losses = [0,0,0,0]
                    # x_trial = _x[trial]
                    # x_trial.requires_grad = True
                    
                    #Group Regularizer
                    if trial == 0 and iteration + 1 == construct_group_mean_at and self.config['group_lazy'] > 0:
                        self.do_group_mean = True
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                    self.dummy_z = None
                    # print(x_trial)
                    closure = self._gradient_closure(optimizer[trial], _x[trial], input_data, labels, losses)
                    rec_loss = optimizer[trial].step(closure)
                    if self.config['lr_decay']:
                        scheduler[trial].step()

                    with torch.no_grad():
                        # Project into image space
                        # _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                        if (iteration + 1 == max_iterations) or iteration % 10 == 0:
                            print(f'It: {iteration}. Rec. loss: {rec_loss.item():.6f}')
                            if self.config['z_norm'] > 0:
                                print(torch.norm(dummy_z[trial], 2).item())

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass

                    
        for trial in range(self.config['restarts']):
            x[trial] = _x[trial].detach()
            scores[trial] = self._score_trial(x[trial], input_data, labels)
            if tol is not None and scores[trial] <= tol:
                break
            if dryrun:
                break
        # Choose optimal result:
        print('Choosing optimal result ...')
        scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
        optimal_index = torch.argmin(scores)
        print(f'Optimal result score: {scores[optimal_index]:2.4f}')
        stats['opt'] = scores[optimal_index].item()
        x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats


    def reconstruct_theta(self, input_gradients, labels, models, candidate_images, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        print("check")
        if eval:
            self.model.eval()

        stats = defaultdict(list)
        x = self._init_images(img_shape)
        scores = torch.zeros(self.config['restarts'])

        self.reconstruct_label = False

        assert self.config['restarts'] == 1
        max_iterations = self.config['max_iterations']
        num_seq = len(models)
        assert num_seq == len(input_gradients)
        assert num_seq == len(labels)
        for l in labels:
            assert l.shape[0] == self.num_images

        try:
            # labels = [None for _ in range(self.config['restarts'])]
            batch_images = [None for _ in range(num_seq)]
            skip_t = []
            current_labels = [label.item() for label in labels[-1]]
            optimize_target = set()

            for t in range(num_seq):
                batch_images[t] = []
                skip_flag = True
                for label_ in labels[t]:
                    label = label_.item()
                    if label in current_labels:
                        skip_flag = False
                    if label not in candidate_images.keys():
                        candidate_images[label] = torch.randn((1, *img_shape), **self.setup).requires_grad_(True)
                    batch_images[t].append(candidate_images[label])
                    if label not in optimize_target:
                        optimize_target.add(candidate_images[label])
                if skip_flag:
                    skip_t.append(t)

            optimizer = torch.optim.Adam(optimize_target, lr=self.config['lr'])
            if self.config['lr_decay']:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                    milestones=[max_iterations // 2.667, max_iterations // 1.6,
                                                                                max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8

            dm, ds = self.mean_std
            for iteration in range(max_iterations):
                losses = [0,0,0,0]
                batch_input = []

                for t in range(num_seq):
                    batch_input.append(torch.cat(batch_images[t], dim=0))

                def closure():
                    total_loss = 0
                    optimizer.zero_grad()
                    for t in range(num_seq):
                        models[t].zero_grad()
                    for t in range(num_seq):
                        if t in skip_t:
                            continue
                        loss = self.loss_fn(models[t](batch_input[t]), labels[t])
                        gradient = torch.autograd.grad(loss, models[t].parameters(), create_graph=True)
                        rec_loss = reconstruction_costs([gradient], input_gradients[t],
                                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                                        weights=self.config['weights'])

                        if self.config['total_variation'] > 0:
                            tv_loss = total_variation(batch_input[t])
                            rec_loss += self.config['total_variation'] * tv_loss
                            losses[0] = tv_loss
                        total_loss += rec_loss
                    total_loss.backward()
                    return total_loss
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():

                    if iteration == 0 or (iteration+1) % 10 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E}')

                if dryrun:
                    break

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
                    
        for t in range(num_seq):
            batch_input.append(torch.cat(batch_images[t], dim=0))

        scores = self._score_trial(batch_input[-1], [input_gradients[-1]], labels[-1])
        scores = scores[torch.isfinite(scores)]
        stats['opt'] = scores.item()

        print(f'Total time: {time.time()-start_time}.')
        return batch_input[-1].detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()


    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, losses):
        def closure():
            num_images = label.shape[0]
            num_gradients = len(input_gradient)
            batch_size = num_images // num_gradients
            num_batch = num_images // batch_size

            total_loss = 0
            optimizer.zero_grad()
            self.model.zero_grad()
            for i in range(num_batch):
                batch_input = x_trial[i,:,].unsqueeze(0)
                batch_label = label[i].reshape(-1)

                loss = self.loss_fn(self.model(batch_input).view(-1), batch_label.float())
                gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                                cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                                weights=self.config['weights'])
                total_loss += rec_loss
            total_loss.backward()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        num_images = label.shape[0]
        num_gradients = len(input_gradient)
        batch_size = num_images // num_gradients
        num_batch = num_images // batch_size

        total_loss = 0
        for i in range(num_batch):
            self.model.zero_grad()
            x_trial.grad = None

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_input = x_trial[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            loss = self.loss_fn(self.model(batch_input).view(-1), batch_label.float())
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                    cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                    weights=self.config['weights'])
            total_loss += rec_loss
        return total_loss