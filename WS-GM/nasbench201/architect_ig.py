import numpy as np
import torch
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

        self._init_arch_parameters = []
        for alpha in self.model.arch_parameters():
            alpha_init = torch.zeros_like(alpha)
            alpha_init.data.copy_(alpha)
            self._init_arch_parameters.append(alpha_init)

        #### mode
        if args.method in ['darts', 'darts-proj', 'sdarts', 'sdarts-proj', 'ws',
                           'gdas', 'ws-gdas', 'ws-snas', 'ws-drnas', 'ws-rsws']:
            self.method = 'fo' # first order update
        elif 'so' in args.method:
            self.method = 'so'
        elif args.method in ['blank', 'blank-proj']:
            self.method = 'blank'
        else:
            print('ERROR: WRONG ARCH UPDATE METHOD', args.method); exit(0)

        self.hvp_method = 'approx' # like DARTS_2nd

    def reset_arch_parameters(self):
        for alpha, alpha_init in zip(self.model.arch_parameters(), self._init_arch_parameters):
            alpha.data.copy_(alpha_init.data)

    def step(self, input_train, target_train, input_valid, target_valid, *args, **kwargs):
        if self.method == 'fo':
            shared = self._step_fo(input_train, target_train, input_valid, target_valid)
        elif self.method == 'so':
            shared = self._step_darts_so(input_train, target_train, input_valid, target_valid,\
                                         eta=kwargs['eta'], model_optimizer=kwargs['model_optimizer'])
        elif self.method == 'blank': ## do not update alpha
            shared = None
        return shared

    #### first order
    def _step_fo(self, input_train, target_train, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()
        self.optimizer.step()
        return None


    #### darts 2nd order
    def _step_darts_so(self, input_train, target_train, input_valid, target_valid, eta, model_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, model_optimizer)
        unrolled_loss = self._val_loss(model=unrolled_model, input=input_valid, target=target_valid)

        # Compute backwards pass with respect to the unrolled model parameters
        unrolled_loss.backward()
        dalpha = [v.grad if v.grad is not None else torch.zeros_like(v) for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data if v.grad is not None else torch.zeros_like(v).data for v in unrolled_model.parameters()]

        # Compute expression (8) from paper
        if self.hvp_method == 'approx':
            implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        elif self.hvp_method == 'exact':
            raise NotImplementedError
            # Lt, logit_t = self.model._loss(input=input_train, target=target_train, return_logits=True)
            # g_Lt_w = torch.autograd.grad(Lt, self.model.parameters(), create_graph=True, retain_graph=True)
            # implicit_grads = hvp_exact_v2(vector, g_Lt_w, self.model.arch_parameters())

        # Compute expression (7) from paper
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()
        return

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self._train_loss(model=self.model, input=input, target=target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)

        ## wrc modified, remove unused ops in subnetworks
        model_params = list(self.model.parameters())
        grads_with_none = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        grads = []
        count = 0
        for idx, grad in enumerate(grads_with_none):
            if grad is None: # pruned ops
                grad = torch.zeros_like(model_params[idx].data)
                count += 1
            assert grad.shape == model_params[idx].shape
            grads.append(grad)
        dtheta = _concat(grads).data + self.network_weight_decay * theta # some ops are unused in subnetworks
        
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta)) # also set encoding
        return unrolled_model

    def _construct_model_from_theta(self, theta):
        if self.model.space_name == '201':
            model_new = self.model.new(self.model.enc) ## set enc for few-shot nas
        elif self.model.space_name == 'darts':
            model_new = self.model.new(self.model.enc_normal, self.model.enc_reduce) ## set enc for few-shot nas
        else:
            print(f'ERROR: UNRECOGNIZED SPACE: {self.model.space_name}')
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _train_loss(self, model, input, target):
        return model._loss(input, target)

    def _val_loss(self, model, input, target):
        return model._loss(input, target)
