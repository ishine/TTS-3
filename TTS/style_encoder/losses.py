import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class ClipLloss(torch.nn.Module):
    '''
        Apply CLIP contrastive loss to the reference style embedding and the resynt-mel-spectrogram style embedding.
        The idea is to guide the style embedding to generate a bottleneck where the generated mel-spectrogram has the same
        style embedding as used to generate itself, while, at the same time, putting far away other representations, such as
        other styles.

        Details: We don't use the projection head layer because our nature problem and pipeline already guarantee that our embeddings
        have the same shape.

        Based on: https://github.com/moein-shariatnia/OpenAI-CLIP
    '''
    def __init__(self, c):
        super(ClipLloss, self).__init__()
        self.temperature = c.clip_temperature
        self.config = c

        print(f'CLIP report - Using alpha loss: {self.config.clip_alpha_loss} and temperature: {self.temperature}')

        
    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, ref_embedding, resynt_embedding):

        # Calculating the Loss
        logits = (ref_embedding @ resynt_embedding.T) / self.temperature
        resynt_similarity = resynt_embedding @ resynt_embedding.T
        ref_similarity = ref_embedding @ ref_embedding.T
        targets = F.softmax(
            (resynt_similarity + ref_similarity) / 2 * self.temperature, dim=-1
        )
        ref_loss = self.cross_entropy(logits, targets, reduction='none')
        resynt_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        clip_loss =  (resynt_loss + ref_loss) / 2.0 # shape: (batch_size)
        clip_loss= clip_loss.mean()
    
        clip_loss = self.config.clip_alpha_loss*clip_loss
        return clip_loss 

class DiffusionStyleEncoderLoss(torch.nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        self.config = c
        self.start_loss_at = self.config.start_loss_at
        self.step = 0

        print("Diffusion report - Using alpha loss: ", self.config.diff_loss_alpha)

        if self.config.diff_loss_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.config.diff_loss_type == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError

    def forward(self, diff_output, diff_target):

        if((self.start_loss_at > 0) and (self.step > self.start_loss_at)):
            self.stop_alpha = 1
        else:
            self.stop_alpha = 0

        diff_loss = self.stop_alpha*self.criterion(diff_output, diff_target)
        
        self.step += 1  

        return diff_loss

class VAEStyleEncoderLoss(torch.nn.Module):
    def __init__(self,c) -> None:
        super().__init__()
        self.config = c  
        self.alpha_vae = self.config.vae_loss_alpha # alpha of the loss function, it will be changed while training
        self.step = 0  # it will be incremented every forward and alpha_vae will be recalculated
        self.start_loss_at = self.config.start_loss_at

        print("VAE report - Using Cyclical Annealing: " ,self.config['use_cyclical_annealing'])

    def forward(self, mean, log_var, step):
        KL = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        
        if((self.start_loss_at > 0) and (self.step > self.start_loss_at)):
            self.stop_alpha = 1
        else:
            self.stop_alpha = 0

        # Doing this here to not affect original training schedules of other style encoders
        if(self.config['use_cyclical_annealing']): 
            # print(self.step, self.alpha_vae) Used to debug, and seems to be working
            self.update_alphavae(step)
        
        return self.stop_alpha*KL

    def update_alphavae(self, step):
        self.alpha_vae = min(self.config.vae_loss_alpha, (step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'])
        # Verbose       
        if((step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'] > 1):
            print("VAE: Cyclical annealing restarting") # This print is not working

class VAEFlowStyleEncoderLoss(torch.nn.Module):
    def __init__(self,c) -> None:
        super().__init__()
        self.config = c  
        self.alpha_vae = self.config.vae_loss_alpha # alpha of the loss function, it will be changed while training
        self.step = 0  # it will be incremented every forward and alpha_vae will be recalculated
        self.start_loss_at = self.config.start_loss_at

        print("VAEFlow report - Using Cyclical Annealing: " ,self.config['use_cyclical_annealing'])

    def forward(self, z_0, z_T, mean, log_var):
        
        log_p_z = self.log_Normal_standard(z_T.squeeze(1), dim=1)
        log_q_z = self.log_Normal_diag(z_0.squeeze(1), mean, log_var, dim=1)

        if((self.start_loss_at > 0) and (self.step > self.start_loss_at)):
            self.stop_alpha = 1
        else:
            self.stop_alpha = 0

        KL = (- torch.sum(log_p_z - log_q_z) )

        if(self.config['use_cyclical_annealing']):
            self.update_alphavae(self.step)

        self.step += 1  
        
        return self.stop_alpha*KL

    def update_alphavae(self, step):
        self.alpha_vae = self.config.vae_loss_alpha - min(self.config.vae_loss_alpha, self.config.vae_loss_alpha*(step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'])
        # Verbose       
        if((step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'] > 1):
            print("VAE: Cyclical annealing restarting") # This print is not working

    def log_Normal_diag(self, x, mean, log_var, average=False, dim=None):
        log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var ), -1) )
        if average:
            return torch.mean(log_normal, dim)
        else:
            return torch.sum(log_normal, dim)

    def log_Normal_standard(self, x, average=False, dim=None):
        log_normal = -0.5 * torch.pow( x , 2 )
        if average:
            return torch.mean(log_normal, dim)
        else:
            return torch.sum(log_normal, dim)

class VQVAEStyleEncoderLoss(torch.nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        self.config = c
        self.step = 0
        self.beta_vqvae = self.config.vqvae_commitment_beta

    def forward(self, z_q_x, z_e_x):
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        self.step += 1  
        return loss_vq, loss_commit
    

################################
################################    Balanced/Focal cross entropy. Ref1: https://www.kaggle.com/code/kaerunantoka/birdclef2022-use-2nd-label-f0
################################                                  Ref2: https://github.com/fcakyon/balanced-loss

# This class is not used at all, only got as reference. Specifically, it is equal to BalancedCrossEntropyLoss(loss_type="focal_loss")
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class BalancedCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        
        Example usage:
        
            # outputs and labels
            logits = torch.tensor([[0.78, 0.1, 0.05]]) # 1 batch, 3 class
            labels = torch.tensor([0]) # 1 batch

            # number of samples per class in the training dataset
            samples_per_class = [30, 100, 25] # 30, 100, 25 samples for labels 0, 1 and 2, respectively

            # class-balanced focal loss
            focal_loss = Loss(
                loss_type="focal_loss",
                samples_per_class=samples_per_class,
                class_balanced=True
            )
            
            loss = focal_loss(logits, labels)
        
        
        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            BalancedCrossEntropyLoss instance
        """
        super(BalancedCrossEntropyLoss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


################################
################################    FINISH BALANCED/FOCAL CODE
################################


################################
################################    CLUB code is based on the official repo: https://github.com/Linear95/CLUB
################################

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


################################
################################    FINISH CLUB CODE
################################