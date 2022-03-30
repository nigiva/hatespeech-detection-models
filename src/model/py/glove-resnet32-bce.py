# %% [markdown]
# # ResNet32 BCE

# %% [markdown]
# ## Variables d'environnement

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Id des GPU disponibles : 0 et 1

# %% [markdown]
# ## Importation

# %%
import os
import time
import datetime
from typing import Any, Union, Dict, List
import uuid
import json

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchtext
import nltk
import sklearn
import transformers
import torchmetrics as tm
from torchmetrics import MetricCollection, Metric, Accuracy, Precision, Recall, AUROC, HammingDistance, F1Score, ROC, AUC, PrecisionRecallCurve


from loguru import logger
from tqdm.auto import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Constantes

# %%
CUSTOME_NAME = "glove-resnet32-bce"

# Dataset
DATA_DIR_PATH = os.path.abspath("../../data")
TRAIN_DATASET_PATH = os.path.join(DATA_DIR_PATH, "jigsaw2019-train.csv")
TEST_DATASET_PATH = os.path.join(DATA_DIR_PATH, "jigsaw2019-test.csv")
LABEL_LIST = ['toxicity', 'obscene', 'sexual_explicit',
            'identity_attack', 'insult', 'threat']
IDENTITY_LIST = ['male', 'female', 'transgender', 'other_gender', 'heterosexual',
                'homosexual_gay_or_lesbian', 'bisexual','other_sexual_orientation',
                'christian', 'jewish', 'muslim', 'hindu','buddhist', 'atheist',
                'other_religion', 'black', 'white', 'asian', 'latino',
                'other_race_or_ethnicity', 'physical_disability',
                'intellectual_or_learning_disability',
                'psychiatric_or_mental_illness','other_disability']
SELECTED_IDENTITY_LIST = ['male', 'female', 'black', 'white', 'homosexual_gay_or_lesbian',
                    'christian', 'jewish', 'muslim', 'psychiatric_or_mental_illness']

# Session
SESSION_DIR_PATH = os.path.abspath("../../session")
SESSION_DATETIME = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
SESSION_NAME = f"{CUSTOME_NAME}_{SESSION_DATETIME}"
CURRENT_SESSION_DIR_PATH = os.path.join(SESSION_DIR_PATH, SESSION_NAME)
# Créer le dossier de la session
os.makedirs(CURRENT_SESSION_DIR_PATH, exist_ok=True)

# Architecture de fichier dans `CURRENT_SESSION_DIR_PATH`
LOG_FILE_NAME = f"{SESSION_NAME}.loguru.log"
MODEL_FILE_NAME = f"{SESSION_NAME}.state_dict.model"
TEST_FILE_NAME = f"{SESSION_NAME}.test.csv"
VALIDATION_DATASET_NAME = f"{SESSION_NAME}.jigsaw2019-validation.csv"
VALIDATION_FILE_NAME = f"{SESSION_NAME}.validation.csv"
METRIC_FILE_NAME = f"{SESSION_NAME}.metric.json"
LOG_FILE_PATH = os.path.join(CURRENT_SESSION_DIR_PATH, LOG_FILE_NAME)
MODEL_FILE_PATH = os.path.join(CURRENT_SESSION_DIR_PATH, MODEL_FILE_NAME)
TEST_FILE_PATH = os.path.join(CURRENT_SESSION_DIR_PATH, TEST_FILE_NAME)
VALIDATION_DATASET_FILE_PATH = os.path.join(CURRENT_SESSION_DIR_PATH, VALIDATION_DATASET_NAME)
VALIDATION_FILE_PATH = os.path.join(CURRENT_SESSION_DIR_PATH, VALIDATION_FILE_NAME)
METRIC_FILE_PATH = os.path.join(CURRENT_SESSION_DIR_PATH, METRIC_FILE_NAME)

# CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# ## Logging

# %%
logger.add(LOG_FILE_PATH, level="TRACE")
logger.info(f"{SESSION_NAME=}")
logger.info(f"{TRAIN_DATASET_PATH=}")
logger.info(f"{TEST_DATASET_PATH=}")
logger.info(f"{CURRENT_SESSION_DIR_PATH=}")
logger.info(f"{LABEL_LIST=}")
logger.info(f"{IDENTITY_LIST=}")
logger.info(f"{SELECTED_IDENTITY_LIST=}")

# %% [markdown]
# ## Vérifier la cohérence de l'architecture et l'accès aux ressources

# %%
logger.info(f"Checking consistency...")

# Vérifier l'accès aux datasets
if not os.path.exists(TRAIN_DATASET_PATH):
    logger.critical(f"Train dataset does not exist !")
    raise RuntimeError("Train dataset does not exist !")
if not os.path.exists(TEST_DATASET_PATH):
    logger.critical(f"Test dataset does not exist !")
    raise RuntimeError("Test dataset does not exist !")
logger.success("Datasets are reachable")

# Vérifier l'accès aux GPU
GPU_IS_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count()
logger.info(f"{GPU_IS_AVAILABLE=}")
logger.info(f"{GPU_COUNT=}")
if not GPU_IS_AVAILABLE:
    logger.critical("GPU and CUDA are not available !")
    raise RuntimeError("GPU and CUDA are not available !")
logger.success("GPU and CUDA are available")
logger.info(f"{device=}")
for gpu_id in range(GPU_COUNT):
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU {gpu_id} : {gpu_name}")

# %% [markdown]
# ## Dataset

# %%
all_train_df = pd.read_csv(TRAIN_DATASET_PATH, index_col=0)
logger.success("Dataset loaded !")

# %%
# Pour réduire le nombre d'exemple et savoir sur quel groupes d'identités
# le modèle est entrainé, on prend un sous ensemble du jeu de données
train_df = all_train_df[~all_train_df.white.isna()]
if CUSTOME_NAME.startswith("test"):
    # Si c'est juste une session pour tester le notebook
    logger.debug("Mode test is enabled. The training set has been truncated to 20 000 samples.")
    train_df = train_df[:20_000]
# Pour le jeu de validation, on veut juste avoir des indications de perf
# Peu importe, c'est juste pour voir si l'entraînement s'est bien passé
# Sans pour autant regarder les biais
validation_df = all_train_df[all_train_df.white.isna()].sample(n=10_000)

# %%
# Remplacer toutes les colonnes correspondantes aux labels par 1 ou 0
# si la probabilité est supérieure ou égale à 0.5 ou non
train_df[LABEL_LIST] = (train_df[LABEL_LIST]>=0.5).astype(int)
validation_df[LABEL_LIST] = (validation_df[LABEL_LIST]>=0.5).astype(int)

# %%
# For GloVe tokenizer and vocabulary
# Get the tokenizer
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
max_len = 70
logger.info(f"Tokenizer : basic_english")
# Build vocab
min_freq = 5
logger.info(f"Tokenizer : {min_freq=}")
special_tokens = ['<unk>', '<pad>']
logger.info(f"Tokenizer : {special_tokens=}")

# Create tokenized column
train_df['tokens'] = train_df['comment_text'].apply(lambda x: tokenizer(x)[:max_len])
logger.info(f"Tokens are stored in column of train_df")

vocab = torchtext.vocab.build_vocab_from_iterator(train_df['tokens'],
                                                  min_freq=min_freq,
                                                  specials=special_tokens)
logger.info(f"Vocab is built")

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']
vocab.set_default_index(unk_index)

# %%
class JigsawDataset(Dataset):
    def __init__(self, data_df, tokenizer, vocab, max_len=64):
        self.data = data_df
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = self.data.iloc[index]["comment_text"]
        label = torch.tensor(self.data.iloc[index][LABEL_LIST].tolist(), dtype=torch.float)
        
        # Tokenize the sentence
        tokenized_sentence = self.tokenizer(comment)[:self.max_len]

        # Get ids from tokens
        ids = torch.tensor([self.vocab[token] for token in tokenized_sentence])

        # Pad the sequence
        padding = torch.full((self.max_len - len(ids), ), self.vocab['<pad>'])
        padded_ids = torch.cat([ids, padding])

        return dict(index=index, ids=padded_ids, labels=label)

# %% [markdown]
# ## Model

# %%
# ResNet implementation from:
# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
# ResNet: resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, kernel_size=(out.size()[2], out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self,
                 nb_labels: int = 6,
                 in_channels: int = 3,
                 pretrained_embedding: str = "glove",
                 resnet_type: str = "32",
                 embedding_dim: int = 300,
                 freeze_embedding: bool = False,
                 stack_embedding: bool = False,
                 vocab = None):
        super().__init__()        

        # If true stack embedding to 3 channels
        self.stack_embedding = stack_embedding

        if vocab:
            vocab_size = len(vocab)
            pad_index = vocab['<pad>']

        # Embedding type
        if pretrained_embedding == "glove":
            self.embedding = nn.Embedding(vocab_size, 300, padding_idx=pad_index)

            # Download pretrained glove embedding
            vectors = torchtext.vocab.GloVe()

            # Get the pretrained embedding vectors according to the vocabulary
            pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
            self.embedding.weight.data = pretrained_embedding

        elif pretrained_embedding == "fasttext":
            self.embedding = nn.Embedding(vocab_size, 300, padding_idx=pad_index)

            # Download pretrained FastText embedding
            vectors = torchtext.vocab.FastText()

            # Get the pretrained embedding vectors according to the vocabulary
            pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
            self.embedding.weight.data = pretrained_embedding

        elif pretrained_embedding == "roberta":
            # Download the pretrained Transformer model
            tr_model = transformers.AutoModel.from_pretrained('roberta-base')

            # set the embedding parameters
            self.embedding = tr_model.embeddings

        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        resnet_types_dic = {"20": 0, "32": 1, "44": 2, "56": 3, "110": 4, "1202": 5}
        resnet_params = [[3, 3, 3], [5, 5, 5], \
                         [7, 7, 7], [9, 9, 9], \
                         [18, 18, 18], [200, 200, 200]]

        if resnet_type not in resnet_types_dic:
            print("Error: 'resnet_type' should be in: ['20', '32', '44', '56', '110', '1202']")

        resnet_param_idx = resnet_types_dic[resnet_type]
        self.resnet = ResNet(BasicBlock,
                             resnet_params[resnet_param_idx],
                             in_channels=in_channels,
                             num_classes=nb_labels)

        # freeze all the embedding parameters if necessary
        for param in self.embedding.parameters():
            param.requires_grad = not freeze_embedding

    def forward(self, ids):
        # ids = [batch_size, padded_seq_len]
        x = self.embedding(ids)

        # x = [batch_size, padded_seq_len, embedding_dim]
        if self.stack_embedding:
            # Transform the image in 3 channels
            split = x.shape[-1] // 3
            x = torch.stack([x[..., 0:split], x[..., split:2*split], x[..., 2*split:]], dim=1)
            # x = [batch_size, 3, padded_seq_len, embedding_dim]
        else:
            x = torch.unsqueeze(x, 1)
            # x = [batch_size, 1, padded_seq_len, embedding_dim]

        x = self.resnet(x)

        # x = [batch_size, nb_labels]
        return x

def preds_fn(batch, model, device):
    '''
    Get the predictions for one batch according to the model
    '''
    ids = batch['ids'].int()
    b_input = ids.to(device, non_blocking=True)
    return model(b_input)

# %% [markdown]
# ### Instancier le modèle

# %%
model = ResNetClassifier(nb_labels=len(LABEL_LIST),
                         in_channels=1,
                         pretrained_embedding="glove",
                         resnet_type="32",
                         freeze_embedding=True,
                         stack_embedding=False,
                         vocab=vocab)

# %% [markdown]
# ## Hyperparamètre

# %%
BATCH_SIZE = 32
LR=1e-4
PIN_MEMORY = True
NUM_WORKERS = 0
PREFETCH_FACTOR = 2
NUM_EPOCHS = 1
logger.info(f"{BATCH_SIZE=}")
logger.info(f"{LR=}")
logger.info(f"{PIN_MEMORY=}")
logger.info(f"{NUM_WORKERS=}")
logger.info(f"{PREFETCH_FACTOR=}")
logger.info(f"{NUM_EPOCHS=}")

# %% [markdown]
# ### Loss

# %%
class FocalLoss(nn.Module):
    def __init__(self,
                 gamma: float = 2,
                 reduction: str = "mean",
                 pos_weight: torch.Tensor = None):
        super(FocalLoss, self).__init__()
        self.gamma= gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", pos_weight=self.pos_weight
        )
        p_t =  p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

# %%
# Taken from: https://github.com/Roche/BalancedLossNLP

class ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                raise RuntimeError("Not defined here")
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            raise RuntimeError("Not defined here")
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha'] # change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = torch.from_numpy(np.asarray(class_freq)).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        # bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(-logpt)
            wtloss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            alpha_t = torch.where(label==1, self.alpha, 1-self.alpha)
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss # balance_param should be a tensor
            loss = reduce_loss(loss, reduction)             # add reduction
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None): 
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight
    

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

# %%
def get_label_weights_bce(df, classes=LABEL_LIST):
    weights = torch.empty((len(classes),))

    nb_samples = len(df)

    for idx, c in enumerate(classes):
        nb_zeros = len(df[df[c] == 0])
        nb_ones = nb_samples - nb_zeros
        weights[idx] = nb_zeros / nb_ones

    return weights

def get_label_inv_freq(df, classes=LABEL_LIST):
    weights = torch.empty((len(classes),))
    nb_samples = len(df)

    for idx, c in enumerate(classes):
        nb_zeros = len(df[df[c] == 0])
        weights[idx] = (nb_zeros / nb_samples)

    return weights

def get_nb_samples_lab(df, classes=LABEL_LIST):
    nb_ones_tot, nb_zeros_tot = [], []
    nb_tot = len(df)

    for c in classes:
        nb_zeros = len(df[df[c] == 0])
        nb_ones = nb_tot - nb_zeros

        nb_ones_tot.append(nb_ones)
        nb_zeros_tot.append(nb_zeros)

    return torch.tensor(nb_ones_tot), torch.tensor(nb_zeros_tot)

# %% [markdown]
# ### Instancier les différents objets

# %%
train_dataset = JigsawDataset(train_df, tokenizer,
                              vocab=vocab,
                              max_len=max_len)
train_dataloader = DataLoader(train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS,
                             prefetch_factor=PREFETCH_FACTOR,
                             pin_memory=PIN_MEMORY)

validation_dataset = JigsawDataset(validation_df, tokenizer,
                              vocab=vocab,
                              max_len=max_len)
validation_dataloader = DataLoader(validation_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS,
                             prefetch_factor=PREFETCH_FACTOR,
                             pin_memory=PIN_MEMORY)

# Pas besoin de Sigmoid en sorti du model seulement pour `BCEWithLogitsLoss`
criterion = torch.nn.BCEWithLogitsLoss()
logger.info(criterion)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
logger.info(optimizer)

model.to(device)
criterion.to(device)

# %% [markdown]
# ## Metric

# %% [markdown]
# ### Variantes de Hamming Loss

# %%
class HammingLossWithoutThreshold(Metric):
    def __init__(self, num_classes=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes

        self.add_state("total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("nbr_sample", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        current_nbr_sample, current_nbr_category = preds.shape
        if current_nbr_category != self.num_classes:
          raise AttributeError("`num_classes` != `current_nbr_category` detected in `pred` parameter")
        
        current_loss_per_pred = torch.absolute(target - preds)
        current_hamming_loss = current_loss_per_pred.sum()

        self.total += current_hamming_loss.float()
        self.nbr_sample += current_nbr_sample

    def compute(self):
        return self.total/(self.num_classes*self.nbr_sample)

# %%
class RebalancedHammingLossWithoutThreshold(Metric):
    def __init__(self, num_classes=1, average="macro", dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes

        # average = "macro" or None
        self.average = average

        # Nombre de positif 1 & negatif 0 par categorie
        self.add_state(
            "number_positive",
            default=torch.tensor([0 for _ in range(num_classes)]),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "number_negative",
            default=torch.tensor([0 for _ in range(num_classes)]),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "hamming_loss_positive",
            default=torch.tensor([0.0 for _ in range(num_classes)]),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "hamming_loss_negative",
            default=torch.tensor([0.0 for _ in range(num_classes)]),
            dist_reduce_fx="sum",
        )

        self.add_state("nbr_sample", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        current_nbr_sample, current_nbr_category = preds.shape
        if current_nbr_category != self.num_classes:
            raise AttributeError(
                "`num_classes` != `current_nbr_category` detected in `pred` parameter"
            )

        # Nombre de positif 1 & negatif 0 par categorie
        current_number_positive = target.sum(axis=0)
        current_number_negative = current_nbr_sample - target.sum(axis=0)

        self.number_positive += current_number_positive.int()
        self.number_negative += current_number_negative.int()

        self.nbr_sample += current_nbr_sample

        for class_id in range(self.num_classes):
            positive_filter = target[:, class_id] == 1
            negative_filter = target[:, class_id] == 0

            target_vector = target[:, class_id]
            preds_vector = preds[:, class_id]

            # Filtered vector
            ## Target
            pos_filtered_target_vector = target_vector[positive_filter]
            neg_filtered_target_vector = target_vector[negative_filter]
            ## Preds
            pos_filtered_preds_vector = preds_vector[positive_filter]
            neg_filtered_preds_vector = preds_vector[negative_filter]

            # Hamming Loss without Threshold
            hamming_loss_on_positive = torch.absolute(
                pos_filtered_target_vector - pos_filtered_preds_vector
            )
            hamming_loss_on_negative = torch.absolute(
                neg_filtered_target_vector - neg_filtered_preds_vector
            )

            self.hamming_loss_positive[class_id] += hamming_loss_on_positive.sum()
            self.hamming_loss_negative[class_id] += hamming_loss_on_negative.sum()

    def compute(self):
        factor_pos = self.nbr_sample / (2 * self.number_positive)
        factor_neg = self.nbr_sample / (2 * self.number_negative)

        rebalanced_hamming_loss_per_class = torch.multiply(
            self.hamming_loss_positive, factor_pos
        ) + torch.multiply(self.hamming_loss_negative, factor_neg)
        if self.average == "macro":
            return rebalanced_hamming_loss_per_class.sum() / (
                self.nbr_sample * self.num_classes
            )
        return rebalanced_hamming_loss_per_class / (self.nbr_sample)


# %% [markdown]
# ### Instanciation des metrics

# %%
num_classes = len(LABEL_LIST)
train_metric_dict = dict()

# AUROC Macro
auroc_macro = AUROC(num_classes=num_classes, compute_on_step=True, average="macro")
train_metric_dict["auroc_macro"] = auroc_macro

# AUROC per class
auroc_per_class = AUROC(num_classes=num_classes, compute_on_step=True, average=None)
train_metric_dict["auroc_per_class"] = auroc_per_class

# F1 score global
f1 = F1Score()
train_metric_dict["f1"] = f1

# F1 score per class
f1_per_calss = F1Score(num_classes=6, average=None)
train_metric_dict["f1_per_calss"] = f1_per_calss

# Hamming Distance without Threshold
hamming_loss_woutt = HammingLossWithoutThreshold(num_classes=num_classes)
train_metric_dict["hamming_loss_without_threshold"] = hamming_loss_woutt

# Rebalanced Hamming Distance without Threshold macro
rebalanced_hamming_loss_woutt_macro = RebalancedHammingLossWithoutThreshold(
    num_classes=num_classes, average="macro"
)
train_metric_dict[
    "rebalanced_hamming_loss_without_threshold_macro"
] = rebalanced_hamming_loss_woutt_macro

# Rebalanced Hamming Distance without Threshold macro
rebalanced_hamming_loss_woutt_per_class = RebalancedHammingLossWithoutThreshold(
    num_classes=num_classes, average=None
)
train_metric_dict[
    "rebalanced_hamming_loss_without_threshold_per_class"
] = rebalanced_hamming_loss_woutt_per_class

# %%
train_metric = MetricCollection(train_metric_dict)
train_metric.to(device)

validation_metric = train_metric.clone()
validation_metric.to(device)

# %% [markdown]
# ### Export metrics

# %%
def serialize(object_to_serialize: Any, ensure_ascii: bool = True) -> str:
    """
    Serialize any object, i.e. convert an object to JSON
    Args:
        object_to_serialize (Any): The object to serialize
        ensure_ascii (bool, optional): If ensure_ascii is true (the default), the output is guaranteed to have all incoming non-ASCII characters escaped. If ensure_ascii is false, these characters will be output as-is. Defaults to True.
    Returns:
            str: string of serialized object (JSON)
    """

    def dumper(obj: Any) -> Union[str, Dict]:
        """
        Function called recursively by json.dumps to know how to serialize an object.
        For example, for datetime, we try to convert it to ISO format rather than
        retrieve the list of attributes defined in its object.
        Args:
            obj (Any): The object to serialize
        Returns:
            Union[str, Dict]: Serialized object
        """
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    return json.dumps(object_to_serialize, default=dumper, ensure_ascii=ensure_ascii)

# %%
def export_metric(metric_collection, **kwargs):
    """
    Export MetricCollection to json file

    Args:
        metric_collection: MetricCollection
        **kwargs: field to add in json line
    """
    with open(METRIC_FILE_PATH, "a") as f:
        metric_collection_value = metric_collection.compute()
        metric_collection_value.update(kwargs)
        serialized_value = serialize(metric_collection_value)
        f.write(serialized_value)
        f.write("\n")
    logger.success("Metrics are exported !")

# %% [markdown]
# ## Entraînement

# %%
def train_epoch(epoch_id=None):
    model.train()
    logger.info(f"START EPOCH {epoch_id=}")

    progress = tqdm(train_dataloader, desc='training batch...', leave=False)
    for batch_id, batch in enumerate(progress):
        if batch_id % 1_000 == 0:
            valid_epoch(epoch_id=epoch, batch_id=batch_id)
        
        logger.trace(f"{batch_id=}")
        token_list_batch = batch["ids"].to(device)
        label_batch = batch["labels"].to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Predict
        prediction_batch = model(token_list_batch)
        transformed_prediction_batch = prediction_batch.squeeze()

        # Loss
        loss = criterion(transformed_prediction_batch.to(torch.float32), label_batch.to(torch.float32))

        # Metrics
        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        train_metrics_collection_dict = train_metric(proba_prediction_batch.to(torch.float32), label_batch.to(torch.int32))
        logger.trace(train_metrics_collection_dict)

        # Backprop        
        loss.backward()
        # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update progress bar description
        progress_description = "Train Loss : {loss:.4f} - Train AUROC : {acc:.4f}"
        auroc_macro_value = float(train_metrics_collection_dict["auroc_macro"])
        progress_description = progress_description.format(loss=loss.item(), acc=auroc_macro_value)
        progress.set_description(progress_description)

    logger.info(f"END EPOCH {epoch_id=}")

# %%
@torch.no_grad()
def valid_epoch(epoch_id=None, batch_id=None):
    model.eval()
    logger.info(f"START VALIDATION {epoch_id=}{batch_id=}")
    validation_metric.reset()

    loss_list = []
    prediction_list = torch.Tensor([])
    target_list = torch.Tensor([])


    progress = tqdm(validation_dataloader, desc="valid batch...", leave=False)
    for _, batch in enumerate(progress):
        
        token_list_batch = batch["ids"].to(device)
        label_batch = batch["labels"].to(device)

        # Predict
        prediction_batch = model(token_list_batch)

        transformed_prediction_batch = prediction_batch.squeeze()

        # Loss
        loss = criterion(
            transformed_prediction_batch.to(torch.float32),
            label_batch.to(torch.float32),
        )

        loss_list.append(loss.item())

        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        prediction_list = torch.concat(
            [prediction_list, proba_prediction_batch.cpu()]
        )
        target_list = torch.concat([target_list, label_batch.cpu()])

        # Metrics
        validation_metric(proba_prediction_batch.to(torch.float32), label_batch.to(torch.int32))

    loss_mean = np.mean(loss_list)
    logger.trace(validation_metric.compute())
    logger.info(f"END VALIDATION {epoch_id=}{batch_id=}")
    export_metric(validation_metric, epoch_id=epoch_id, batch_id=batch_id, loss=loss_mean)

# %%
torch.cuda.empty_cache()
progress =  tqdm(range(1,NUM_EPOCHS+1), desc='training epoch...', leave=True)
for epoch in progress:
    # Train
    train_epoch(epoch_id=epoch)

    # Validation
    valid_epoch(epoch_id=epoch)

    # Save
    # torch.save(model, MODEL_FILE_PATH)
    torch.save(model.state_dict(), MODEL_FILE_PATH)

# %% [markdown]
# ## Evaluation

# %%
try:
    del train_df
    del validation_df
except NameError:
    logger.warning("Train DataFrame & Validation DataFrame already deleted")

# %%
test_df = pd.read_csv(TEST_DATASET_PATH, index_col=0)

# %%
test_df = test_df[~test_df.white.isna()]

# %%
test_dataset = JigsawDataset(test_df, tokenizer,
                              vocab=vocab,
                              max_len=max_len)
test_dataloader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# %%
@torch.no_grad()
def evaluation(model):
    model.eval()
    logger.info(f"START EVALUATION")

    index_tensor = torch.Tensor([])
    prediction_tensor = torch.Tensor([])

    progress = tqdm(test_dataloader, desc='test batch...', leave=False)
    for batch_id, batch in enumerate(progress):
        logger.trace(f"{batch_id=}")
        index_batch = batch["index"].to(device)
        token_list_batch = batch["ids"].to(device)
        label_batch = batch["labels"].to(device)

        # Predict
        prediction_batch = model(token_list_batch)
        transformed_prediction_batch = prediction_batch.squeeze()
        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        
        index_tensor = torch.concat([index_tensor, index_batch.cpu()])
        prediction_tensor = torch.concat([prediction_tensor, proba_prediction_batch.cpu()])
    
    logger.info(f"END EVALUATION")
    prediction_test_df = pd.DataFrame(prediction_tensor.tolist(), 
                                     columns=LABEL_LIST,
                                     index=index_tensor.to(int).tolist())
    prediction_test_df.to_csv(TEST_FILE_PATH)
    logger.success(f"Test predictions exported !")

# %%
evaluation(model)

# %% [markdown]
# ## Exporter les prédictions de la dataset de validation

# %%
@torch.no_grad()
def export_validation(model):
    model.eval()
    logger.info(f"START GET PREDICTION ON VALIDATION DATASET")

    index_tensor = torch.Tensor([])
    prediction_tensor = torch.Tensor([])
    label_tensor = torch.Tensor([])

    progress = tqdm(validation_dataloader, desc='valid batch...', leave=False)
    for batch_id, batch in enumerate(progress):
        logger.trace(f"{batch_id=}")
        index_batch = batch["index"].to(device)
        token_list_batch = batch["ids"].to(device)
        label_batch = batch["labels"].to(device)

        # Predict
        prediction_batch = model(token_list_batch)
        transformed_prediction_batch = prediction_batch.squeeze()
        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        
        index_tensor = torch.concat([index_tensor, index_batch.cpu()])
        prediction_tensor = torch.concat([prediction_tensor, proba_prediction_batch.cpu()])
        label_tensor = torch.concat([label_tensor, label_batch.cpu()])
    
    logger.info(f"END GET PREDICTION ON VALIDATION DATASET")
    prediction_valid_df = pd.DataFrame(prediction_tensor.tolist(), 
                                     columns=LABEL_LIST,
                                     index=index_tensor.to(int).tolist())
    label_valid_df = pd.DataFrame(label_tensor.tolist(), 
                                     columns=LABEL_LIST,
                                     index=index_tensor.to(int).tolist())
    prediction_valid_df.to_csv(VALIDATION_FILE_PATH)
    label_valid_df.to_csv(VALIDATION_DATASET_FILE_PATH)
    logger.success(f"Validation predictions exported !")
export_validation(model)


