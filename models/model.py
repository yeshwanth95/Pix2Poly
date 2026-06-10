import timm
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

import os
import sys
sys.path.insert(1, os.getcwd())

from config import CFG
from utils import (
    create_mask,
    log
)


# Borrowed from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py#L143
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

# Borrowed from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py#L152
def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class ScoreNet(nn.Module):
    def __init__(self, n_vertices, in_channels=512):
        super().__init__()
        self.n_vertices = n_vertices
        self.in_channels = in_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feats):
        feats = feats[:, 1:]
        feats = feats.unsqueeze(2)
        feats = feats.view(feats.size(0), feats.size(1)//2, 2, feats.size(3))
        feats = torch.mean(feats, dim=2)

        x = torch.transpose(feats, 1, 2)
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, 1, self.n_vertices)
        t = torch.transpose(x, 2, 3)
        x = torch.cat((x, t), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)

        return x[:, 0]


class Encoder(nn.Module):
    def __init__(self, model_name='deit3_small_patch16_384_in21ft1k', pretrained=False, out_dim=256) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            num_classes=0,
            global_pool='',
            pretrained=pretrained
        )
        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):
        features = self.model(x)
        return self.bottleneck(features[:, 1:])


class Decoder(nn.Module):
    def __init__(self, cfg, vocab_size, encoder_len, dim, num_heads, num_layers):
        super().__init__()
        self.cfg = cfg
        self.dim = dim

        self.embedding = nn.Embedding(vocab_size, dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.cfg.MAX_LEN-1, dim) * .02)
        self.decoder_pos_drop = nn.Dropout(p=0.05)

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, encoder_len, dim) * .02)
        self.encoder_pos_drop = nn.Dropout(p=0.05)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'encoder_pos_embed' in name or 'decoder_pos_embed' in name:
                log("Skipping initialization of pos embed layers...")
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        trunc_normal_(self.encoder_pos_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)

    def forward(self, encoder_out, tgt):
        """
        encoder_out shape: (N, L, D)
        tgt shape: (N, L)
        """

        tgt_mask, tgt_padding_mask = create_mask(tgt, self.cfg.PAD_IDX)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )

        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )

        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(
            memory=encoder_out,
            tgt=tgt_embedding,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        preds = preds.transpose(0, 1)
        return self.output(preds), preds

    def predict(self, encoder_out, tgt):
        length = tgt.size(1)
        padding = torch.ones((tgt.size(0), self.cfg.MAX_LEN-length-1), device=tgt.device).fill_(self.cfg.PAD_IDX).long()
        tgt = torch.cat([tgt, padding], dim=1)
        tgt_mask, tgt_padding_mask = create_mask(tgt, self.cfg.PAD_IDX)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )

        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )

        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(
            memory=encoder_out,
            tgt=tgt_embedding,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask = tgt_padding_mask
        )

        preds = preds.transpose(0, 1)
        return self.output(preds)[:, length-1, :], preds


class EncoderDecoder(nn.Module):
    def __init__(self, cfg, encoder, decoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder
        self.scorenet1 = ScoreNet(self.cfg.N_VERTICES)
        self.scorenet2 = ScoreNet(self.cfg.N_VERTICES)
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, image, tgt):
        encoder_out = self.encoder(image)
        preds, feats = self.decoder(encoder_out, tgt)
        perm_mat1 = self.scorenet1(feats)
        perm_mat2 = self.scorenet2(feats)
        perm_mat = perm_mat1 + torch.transpose(perm_mat2, 1, 2)

        perm_mat = log_optimal_transport(perm_mat, self.bin_score, self.cfg.SINKHORN_ITERATIONS)[:, :perm_mat.shape[1], :perm_mat.shape[2]]
        perm_mat = F.softmax(perm_mat, dim=-1)  # NOTE: perhaps try gumbel softmax here?
        # perm_mat = F.gumbel_softmax(perm_mat, tau=1.0, hard=False)

        return preds, perm_mat

    def predict(self, image, tgt):
        encoder_out = self.encoder(image)
        preds, feats = self.decoder.predict(encoder_out, tgt)
        return preds, feats


if __name__ == "__main__":
    # run this script as main for debugging.
    from tokenizer import Tokenizer
    from torch.nn.utils.rnn import pad_sequence
    import numpy as np
    import torch
    from torch import nn

    image = torch.randn(1, 3, CFG.INPUT_HEIGHT, CFG.INPUT_WIDTH).to('cuda')

    n_vertices = 192
    gt_coords = np.random.randint(size=(n_vertices, 2), low=0, high=CFG.IMG_SIZE).astype(np.float32)
    # in dataset
    tokenizer = Tokenizer(num_classes=1, num_bins=CFG.NUM_BINS, width=CFG.IMG_SIZE, height=CFG.IMG_SIZE, max_len=CFG.MAX_LEN)
    gt_seqs, rand_idxs = tokenizer(gt_coords)
    # in dataloader collate
    gt_seqs = [torch.LongTensor(gt_seqs)]
    gt_seqs = pad_sequence(gt_seqs, batch_first=True, padding_value=tokenizer.PAD_code)
    pad = torch.ones(gt_seqs.size(0), CFG.MAX_LEN - gt_seqs.size(1)).fill_(tokenizer.PAD_code).long()
    gt_seqs = torch.cat([gt_seqs, pad], dim=1).to('cuda')
    # in train fn
    gt_seqs_input = gt_seqs[:, :-1]
    gt_seqs_expected = gt_seqs[:, 1:]
    CFG.PAD_IDX = tokenizer.PAD_code

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=False, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size, encoder_len=CFG.NUM_PATCHES, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder).to('cuda')
    vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=CFG.PAD_IDX)

    # Forward pass during training.
    preds_f, perm_mat, batch_polygons = model(image, gt_seqs_input)
    loss = vertex_loss_fn(preds_f.reshape(-1, preds_f.shape[-1]), gt_seqs_expected.reshape(-1))

    # Sequence generation during prediction.
    batch_preds = torch.ones(image.size(0), 1).fill_(tokenizer.BOS_code).long().to(CFG.DEVICE)
    batch_feats = torch.ones(image.size(0), 1).fill_(tokenizer.BOS_code).long().to(CFG.DEVICE)
    sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)

    out_coords = []
    out_confs = []

    confs = []
    with torch.no_grad():
        for i in range(1 + n_vertices*2):
            try:
                log(f"Iteration {i}")
                preds_p, feats_p = model.predict(image, batch_preds)
                if i % 2 == 0:
                    confs_ = torch.softmax(preds_p, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                    confs.append(confs_)
                preds_p = sample(preds_p)
                batch_preds = torch.cat([batch_preds, preds_p], dim=1)
            except:
                log(f"Error at iteration: {i}")
        perm_pred = model.scorenet(feats_p)

        # Postprocessing.
        EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
        invalid_idxs = ((EOS_idxs - 1) % 2 != 0).nonzero().view(-1)  # sanity check
        EOS_idxs[invalid_idxs] = 0

        all_coords = []
        all_confs = []
        for i, EOS_idx in enumerate(EOS_idxs.tolist()):
            if EOS_idx == 0:
                all_coords.append(None)
                all_confs.append(None)
                continue
            coords = tokenizer.decode(batch_preds[i, :EOS_idx+1])
            confs = [round(confs[j][i].item(), 3) for j in range(len(coords))]

            all_coords.append(coords)
            all_confs.append(confs)

        out_coords.extend(all_coords)
        out_confs.extend(out_confs)

    log(f"preds_f shape: {preds_f.shape}")
    log(f"preds_f grad: {preds_f.requires_grad}")
    log(f"preds_f min: {preds_f.min()}, max: {preds_f.max()}")

    log(f"perm_mat shape: {perm_mat.shape}")
    log(f"perm_mat grad: {perm_mat.requires_grad}")
    log(f"perm_mat min: {perm_mat.min()}, max: {preds_f.max()}")

    log(f"batch_preds shape: {batch_preds.shape}")
    log(f"batch_preds grad: {batch_preds.requires_grad}")
    log(f"batch_preds min: {batch_preds.min()}, max: {batch_preds.max()}")

    log(f"loss : {loss}")
    log(f"loss grad: {loss.requires_grad}")

