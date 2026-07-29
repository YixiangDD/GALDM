"""Two-stage GA-LDM training and sampling.

Phase 1: train JointMultiOmicsVAE to learn the shared mRNA-miRNA latent space.
Phase 2: freeze the VAE and train GA-DiT in that latent space, with GDVD heterogeneous SNR,
         PCA-CTG causal constraints and the empirical consistency losses.
Sampling: DDIM denoising of the joint latent, VAE decode, de-standardise, post-process.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cross_modal import JointMultiOmicsVAE
from models.mirna_vae import MiRNAVAE
from models.ga_dit import GADiT, MLPDenoiser
from models.diffusion import Diffusion, GDVDSchedule
from models.postprocess import postprocess
from losses.structural import (CausalHingeLoss, TopKCorrLoss, PathwayActivityLoss,
                               PathwayCoexpressionLoss, MiRNAFamilyCorrLoss,
                               build_token_causal_mask, compute_topk_pairs,
                               compute_pathway_targets)
from bio_priors.segment_mapping import (build_dim_segment_map, build_token_layer_map,
                                        build_token_pathway_map)
from models.pa_vae import cyclical_beta


class EMA:
    def __init__(self, model, decay=0.9999):
        self.base_decay = decay
        self.decay = decay
        self.step = 0
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        # warmup: with few steps, track the live weights at a lower effective decay so a short
        # run does not leave the EMA sitting at initialisation
        self.step += 1
        d = min(self.base_decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)


class GALDMTrainer:
    def __init__(self, cfg, bio_priors, dataset, device="cuda",
                 ablation_flags=None):
        """ablation_flags: dict of module on/off switches used by the ablations.
        Everything is on by default, i.e. the full model."""
        self.cfg = cfg
        self.bp = bio_priors
        self.ds = dataset
        self.device = device
        f = dict(use_pavae=True, use_gadit=True, use_gdvd=True, use_pcactg=True,
                 use_consistency=True, use_crossmodal=True)
        if ablation_flags:
            f.update(ablation_flags)
        self.flags = f
        self._build()

    def _build(self):
        bp, cfg = self.bp, self.cfg
        # VAE; with use_pavae=False this degenerates to a plain dense VAE on a trivial mask
        pmask = bp.pathway_mask
        fmask = bp.mirna_family_mask
        if not self.flags["use_pavae"]:
            # plain VAE: one group covering every gene, no pathway structure
            pmask = np.ones((1, pmask.shape[1]), dtype=np.float32)
        self.vae = JointMultiOmicsVAE(pmask, fmask,
                                      mrna_latent=cfg.pavae.mrna_latent_dim,
                                      mirna_latent=cfg.pavae.mirna_latent_dim,
                                      common_dim=cfg.pavae.cross_modal_dim,
                                      use_cross_modal=self.flags["use_crossmodal"],
                                      lambda_cm=self.flags.get("lambda_cm", 1.0)).to(self.device)
        self.joint_dim = self.vae.joint_dim

        # segment mapping, used by GDVD and PCA-CTG
        K = cfg.data.n_pathways
        M = cfg.data.n_mirna_families
        self.seg_of_dim = build_dim_segment_map(
            cfg.pavae.mrna_latent_dim, cfg.pavae.cross_modal_dim, K, M, mrna_blocks=K)
        node_layers = np.array(bp.grn.node_layers)
        causal_depth_full = np.array(bp.grn.causal_depth)
        token_layer = build_token_layer_map(self.seg_of_dim, node_layers, cfg.gadit.n_tokens)
        token_pw = build_token_pathway_map(self.seg_of_dim, cfg.gadit.n_tokens, K, M)

        # denoising backbone: GA-DiT when use_gadit=True, a plain MLP otherwise (ablations M0/M1)
        n_pw_emb = M + K + 1
        if self.flags["use_gadit"]:
            self.dit = GADiT(self.joint_dim, n_tokens=cfg.gadit.n_tokens,
                             hidden_dim=cfg.gadit.hidden_dim, n_blocks=cfg.gadit.n_blocks,
                             n_heads=cfg.gadit.n_heads, ffn_dim=cfg.gadit.ffn_dim,
                             cond_dim=cfg.gadit.cond_dim, n_classes=2,
                             n_pathways=n_pw_emb, n_activity=K,
                             use_pathway_identity=self.flags.get("use_pathway_identity", True)).to(self.device)
            self.dit.set_token_pathway(token_pw)
            # PCA-CTG causal mask; only GA-DiT supports an attention mask
            if self.flags["use_pcactg"]:
                self.dit.set_causal_mask(build_token_causal_mask(token_layer))
        else:
            self.dit = MLPDenoiser(self.joint_dim, hidden_dim=512,
                                   cond_dim=cfg.gadit.cond_dim, n_classes=2,
                                   n_activity=K).to(self.device)

        # GDVD, using causal depths padded with the background entry to match seg_of_dim
        gdvd = GDVDSchedule(self.joint_dim, self.seg_of_dim, causal_depth_full,
                            enabled=self.flags["use_gdvd"],
                            depth_scale=cfg.gdvd.depth_offset_scale).to(self.device)
        self.diff = Diffusion(self.joint_dim, cfg.diffusion.timesteps,
                              cfg.diffusion.cosine_s, cfg.diffusion.min_snr_gamma,
                              gdvd=gdvd if self.flags["use_gdvd"] else None).to(self.device)

        # empirical consistency loss components
        self.consistency = None
        if self.flags["use_consistency"]:
            mrna_tr = self.ds["train"]["mrna"]
            pairs, tgt = compute_topk_pairs(mrna_tr, cfg.consistency.top_k_pairs)
            pw15 = bp.pathway_mask[:K]   # background excluded
            pmean, pvar = compute_pathway_targets(mrna_tr, pw15)
            self.topk_loss = TopKCorrLoss(pairs, tgt).to(self.device)
            self.pathway_loss = PathwayActivityLoss(pw15, pmean, pvar).to(self.device)
            self.coexpr_loss = PathwayCoexpressionLoss(pw15, mrna_tr).to(self.device)
            self.mirna_corr_loss = MiRNAFamilyCorrLoss(
                bp.mirna_family_mask, self.ds["train"]["mirna"]).to(self.device)
        # PCA-CTG hinge loss
        self.hinge = None
        if self.flags["use_pcactg"]:
            self.hinge = CausalHingeLoss(self.seg_of_dim, bp.grn.activating_edges,
                                         bp.grn.suppressive_edges,
                                         cfg.pcactg.hinge_margin).to(self.device)
        # latent standardisation statistics, filled in before Phase 2
        self.latent_mu = None
        self.latent_sd = None

    # ---------- Phase 1 ----------
    def train_vae(self, epochs=None, verbose=True):
        cfg = self.cfg
        epochs = epochs or cfg.train.vae_epochs
        xm = torch.tensor(self.ds["train"]["mrna"], device=self.device)
        xmi = torch.tensor(self.ds["train"]["mirna"], device=self.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=cfg.train.vae_lr)
        bs = cfg.train.vae_batch_size
        n = xm.shape[0]
        self.vae.train()
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            beta = cyclical_beta(ep, cfg.pavae.cycle_epochs, cfg.pavae.beta_max)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                rm, rmi, zj, stats = self.vae(xm[idx], xmi[idx])
                loss, _ = self.vae.loss(xm[idx], xmi[idx], rm, rmi, stats, beta,
                                        lambda_cross=cfg.consistency.lambda_cross_omic
                                        if self.flags["use_crossmodal"] else 0.0)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.vae.parameters(), cfg.train.grad_clip)
                opt.step()
                tot += loss.item()
            if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
                print(f"    [VAE] ep{ep} loss={tot/(n//bs+1):.4f} beta={beta:.3f}")
        # compute the latent standardisation statistics
        self.vae.eval()
        with torch.no_grad():
            zj, _ = self.vae.encode_joint(xm, xmi)
            self.latent_mu = zj.mean(0, keepdim=True)
            self.latent_sd = zj.std(0, keepdim=True).clamp(min=1e-4)
            # covariance of the standardised latent, used to recolour at sampling time so that
            # inter-dimension correlation structure is restored
            zn = ((zj - self.latent_mu) / self.latent_sd).cpu().numpy()
            cov = np.cov(zn, rowvar=False) + 1e-4 * np.eye(zn.shape[1])
            # matrix square root L_target, so that z_white @ L_target has the target covariance
            import scipy.linalg as sla
            self.color_target = torch.tensor(
                sla.sqrtm(cov).real.astype(np.float32), device=self.device)

    # ---------- Phase 2 ----------
    def train_dit(self, epochs=None, verbose=True):
        cfg = self.cfg
        epochs = epochs or cfg.train.dit_epochs
        xm = torch.tensor(self.ds["train"]["mrna"], device=self.device)
        xmi = torch.tensor(self.ds["train"]["mirna"], device=self.device)
        y = torch.tensor(self.ds["train"]["labels"], device=self.device)
        self.vae.eval()
        with torch.no_grad():
            z0_all, _ = self.vae.encode_joint(xm, xmi)
            z0_all = (z0_all - self.latent_mu) / self.latent_sd
        params = list(self.dit.parameters())
        if self.flags["use_gdvd"]:
            params += list(self.diff.gdvd.parameters())
            self.diff.gdvd.freeze()  # frozen until unfreeze_epoch
        opt = torch.optim.AdamW(params, lr=cfg.train.dit_lr,
                                weight_decay=cfg.train.dit_weight_decay)
        ema = EMA(self.dit, cfg.diffusion.ema_decay)
        bs = cfg.train.dit_batch_size
        n = z0_all.shape[0]
        # pathway-activity conditioning: real pathway activity, fed to adaLN
        K = cfg.data.n_pathways
        pw15 = torch.tensor(self.bp.pathway_mask[:K], device=self.device, dtype=torch.float32)
        counts = pw15.sum(1).clamp(min=1)
        activity_all = (xm @ pw15.T) / counts.unsqueeze(0)

        self.dit.train()
        for ep in range(epochs):
            if self.flags["use_gdvd"] and ep == cfg.gdvd.unfreeze_epoch:
                self.diff.gdvd.unfreeze()
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                z0 = z0_all[idx]
                t = torch.randint(1, cfg.diffusion.timesteps, (len(idx),), device=self.device)
                loss, eps_hat, z_t = self.diff.p_loss(
                    self.dit, z0, t, y[idx], activity_all[idx],
                    p_uncond=cfg.diffusion.p_uncond)
                # The structural losses (PCA-CTG hinge and empirical consistency) are more
                # meaningful at low noise levels, where z0_hat is clean. One shared one-step
                # prediction z0_lo at small t (<0.3T) serves both the latent-space causal hinge
                # and the decoded correlation losses.
                need_struct = ((self.flags["use_pcactg"] and self.hinge is not None) or
                               self.flags["use_consistency"])
                if need_struct and (i // bs) % cfg.consistency.consistency_every == 0:
                    t_lo = torch.randint(1, int(0.3 * cfg.diffusion.timesteps),
                                         (len(idx),), device=self.device)
                    z_lo, noise_lo, _ = self.diff.q_sample(z0, t_lo)
                    eps_lo = self.dit(z_lo, t_lo, y[idx], activity_all[idx])
                    ab_lo = self.diff.abar[t_lo].unsqueeze(1)
                    z0_lo = (z_lo - torch.sqrt(1 - ab_lo) * eps_lo) / torch.sqrt(ab_lo)
                    # PCA-CTG causal hinge: constrain edge directions on the clean latent z0_lo
                    if self.flags["use_pcactg"] and self.hinge is not None:
                        loss = loss + cfg.consistency.lambda_causal * self.hinge(z0_lo)
                    # empirical consistency: decode to gene space and match pathway/family
                    # correlation structure
                    if self.flags["use_consistency"]:
                        z0_dec = z0_lo * self.latent_sd + self.latent_mu
                        rm, rmi = self.vae.decode_joint(z0_dec)
                        loss = loss + cfg.consistency.lambda_corr * self.topk_loss(rm)
                        loss = loss + cfg.consistency.lambda_pathway * self.pathway_loss(rm)
                        loss = loss + cfg.consistency.lambda_coexpr * self.coexpr_loss(rm)
                        loss = loss + cfg.consistency.lambda_mirna_corr * self.mirna_corr_loss(rmi)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                opt.step(); ema.update(self.dit)
                tot += loss.item()
            if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
                print(f"    [DiT] ep{ep} loss={tot/(n//bs+1):.4f}")
        self.ema = ema
        self.activity_ref = activity_all

    # ---------- sampling ----------
    @torch.no_grad()
    def generate(self, n_samples, label=None, use_ema=True, postproc=True,
                 do_recolor=True):
        """do_recolor: latent whitening-recoloring (2nd-order latent calibration).
        postproc: expression-space quantile calibration (marginal alignment).
        Both default True (unchanged). Toggle for the calibration ablation (reviewer P0-3):
        raw=(recolor F, postproc F); +latent-cov=(T,F); +quantile=(F,T); full=(T,T)."""
        cfg = self.cfg
        model = self.dit
        if use_ema and hasattr(self, "ema"):
            model = copy.deepcopy(self.dit); self.ema.copy_to(model); model.eval()
        # class labels: by default sampled at the training class proportions; pass `label`
        # explicitly for balanced augmentation
        if label is None:
            y = self.ds["train"]["labels"]
            labels = torch.tensor(np.random.choice(y, n_samples), device=self.device)
        else:
            labels = torch.full((n_samples,), label, device=self.device, dtype=torch.long)
        # pathway-activity conditioning: resampled from same-class training samples, which
        # preserves the class-specific activity signal that DE depends on
        act_ref = self.activity_ref
        train_y = torch.tensor(self.ds["train"]["labels"], device=self.device)
        ridx = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        for c in torch.unique(labels):
            cls_pool = torch.where(train_y == c)[0]
            if len(cls_pool) == 0:
                cls_pool = torch.arange(act_ref.shape[0], device=self.device)
            sel = torch.where(labels == c)[0]
            ridx[sel] = cls_pool[torch.randint(0, len(cls_pool), (len(sel),), device=self.device)]
        activity = act_ref[ridx]
        z = self.diff.ddim_sample(model, n_samples, self.joint_dim,
                                  steps=cfg.diffusion.ddim_steps, device=self.device,
                                  class_label=labels, activity=activity,
                                  cfg_scale=cfg.diffusion.cfg_scale)
        # Latent distribution correction (whiten then recolour). Diffusion plus CFG leaves the
        # generated latent with a covariance structure that departs from the real encoded
        # latent, and wrong inter-dimension correlation is the main driver of miRNA_Corr and PCE
        # error. So: whiten the generated latent (centre, covariance -> I), then recolour with
        # the square root of the real latent covariance so the covariance matches.
        zc = z - z.mean(0, keepdim=True)
        cov_g = (zc.T @ zc) / (zc.shape[0] - 1) + 1e-4 * torch.eye(zc.shape[1], device=z.device)
        import scipy.linalg as sla
        inv_sqrt = torch.tensor(
            sla.sqrtm(np.linalg.inv(cov_g.cpu().numpy())).real.astype(np.float32),
            device=z.device)
        if do_recolor:
            z_white = zc @ inv_sqrt                   # covariance approximately I
            if getattr(self, "color_target", None) is not None:
                z = z_white @ self.color_target       # recolour to the real latent covariance
            else:
                z = z_white
        else:
            z = zc                                    # skip whiten-recolour (calibration ablation)
        z = z * self.latent_sd + self.latent_mu
        rm, rmi = self.vae.decode_joint(z)
        gm = rm.cpu().numpy(); gmi = rmi.cpu().numpy()
        labels_np = labels.cpu().numpy()
        # de-standardise back to the original log2 space
        norm = self.ds["norm"]
        gm = gm * norm["mrna_sd"] + norm["mrna_mu"]
        gmi = gmi * norm["mirna_sd"] + norm["mirna_mu"]
        if postproc:
            real_m = self.ds["train"]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
            real_mi = self.ds["train"]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
            gm, gmi, labels_np = postprocess(
                gm, gmi, real_m, real_mi,
                cosine_thresh=cfg.eval.quality_cosine_thresh,
                low_pct=cfg.eval.clip_low_pct, high_pct=cfg.eval.clip_high_pct,
                calib_strength=cfg.eval.calib_strength,
                gen_labels=labels_np, real_labels=self.ds["train"]["labels"])
        return gm, gmi, labels_np

    def real_original_space(self, split="test"):
        """Returns the real data for one split, in the original log2 space, for evaluation."""
        norm = self.ds["norm"]
        m = self.ds[split]["mrna"] * norm["mrna_sd"] + norm["mrna_mu"]
        mi = self.ds[split]["mirna"] * norm["mirna_sd"] + norm["mirna_mu"]
        return m, mi, self.ds[split]["labels"]
