"""Transparent C0 / v11 feature-modulation extension; no retired branches."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from .cross_ilda import ILALoss
from .lgmrec import LGMRec
from .msca_behavior import MSCABehaviorView
from .smore_fusion import CrossModalSpectrumFusion
from .behavior_hypergraph import BehaviorHypergraphRefiner

IMPLEMENTATION_VERSION = 'lgmrec-opt-behavior-hyper-v3'
FEAT_MOD_SCALE = 0.5
SCORE_MODES = ('original', 'separate', 'cross')
CROSS_BRANCH_PAIRS = (
    ('c', 'v'), ('c', 't'), ('c', 'g'),
    ('v', 't'), ('v', 'g'), ('t', 'g'),
)


class LGMRecOpt(LGMRec):
    """Default is mathematically C0. Feature modulation remains opt-in."""

    def __init__(self, config, dataset):
        if config['protocol'] != 'train_only':
            raise ValueError('Only train_only is supported')
        super().__init__(config, dataset)
        self.score_mode = config['score_mode'] or 'original'
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                'score_mode must be original, separate, or cross'
            )
        self.cross_logits = None
        if self.score_mode == 'cross':
            # 2 * sigmoid(0) == 1, so the initial cross score is the
            # algebraic expansion of the original summed-embedding score.
            self.cross_logits = nn.ParameterDict({
                f'{left}_{right}': nn.Parameter(torch.zeros(()))
                for left, right in CROSS_BRANCH_PAIRS
            })
        self.feat_mod_mode = config['feat_mod_mode'] or 'off'
        if self.feat_mod_mode not in ('off', 'id_cond'):
            raise ValueError('feat_mod_mode must be off or id_cond')
        self.train_context = config['train_context'] or 'full'
        if self.train_context not in ('full', 'target_mask'):
            raise ValueError('train_context must be full or target_mask')
        if self.train_context == 'target_mask':
            if self.score_mode != 'original':
                raise ValueError('target_mask requires score_mode=original')
            if self.feat_mod_mode != 'off':
                raise ValueError('target_mask requires feat_mod_mode=off')
            edge_indices = self.adj.coalesce().indices().detach().cpu()
            self._full_user_degrees = torch.bincount(
                edge_indices[0], minlength=self.n_users
            ).tolist()
            self._full_item_degrees = torch.bincount(
                edge_indices[1], minlength=self.n_items
            ).tolist()
        self.directed_align_mode = config['directed_align_mode'] or 'off'
        if self.directed_align_mode not in ('off', 'cross_ilda_dt'):
            raise ValueError(
                'directed_align_mode must be off or cross_ilda_dt'
            )
        self.align_weight = float(config['align_weight'])
        if not math.isfinite(self.align_weight) or self.align_weight < 0:
            raise ValueError('align_weight must be finite and non-negative')
        self.directed_align = None
        if self.directed_align_mode == 'cross_ilda_dt':
            if self.train_context != 'full':
                raise ValueError('cross_ilda_dt requires train_context=full')
            if self.score_mode != 'original':
                raise ValueError('cross_ilda_dt requires score_mode=original')
            if self.feat_mod_mode != 'off':
                raise ValueError('cross_ilda_dt requires feat_mod_mode=off')
            self.directed_align = ILALoss(
                dim=self.embedding_dim, gamma=0.007, leaky_bi=True
            )
        self.joint_fusion_mode = config['joint_fusion_mode'] or 'off'
        if self.joint_fusion_mode not in ('off', 'smore_cross'):
            raise ValueError(
                'joint_fusion_mode must be off or smore_cross'
            )
        self.fusion_beta = float(config['fusion_beta'])
        if not math.isfinite(self.fusion_beta) or self.fusion_beta < 0:
            raise ValueError('fusion_beta must be finite and non-negative')
        self.joint_fusion_gate = config['joint_fusion_gate'] or 'off'
        if self.joint_fusion_gate not in ('off', 'smore_prefer'):
            raise ValueError(
                'joint_fusion_gate must be off or smore_prefer'
            )
        if (self.joint_fusion_gate == 'smore_prefer'
                and self.joint_fusion_mode != 'smore_cross'):
            raise ValueError(
                'smore_prefer requires joint_fusion_mode=smore_cross'
            )
        self.smore_fusion = None
        self.smore_preference_gate = None
        if self.joint_fusion_mode == 'smore_cross':
            if self.directed_align_mode != 'off':
                raise ValueError('smore_cross requires directed_align_mode=off')
            if self.train_context != 'full':
                raise ValueError('smore_cross requires train_context=full')
            if self.score_mode != 'original':
                raise ValueError('smore_cross requires score_mode=original')
            if self.feat_mod_mode != 'off':
                raise ValueError('smore_cross requires feat_mod_mode=off')
            with torch.random.fork_rng(devices=[]):
                self.smore_fusion = CrossModalSpectrumFusion(
                    self.feat_embed_dim
                )
            if self.joint_fusion_gate == 'smore_prefer':
                # SMORE gate_fusion_prefer: one shared gate for user/item
                # collaborative representations. Keep its initialization from
                # advancing the experiment's global RNG stream.
                with torch.random.fork_rng(devices=[]):
                    self.smore_preference_gate = nn.Sequential(
                        nn.Linear(self.embedding_dim, self.embedding_dim),
                        nn.Sigmoid(),
                    )
        self.behavior_view_mode = config['behavior_view_mode'] or 'off'
        if self.behavior_view_mode not in ('off', 'msca_struct'):
            raise ValueError(
                'behavior_view_mode must be off or msca_struct'
            )
        self.behavior_eta = float(config['behavior_eta'])
        if not math.isfinite(self.behavior_eta) or self.behavior_eta < 0:
            raise ValueError('behavior_eta must be finite and non-negative')
        self.behavior_view = None
        self.behavior_residual_target = config['behavior_residual_target'] or 'both'
        if self.behavior_residual_target not in ('both', 'item', 'user'):
            raise ValueError('behavior_residual_target must be both, item, or user')
        if self.behavior_view_mode == 'msca_struct':
            if self.joint_fusion_mode != 'off':
                raise ValueError('msca_struct requires joint_fusion_mode=off')
            if self.directed_align_mode != 'off':
                raise ValueError('msca_struct requires directed_align_mode=off')
            if self.train_context != 'full':
                raise ValueError('msca_struct requires train_context=full')
            if self.score_mode != 'original':
                raise ValueError('msca_struct requires score_mode=original')
            if self.feat_mod_mode != 'off':
                raise ValueError('msca_struct requires feat_mod_mode=off')
            self.behavior_view = MSCABehaviorView(
                self.interaction_matrix, self.device,
                topk=config['behavior_topk'], minimum=config['behavior_minimum'],
                graph_mode=config['behavior_graph_mode'], graph_seed=config['behavior_graph_seed'],
                self_loop=config['behavior_self_loop'], edge_weight=config['behavior_edge_weight'],
                block_size=config['behavior_block_size'],
            )
        self.hyper_behavior_weight = float(config['hyper_behavior_weight'])
        if not math.isfinite(self.hyper_behavior_weight) or not 0 <= self.hyper_behavior_weight <= 1:
            raise ValueError('hyper_behavior_weight must be finite and in [0, 1]')
        hyper_mode = config['hyper_behavior_graph_mode']
        if hyper_mode not in ('cooccurrence', 'random_relabel'):
            raise ValueError('Unknown hyper_behavior_graph_mode')
        self.hyper_behavior = None
        if self.hyper_behavior_weight > 0:
            if self.behavior_view is None:
                raise ValueError('Hypergraph refinement requires behavior_view_mode=msca_struct')
            if config['behavior_graph_mode'] != 'cooccurrence':
                raise ValueError('Keep the residual graph real; randomize hyper_behavior_graph_mode only')
            if self.n_hyper_layer < 1 or self.alpha <= 0:
                raise ValueError('Hypergraph refinement requires active global hypergraph propagation')
            self.hyper_behavior = BehaviorHypergraphRefiner(
                self.behavior_view.structural_adjacency, self.hyper_behavior_weight,
                graph_mode=hyper_mode, seed=config['behavior_graph_seed'],
            )
        value = config['lambda_hcl']
        self.lambda_hcl = self.cl_weight if value is None else float(value)
        if not math.isfinite(self.lambda_hcl) or self.lambda_hcl < 0:
            raise ValueError('lambda_hcl must be finite and non-negative')
        if self.v_feat is None or self.t_feat is None:
            raise ValueError('Both modality feature arrays are required')
        if not torch.isfinite(self.v_feat).all() or not torch.isfinite(self.t_feat).all():
            raise ValueError('Features must be finite')
        self.feat_mod = None
        if self.feat_mod_mode == 'id_cond':
            with torch.random.fork_rng(devices=[]):
                layers = {kind: nn.Linear(self.embedding_dim, self.feat_embed_dim)
                          for kind in ('v', 't')}
            for layer in layers.values():
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            self.feat_mod = nn.ModuleDict(layers)
        self.pre_epoch_processing()

    def hyperedge_logits(self, features, projection):
        logits = super().hyperedge_logits(features, projection)
        return logits if self.hyper_behavior is None else self.hyper_behavior(logits)

    def mge(self, str='v'):
        if self.feat_mod is None:
            features = super().mge(str)
            if self.smore_fusion is not None:
                if str == 'v':
                    self._smore_visual_items = torch.mm(
                        self.image_embedding.weight, self.item_image_trs
                    )
                elif str == 't':
                    self._smore_text_items = torch.mm(
                        self.text_embedding.weight, self.item_text_trs
                    )
        else:
            if str == 'v':
                items = self.image_embedding.weight @ self.item_image_trs
            elif str == 't':
                items = self.text_embedding.weight @ self.item_text_trs
            else:
                raise ValueError('Expected v or t')
            condition = F.normalize(self.item_id_embedding.weight.detach(), dim=-1)
            items = items * (1 + FEAT_MOD_SCALE * torch.tanh(self.feat_mod[str](condition)))
            users = torch.sparse.mm(self.adj, items) * self.num_inters[:self.n_users]
            features = torch.cat((users, items), dim=0)
            for _ in range(self.n_mm_layer):
                features = torch.sparse.mm(self.norm_adj, features)
        if str == 'v':
            self._score_visual = features
        elif str == 't':
            self._score_text = features
        else:
            raise ValueError('Expected v or t')
        return features

    def cge(self):
        features = super().cge()
        self._score_collaborative = features
        return features

    def fuse_modal_embeddings(self, visual_embeddings, text_embeddings):
        visual = F.normalize(visual_embeddings)
        text = F.normalize(text_embeddings)
        self._score_visual = visual
        self._score_text = text
        return visual + text

    def fuse_hyper_embeddings(self, visual_embeddings, text_embeddings):
        features = super().fuse_hyper_embeddings(
            visual_embeddings, text_embeddings
        )
        self._score_hyper = features
        return features

    def forward(self):
        """Preserve the public base interface while retaining C/V/T/G."""
        users, items, hyper = super().forward()
        global_features = (
            self.global_hypergraph_scale()
            * F.normalize(self._score_hyper)
        )
        all_branches = (
            self._score_collaborative,
            self._score_visual,
            self._score_text,
            global_features,
        )
        self._score_user_branches = tuple(
            branch[:self.n_users] for branch in all_branches
        )
        self._score_item_branches = tuple(
            branch[self.n_users:] for branch in all_branches
        )
        if self.smore_fusion is not None:
            joint = self._smore_joint_mge()
            joint = F.normalize(joint)
            if self.smore_preference_gate is not None:
                gate = self.smore_preference_gate(
                    self._score_collaborative
                )
                joint = gate * joint
            joint_users, joint_items = torch.split(
                joint, [self.n_users, self.n_items], dim=0
            )
            users = users + self.fusion_beta * joint_users
            items = items + self.fusion_beta * joint_items
        if self.behavior_view is not None:
            behavior = self.behavior_view(self.item_id_embedding.weight)
            behavior_users, behavior_items = torch.split(
                behavior, [self.n_users, self.n_items], dim=0
            )
            if self.behavior_residual_target in ('both', 'user'):
                users = users + self.behavior_eta * behavior_users
            if self.behavior_residual_target in ('both', 'item'):
                items = items + self.behavior_eta * behavior_items
        return users, items, hyper

    def _smore_joint_mge(self):
        """Apply SMORE fusion, then the unchanged LGMRec MGE propagation."""
        joint_items = self.smore_fusion(
            self._smore_visual_items, self._smore_text_items
        )
        joint_users = (
            torch.sparse.mm(self.adj, joint_items)
            * self.num_inters[:self.n_users]
        )
        joint = torch.cat((joint_users, joint_items), dim=0)
        for _ in range(self.n_mm_layer):
            joint = torch.sparse.mm(self.norm_adj, joint)
        return joint

    @staticmethod
    def _dot(users, items, full_sort):
        if full_sort:
            return torch.matmul(users, items.T)
        return torch.sum(torch.mul(users, items), dim=1)

    def score(self, user_branches, item_branches, *, full_sort=False,
              combined=None):
        """Score aligned pairs or all items through the configured rule."""
        if self.score_mode == 'original':
            if combined is None:
                raise ValueError('original scoring requires combined embeddings')
            return self._dot(combined[0], combined[1], full_sort)

        branch_names = ('c', 'v', 't', 'g')
        users = dict(zip(branch_names, user_branches))
        items = dict(zip(branch_names, item_branches))
        result = sum(
            self._dot(users[name], items[name], full_sort)
            for name in branch_names
        )
        if self.score_mode == 'separate':
            return result

        for left, right in CROSS_BRANCH_PAIRS:
            coefficient = 2 * torch.sigmoid(
                self.cross_logits[f'{left}_{right}']
            )
            result = result + coefficient * (
                self._dot(users[left], items[right], full_sort)
                + self._dot(users[right], items[left], full_sort)
            )
        return result

    def _build_target_mask_graph(self, users, positive_items):
        """Build one batch graph with eligible, stable-order targets removed."""
        pairs = list(zip(
            users.detach().cpu().tolist(),
            positive_items.detach().cpu().tolist(),
        ))
        user_degrees = list(self._full_user_degrees)
        item_degrees = list(self._full_item_degrees)
        seen = set()
        unique_pairs = []
        masked_pairs = []
        for user, item in pairs:
            pair = (int(user), int(item))
            if pair in seen:
                continue
            seen.add(pair)
            unique_pairs.append(pair)
            if not (0 <= pair[0] < self.n_users
                    and 0 <= pair[1] < self.n_items):
                raise ValueError(f'Target edge is out of range: {pair}')
            if user_degrees[pair[0]] > 1 and item_degrees[pair[1]] > 1:
                masked_pairs.append(pair)
                user_degrees[pair[0]] -= 1
                item_degrees[pair[1]] -= 1

        full_adj = self.adj.coalesce()
        indices = full_adj.indices()
        values = full_adj.values()
        full_codes = indices[0] * self.n_items + indices[1]
        target_codes = torch.tensor(
            [user * self.n_items + item for user, item in unique_pairs],
            dtype=full_codes.dtype,
            device=full_codes.device,
        )
        if target_codes.numel() and not torch.isin(
                target_codes, full_codes).all().item():
            raise ValueError('Every target edge must exist in the TRAIN graph')
        if not masked_pairs:
            return None, None, None, len(unique_pairs), 0

        masked_codes = torch.tensor(
            [user * self.n_items + item for user, item in masked_pairs],
            dtype=full_codes.dtype,
            device=full_codes.device,
        )
        keep = ~torch.isin(full_codes, masked_codes)
        kept_indices = indices[:, keep]
        kept_values = values[keep]
        masked_adj = torch.sparse_coo_tensor(
            kept_indices,
            kept_values,
            full_adj.shape,
            device=full_adj.device,
        ).coalesce()

        user_degree = torch.bincount(
            kept_indices[0], minlength=self.n_users
        ).to(dtype=kept_values.dtype)
        item_degree = torch.bincount(
            kept_indices[1], minlength=self.n_items
        ).to(dtype=kept_values.dtype)
        edge_norm = torch.rsqrt(user_degree[kept_indices[0]]) * torch.rsqrt(
            item_degree[kept_indices[1]]
        )
        item_nodes = kept_indices[1] + self.n_users
        symmetric_indices = torch.cat((
            torch.stack((kept_indices[0], item_nodes)),
            torch.stack((item_nodes, kept_indices[0])),
        ), dim=1)
        masked_norm_adj = torch.sparse_coo_tensor(
            symmetric_indices,
            torch.cat((edge_norm, edge_norm)),
            self.norm_adj.shape,
            device=self.norm_adj.device,
        ).coalesce()
        degrees = torch.cat((user_degree, item_degree))
        masked_num_inters = 1.0 / (degrees.unsqueeze(1) + 1e-7)
        return (masked_adj, masked_norm_adj, masked_num_inters,
                len(unique_pairs), len(masked_pairs))

    @contextmanager
    def _target_mask_graph(self, users, positive_items):
        graph = self._build_target_mask_graph(users, positive_items)
        masked_adj, masked_norm_adj, masked_num_inters, total, masked = graph
        if masked_adj is None:
            yield total, masked
            return
        full_graph = self.adj, self.norm_adj, self.num_inters
        self.adj = masked_adj
        self.norm_adj = masked_norm_adj
        self.num_inters = masked_num_inters
        try:
            yield total, masked
        finally:
            self.adj, self.norm_adj, self.num_inters = full_graph

    def calculate_loss(self, interaction):
        if interaction.size(0) != 3:
            raise ValueError('Expected one TRAIN-only negative per positive')
        uid, positive, negative = interaction.long()
        if self.training and self.train_context == 'target_mask':
            with self._target_mask_graph(uid, positive) as mask_counts:
                users, items, hyper = self.forward()
            self._target_candidates += mask_counts[0]
            self._target_masked += mask_counts[1]
        else:
            users, items, hyper = self.forward()
        u, p, n = users[uid], items[positive], items[negative]
        user_branches = tuple(branch[uid] for branch in self._score_user_branches)
        positive_branches = tuple(
            branch[positive] for branch in self._score_item_branches
        )
        negative_branches = tuple(
            branch[negative] for branch in self._score_item_branches
        )
        positive_scores = self.score(
            user_branches, positive_branches, combined=(u, p)
        )
        negative_scores = self.score(
            user_branches, negative_branches, combined=(u, n)
        )
        bpr = -torch.mean(F.logsigmoid(positive_scores - negative_scores))
        uv, iv, ut, it = hyper
        hcl = (self.ssl_triple_loss(uv[uid], ut[uid], ut)
               + self.ssl_triple_loss(iv[positive], it[positive], it))
        reg = self.reg_weight * self.reg_loss(u, p, n)
        weighted = self.lambda_hcl * hcl
        align = None
        weighted_align = None
        if self.directed_align is not None:
            collaborative_users = self._score_user_branches[0]
            collaborative_items = self._score_item_branches[0]
            visual_items = self._score_item_branches[1]
            text_items = self._score_item_branches[2]
            align = self.directed_align(
                collaborative_users,
                collaborative_items,
                visual_items,
                text_items,
                uid,
                positive,
            )
            if not torch.isfinite(align):
                raise RuntimeError('CROSS ILDA+DT loss is non-finite')
            weighted_align = self.align_weight * align
        if self.training:
            count = uid.numel()
            diagnostics = dict(
                loss_bpr=bpr,
                loss_hcl_raw=hcl,
                loss_hcl_weighted=weighted,
                loss_reg=reg,
            )
            if align is not None:
                diagnostics.update(
                    loss_ilda_dt_raw=align,
                    loss_ilda_dt_weighted=weighted_align,
                )
            for key, value in diagnostics.items():
                self._sums[key] = self._sums.get(key, 0.) + float(value.detach()) * count
            self._count += count
        original_loss = bpr + weighted + reg
        if weighted_align is not None:
            original_loss = original_loss + weighted_align
        return original_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        users, items, _ = self.forward()
        user_branches = tuple(
            branch[user] for branch in self._score_user_branches
        )
        return self.score(
            user_branches,
            self._score_item_branches,
            full_sort=True,
            combined=(users[user], items),
        )

    def pre_epoch_processing(self):
        self._sums, self._count = {}, 0
        self._target_candidates, self._target_masked = 0, 0

    def post_epoch_processing(self):
        values = [
            f'{key}={value / max(1, self._count):.6f}'
            for key, value in self._sums.items()
        ]
        if self.train_context == 'target_mask':
            values.append(
                'target_mask_ratio='
                f'{self._target_masked / max(1, self._target_candidates):.6f}'
            )
            values.append(f'target_masked={self._target_masked}')
            values.append(f'target_candidates={self._target_candidates}')
        return 'LGMRecOpt diagnostics: ' + ', '.join(values)
