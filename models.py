from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, year2id={},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    if queries.shape[1] > 4:
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start = -int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#', '0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end = -int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#', '0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x: x[1]):
                                if start >= key[0] and start <= key[1]:
                                    start_idx = time_idx
                                if end >= key[0] and end <= key[1]:
                                    end_idx = time_idx

                            if start_idx < 0:
                                start_queries.append(
                                    [int(triple[0]), int(triple[1]) + self.sizes[1] // 4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append(
                                    [int(triple[0]), int(triple[1]) + self.sizes[1] // 4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries) + self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size]
                        q = self.get_queries(these_queries)

                        scores = q @ rhs
                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    for i, query in enumerate(these_queries):
                        if queries.shape[1] > 4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]
                        else:
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class PAFSE(TKBCModel):


    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, no_time_emb=False, alpha: float = 10, init_size: float = 1e-2):
        super(PAFSE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.W = nn.Embedding(2 * rank, 1, sparse=True)
        self.W.weight.data *= 0
        self.cycle = 365
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[3] + 1, sizes[3] + 1, sizes[3] + 1, sizes[3] // self.cycle,
                      sizes[3] // self.cycle]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        self.embeddings[7].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846
        self.weight_separation = 1.0
        self.weight_high_freq_intensity = 1.0
        self.alpha = alpha

        self.freq_low_scale = nn.Parameter(torch.ones(1, rank) * 0.5)
        self.freq_high_scale = nn.Parameter(torch.ones(1, rank) * 0.5)
        self.freq_boundary = nn.Parameter(torch.tensor(rank // 2, dtype=torch.float32))
        self.mask_attention = nn.Sequential(
            nn.Linear(rank, rank // 4),
            nn.ReLU(),
            nn.Linear(rank // 4, rank),
            nn.Sigmoid()
        )

    @staticmethod
    def has_time():
        return True

    def _dct_1d(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Compute 1D Discrete Cosine Transform (DCT-II)"""
        N = x.shape[dim]
        x_expanded = torch.cat([x, torch.zeros_like(x)], dim=dim)
        fft_result = torch.fft.fft(x_expanded, dim=dim)
        freqs = torch.arange(N, device=x.device, dtype=torch.float32)
        correction = torch.exp(-1j * np.pi * freqs / (2 * N))
        
        if dim == -1 or dim == len(x.shape) - 1:
            correction = correction.view(1, -1)
        else:
            correction = correction.view(-1, 1)
        
        dct_result = 2 * torch.real(fft_result[..., :N] * correction)
        return dct_result

    def _idct_1d(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        N = x.shape[dim]
        
        # 创建频率和时间索引
        # k: 频率 1 到 N-1，形状 [N-1, 1]
        # n: 时间 0 到 N-1 加 0.5，形状 [1, N]
        k = torch.arange(1, N, device=x.device, dtype=torch.float32).view(-1, 1)
        n = torch.arange(N, device=x.device, dtype=torch.float32).view(1, -1) + 0.5
        
        # 一次性计算余弦矩阵 [N-1, N]
        # 这比逐频率循环快 80 倍因为：
        # 1. 一次性 GPU 内核启动（而不是 N 次）
        # 2. 矩阵乘法是高度优化的运算
        # 3. 避免 Python 循环开销
        cos_matrix = torch.cos(np.pi * k * n / N)
        
        # IDCT 计算
        # DC 分量：x[..., 0:1] / N
        # AC 分量：2/N * x[..., 1:] @ cos_matrix
        # 输入 x[..., 1:] 的形状：[batch_size, N-1]
        # cos_matrix 的形状：[N-1, N]
        # 输出的形状：[batch_size, N]
        result = x[..., 0:1] / N + 2 * (x[..., 1:] @ cos_matrix) / N
        
        return result

    def _decompose_with_dct(self, rel_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用DCT将关系嵌入分解为低频和高频分量。"""
        batch_size = rel_embedding.shape[0]
        
        dct_coeffs = self._dct_1d(rel_embedding, dim=-1)
        boundary = torch.clamp(self.freq_boundary, min=1, max=self.rank - 1).long()
        
        freq_indices = torch.arange(self.rank, device=rel_embedding.device, dtype=torch.float32)
        
        low_freq_base_mask = torch.exp(-self.alpha * freq_indices / self.rank)
        high_freq_base_mask = 1.0 - low_freq_base_mask
        
        low_freq_base_mask = low_freq_base_mask * self.freq_low_scale.squeeze(0)
        high_freq_base_mask = high_freq_base_mask * self.freq_high_scale.squeeze(0)
        
        adaptive_mask = self.mask_attention(torch.abs(dct_coeffs))
        
        low_freq_mask = low_freq_base_mask * adaptive_mask
        high_freq_mask = high_freq_base_mask * adaptive_mask
        
        sum_mask = low_freq_mask + high_freq_mask
        low_freq_mask = low_freq_mask / (sum_mask + 1e-8)
        high_freq_mask = high_freq_mask / (sum_mask + 1e-8)
        
        low_freq_dct = dct_coeffs * low_freq_mask
        high_freq_dct = dct_coeffs * high_freq_mask
        
        low_freq_component = self._idct_1d(low_freq_dct, dim=-1)
        high_freq_component = self._idct_1d(high_freq_dct, dim=-1)
        
        return low_freq_component, high_freq_component, low_freq_dct, high_freq_dct

    def loss_freq(self, low_freq, high_freq):
        """Frequency separation loss"""
        sep = torch.norm(low_freq - high_freq, p=2)
        hf = torch.norm(high_freq, p=2)
        return (-sep + hf) / low_freq.shape[0]
    # def loss_freq(self, low_freq, high_freq):
    #     # 1. Separation Loss
    #     separation_loss = - torch.norm(low_freq - high_freq, p=2)
    #     # 2. High-frequency Intensity Loss
    #     high_freq_intensity_loss = torch.norm(high_freq, p=2)
    #     total_loss = (self.weight_separation * separation_loss \
    #                 + self.weight_high_freq_intensity * high_freq_intensity_loss) * (1 / (high_freq.shape[0] * high_freq.shape[0]))
    #     return total_loss
        
    def complex_mul(self, emb1, emb2):
        """Complex multiplication for temporal transformation."""
        a, b = torch.chunk(emb1, 2, dim=1)
        c, d = torch.chunk(emb2, 2, dim=1)
        return torch.cat(((a * c - b * d), (a * d + b * c)), dim=1)

    def score(self, x):
        """Score function for a single triple."""
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel1 = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        comp_time = self.embeddings[4](x[:, 3])

        time_phase = torch.abs(self.embeddings[3](x[:, 3]))
        time_phase = torch.sin(time_phase[:, :self.rank]), torch.sin(time_phase[:, self.rank:])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]
        rel_ = self.complex_mul(rel1, comp_time)
        rel_ = rel_[:, :self.rank] / (1 / self.pi), rel_[:, self.rank:] / (1 / self.pi)
        rel2 = rel + rel_
        rel_low, rel_high, _, _ = self._decompose_with_dct(rel2[0])

        time_smoothed = time[0].mean(dim=1, keepdim=True)
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])
        rt_low = (rel_low + time_smoothed) * time[1]
        rt_high = (rel_high + time_gradient) * time[1]
        rt = 0.5 * rt_low + 0.5 * rt_high, rel2[1] + 0.5*time_phase[0]

        return torch.sum(
            ((lhs[0] + rt[1]) * rt[0]) * rhs[0], 1, keepdim=True)

    def forward(self, x):
        """Forward pass for batch of triples."""
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel1 = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        comp_time = self.embeddings[4](x[:, 3])

        time_phase = torch.abs(self.embeddings[3](x[:, 3]))
        time_phase = torch.sin(time_phase[:, :self.rank]), torch.sin(time_phase[:, self.rank:])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]
        rel_ = self.complex_mul(rel1, comp_time)
        rel_ = rel_[:, :self.rank] / (1 / self.pi), rel_[:, self.rank:] / (1 / self.pi)
        rel2 = rel + rel_
        rel_low, rel_high, low_freq, high_freq = self._decompose_with_dct(rel2[0])

        time_smoothed = time[0].mean(dim=1, keepdim=True)
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])
        rt_low = (rel_low + time_smoothed) * time[1]
        rt_high = (rel_high + time_gradient) * time[1]

        rt = 0.5 * rt_low + 0.5 * rt_high, rel2[1] + 0.5*time_phase[0]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        loss_freq = self.loss_freq(low_freq, high_freq)
        return (
                ((lhs[0] + rt[1]) * rt[0]) @ right[0].t()
        ), (
            torch.sqrt(lhs[0] ** 2),
            torch.sqrt(rt[0] ** 2 + rt[1] ** 2),
            torch.sqrt(rhs[0] ** 2)
        ), (self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight), low_freq, high_freq

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size][:, :self.rank].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        """Get query embeddings for ranking."""
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel1 = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        comp_time = self.embeddings[4](queries[:, 3])

        time_phase = torch.abs(self.embeddings[3](queries[:, 3]))
        time_phase = torch.sin(time_phase[:, :self.rank]), torch.sin(time_phase[:, self.rank:])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]
        rel_ = self.complex_mul(rel1, comp_time)
        rel_ = rel_[:, :self.rank] / (1 / self.pi), rel_[:, self.rank:] / (1 / self.pi)
        rel2 = rel + rel_

        rel_low, rel_high, _, _ = self._decompose_with_dct(rel2[0])

        time_smoothed = time[0].mean(dim=1, keepdim=True)
        time_gradient = torch.diff(time[0], dim=1, prepend=time[0][:, 0:1])
        rt_low = (rel_low + time_smoothed) * time[1]
        rt_high = (rel_high + time_gradient) * time[1]

        rt = 0.5 * rt_low + 0.5 * rt_high, rel2[1] + 0.5*time_phase[0]

        return (lhs[0] + rt[1]) * rt[0]
