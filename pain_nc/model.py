"""PAIN layers and a node-wise classification head.

This follows the official ExpressivePathGNNs architecture through its final
node embeddings. The graph-level readout is deliberately omitted.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.checkpoint import checkpoint

from .data import PainGraph
from .link_data import PainLinkGraph


class PathAggregator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        path_length: int,
        lstm_depth: int,
        num_edge_types: int,
        *,
        mark_neighbors: bool,
        shortest_path_encoding: bool,
        aggregation: Literal["sum", "mean"],
        dropout: float,
        chunk_size: int,
        checkpoint_chunks: bool,
        lstm: nn.LSTM | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.path_length = path_length
        self.mark_neighbors = mark_neighbors
        self.shortest_path_encoding = shortest_path_encoding
        self.aggregation = aggregation
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim)
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        if shortest_path_encoding:
            # The reference reserves one additional distance/padding slot.
            self.distance_embedding = nn.Embedding(path_length + 2, hidden_dim)
        else:
            self.position_embedding = nn.Parameter(
                torch.empty(path_length + 1, 1, hidden_dim)
            )
            # Match the authors' PathGNN positional encoding initialization.
            nn.init.xavier_normal_(self.position_embedding)

        input_dim = hidden_dim * 3 + int(mark_neighbors)
        self.lstm = lstm or nn.LSTM(
            input_dim, hidden_dim, num_layers=lstm_depth, batch_first=False
        )

    def _device_chunk(
        self,
        tensor: torch.Tensor,
        start: int,
        stop: int,
        device: torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        chunk = tensor[..., start:stop]
        return chunk.to(device, dtype=dtype, non_blocking=True)

    @staticmethod
    def _deterministic_segment_add(
        aggregate: torch.Tensor,
        roots: torch.Tensor,
        messages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add root-sorted messages without CUDA atomic scatter operations."""
        if roots.numel() == 0:
            return aggregate, roots.new_zeros((0, 1), dtype=messages.dtype)
        unique_roots, lengths = torch.unique_consecutive(
            roots, return_counts=True
        )
        ends = torch.cumsum(lengths, dim=0)
        prefix = torch.cat(
            (messages.new_zeros((1, messages.shape[1])), messages.cumsum(dim=0)),
            dim=0,
        )
        segment_sums = prefix[ends] - prefix[ends - lengths]
        updated = aggregate.index_copy(
            0, unique_roots, aggregate[unique_roots] + segment_sums
        )
        return updated, lengths.to(messages.dtype).unsqueeze(1)

    def _encode_chunk(
        self,
        x: torch.Tensor,
        edge_type: torch.Tensor,
        path_index: torch.Tensor,
        path_lengths: torch.Tensor,
        path_edge_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        safe_nodes = path_index.clamp(min=0)
        node_features = x[safe_nodes]

        safe_edges = path_edge_idx.clamp(min=0)
        valid_edges = path_edge_idx >= 0
        edge_features = self.edge_embedding(edge_type[safe_edges])
        edge_features = edge_features * valid_edges.unsqueeze(-1)

        if self.shortest_path_encoding:
            position_features = self.distance_embedding(
                distances.clamp(min=0, max=self.path_length + 1).t()
            )
        else:
            position_features = self.position_embedding.expand(
                -1, path_index.shape[1], -1
            )

        pieces = [node_features, position_features, edge_features]
        if self.mark_neighbors:
            neighbors = neighbor_mask.t().clamp(min=0).to(x.dtype).unsqueeze(-1)
            pieces.append(neighbors)
        sequence = torch.cat(pieces, dim=-1)
        # The reference PathConv applies dropout to the concatenated path
        # representation before packing it for the LSTM.
        sequence = F.dropout(sequence, p=self.dropout, training=self.training)
        packed = pack_padded_sequence(
            sequence,
            path_lengths.cpu().long(),
            batch_first=False,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]

    def forward(self, x: torch.Tensor, graph: PainGraph) -> torch.Tensor:
        total = graph.num_paths
        aggregate = x.new_zeros((graph.num_nodes, self.hidden_dim))
        counts = x.new_zeros((graph.num_nodes, 1)) if self.aggregation == "mean" else None
        edge_type = graph.edge_type.to(
            x.device, dtype=torch.long, non_blocking=True
        )

        for start in range(0, total, self.chunk_size):
            stop = min(start + self.chunk_size, total)
            # DBLP artifacts use compact integer dtypes on disk/host memory.
            # Embedding lookup indices are widened only for the active GPU chunk.
            path_index = self._device_chunk(
                graph.path_index, start, stop, x.device, dtype=torch.long
            )
            path_edge_idx = self._device_chunk(
                graph.path_edge_idx, start, stop, x.device, dtype=torch.long
            )
            neighbor_mask = graph.neighbor_mask[start:stop].to(
                x.device, dtype=torch.long, non_blocking=True
            )
            distances = graph.distances[start:stop].to(
                x.device, dtype=torch.long, non_blocking=True
            )
            lengths = graph.path_lengths[start:stop]
            roots = graph.mask_index[start:stop].to(
                x.device, dtype=torch.long, non_blocking=True
            )

            def encode(
                features: torch.Tensor,
                chunk_path_index: torch.Tensor,
                chunk_lengths: torch.Tensor,
                chunk_path_edge_idx: torch.Tensor,
                chunk_neighbor_mask: torch.Tensor,
                chunk_distances: torch.Tensor,
            ) -> torch.Tensor:
                return self._encode_chunk(
                    features,
                    edge_type,
                    chunk_path_index,
                    chunk_lengths,
                    chunk_path_edge_idx,
                    chunk_neighbor_mask,
                    chunk_distances,
                )

            if self.training and self.checkpoint_chunks:
                # Pass every chunk-local tensor explicitly. Capturing these
                # values in the loop closure makes backward recomputation use
                # the final chunk for earlier checkpoint frames.
                messages = checkpoint(
                    encode,
                    x,
                    path_index,
                    lengths,
                    path_edge_idx,
                    neighbor_mask,
                    distances,
                    use_reentrant=False,
                )
            else:
                messages = encode(
                    x,
                    path_index,
                    lengths,
                    path_edge_idx,
                    neighbor_mask,
                    distances,
                )
            aggregate, segment_counts = self._deterministic_segment_add(
                aggregate, roots, messages
            )
            if counts is not None:
                unique_roots = torch.unique_consecutive(roots)
                counts = counts.index_copy(
                    0,
                    unique_roots,
                    counts[unique_roots] + segment_counts,
                )
        if counts is not None:
            aggregate = aggregate / counts.clamp_min(1)
        return aggregate


class PainLayer(nn.Module):
    def __init__(self, aggregator: PathAggregator, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, graph: PainGraph) -> torch.Tensor:
        messages = self.aggregator(x, graph)
        x = self.batch_norm(x + messages)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mlp(x)


class PainNodeClassifier(nn.Module):
    """PAIN with graph pooling replaced by a node-wise MLP classifier."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        num_node_types: int,
        num_edge_types: int,
        hidden_dim: int = 128,
        num_layers: int = 5,
        lstm_depth: int = 2,
        head_layers: int = 2,
        path_length: int = 3,
        dropout: float = 0.0,
        share_lstm: bool = True,
        mark_neighbors: bool = True,
        shortest_path_encoding: bool = False,
        path_aggregation: Literal["sum", "mean"] = "sum",
        jumping_knowledge: Literal["last", "mean", "concat"] = "last",
        path_chunk_size: int = 100_000,
        checkpoint_chunks: bool = True,
        use_node_types: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1 or head_layers < 1 or path_chunk_size < 1:
            raise ValueError("num_layers, head_layers, and path_chunk_size must be positive")
        self.num_layers = num_layers
        self.jumping_knowledge = jumping_knowledge
        self.use_node_types = use_node_types
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.node_type_embedding = (
            nn.Embedding(num_node_types, hidden_dim) if use_node_types else None
        )

        lstm_input_dim = hidden_dim * 3 + int(mark_neighbors)
        shared_lstm = (
            nn.LSTM(lstm_input_dim, hidden_dim, num_layers=lstm_depth, batch_first=False)
            if share_lstm
            else None
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            aggregator = PathAggregator(
                hidden_dim,
                path_length,
                lstm_depth,
                num_edge_types,
                mark_neighbors=mark_neighbors,
                shortest_path_encoding=shortest_path_encoding,
                aggregation=path_aggregation,
                dropout=dropout,
                chunk_size=path_chunk_size,
                checkpoint_chunks=checkpoint_chunks,
                lstm=shared_lstm,
            )
            self.layers.append(PainLayer(aggregator, hidden_dim, dropout))

        representation_dim = hidden_dim * num_layers if jumping_knowledge == "concat" else hidden_dim
        head = []
        for _ in range(head_layers - 1):
            # Match the reference final MLP order.
            head.extend(
                (
                    nn.Linear(representation_dim, hidden_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                )
            )
            representation_dim = hidden_dim
        head.append(nn.Linear(representation_dim, num_classes))
        self.classifier = nn.Sequential(*head)

    def node_embeddings(self, graph: PainGraph) -> torch.Tensor:
        x = self.node_encoder(graph.x)
        if self.node_type_embedding is not None:
            x = x + self.node_type_embedding(graph.node_type)
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, graph)
            layer_outputs.append(x)
        if self.jumping_knowledge == "last":
            return layer_outputs[-1]
        if self.jumping_knowledge == "mean":
            return torch.stack(layer_outputs).mean(dim=0)
        if self.jumping_knowledge == "concat":
            return torch.cat(layer_outputs, dim=-1)
        raise ValueError(f"Unknown jumping_knowledge={self.jumping_knowledge!r}")

    def forward(self, graph: PainGraph) -> torch.Tensor:
        return self.classifier(self.node_embeddings(graph))


class PainLinkPredictor(nn.Module):
    """Faithful PAIN encoder with a dot-product link-prediction decoder.

    DBLP has no shared input-feature matrix in the existing benchmark contract,
    so the input representation is a learned node-id embedding plus the same
    learned node-type embedding used by PAIN-NC.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_types: int,
        num_edge_types: int,
        hidden_dim: int = 128,
        num_layers: int = 5,
        lstm_depth: int = 2,
        path_length: int = 3,
        dropout: float = 0.0,
        share_lstm: bool = True,
        mark_neighbors: bool = True,
        shortest_path_encoding: bool = False,
        path_aggregation: Literal["sum", "mean"] = "sum",
        jumping_knowledge: Literal["last", "mean", "concat"] = "last",
        path_chunk_size: int = 50_000,
        checkpoint_chunks: bool = True,
        use_node_types: bool = True,
        node_embedding_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1 or path_chunk_size < 1:
            raise ValueError("num_layers and path_chunk_size must be positive")
        if node_embedding_init_std <= 0:
            raise ValueError("node_embedding_init_std must be positive")
        self.num_layers = num_layers
        self.jumping_knowledge = jumping_knowledge
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.normal_(self.node_embedding.weight, std=node_embedding_init_std)
        self.node_type_embedding = (
            nn.Embedding(num_node_types, hidden_dim) if use_node_types else None
        )

        lstm_input_dim = hidden_dim * 3 + int(mark_neighbors)
        shared_lstm = (
            nn.LSTM(
                lstm_input_dim,
                hidden_dim,
                num_layers=lstm_depth,
                batch_first=False,
            )
            if share_lstm
            else None
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            aggregator = PathAggregator(
                hidden_dim,
                path_length,
                lstm_depth,
                num_edge_types,
                mark_neighbors=mark_neighbors,
                shortest_path_encoding=shortest_path_encoding,
                aggregation=path_aggregation,
                dropout=dropout,
                chunk_size=path_chunk_size,
                checkpoint_chunks=checkpoint_chunks,
                lstm=shared_lstm,
            )
            self.layers.append(PainLayer(aggregator, hidden_dim, dropout))

        if jumping_knowledge == "concat":
            self.decoder_projection: nn.Module = nn.Linear(
                hidden_dim * num_layers, hidden_dim
            )
        else:
            self.decoder_projection = nn.Identity()

    def node_embeddings(self, graph: PainLinkGraph) -> torch.Tensor:
        node_ids = torch.arange(graph.num_nodes, device=graph.node_type.device)
        x = self.node_embedding(node_ids)
        if self.node_type_embedding is not None:
            x = x + self.node_type_embedding(graph.node_type.long())
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, graph)
            layer_outputs.append(x)
        if self.jumping_knowledge == "last":
            representation = layer_outputs[-1]
        elif self.jumping_knowledge == "mean":
            representation = torch.stack(layer_outputs).mean(dim=0)
        elif self.jumping_knowledge == "concat":
            representation = torch.cat(layer_outputs, dim=-1)
        else:
            raise ValueError(
                f"Unknown jumping_knowledge={self.jumping_knowledge!r}"
            )
        return self.decoder_projection(representation)

    @staticmethod
    def score_pairs(embeddings: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return (
            embeddings[pairs[:, 0].long()]
            * embeddings[pairs[:, 1].long()]
        ).sum(dim=-1)

    def forward(self, graph: PainLinkGraph, pairs: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(self.node_embeddings(graph), pairs)
