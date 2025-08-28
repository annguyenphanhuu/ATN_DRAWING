import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm, global_mean_pool
import math
import numpy as np

class GeometricEncoder(nn.Module):
    """
    Advanced geometric encoder using Graph Attention Networks with residual connections.
    """
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels, num_heads=4):
        super(GeometricEncoder, self).__init__()

        # Node feature preprocessing
        self.node_preprocess = nn.Sequential(
            nn.Linear(node_in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Edge feature preprocessing
        self.edge_preprocess = nn.Sequential(
            nn.Linear(edge_in_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU()
        )

        # Graph attention layers with residual connections
        self.gat1 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads,
                           edge_dim=hidden_channels // 2, dropout=0.1)
        self.bn1 = BatchNorm(hidden_channels)

        self.gat2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads,
                           edge_dim=hidden_channels // 2, dropout=0.1)
        self.bn2 = BatchNorm(hidden_channels)

        self.gat3 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads,
                           edge_dim=hidden_channels // 2, dropout=0.1)
        self.bn3 = BatchNorm(hidden_channels)

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None):
        # Preprocess node features
        x = self.node_preprocess(x)
        residual = x

        # Preprocess edge features if available
        if edge_attr is not None:
            edge_attr = self.edge_preprocess(edge_attr)

        # First GAT layer with residual
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x + residual)
        residual = x

        # Second GAT layer with residual
        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x + residual)
        residual = x

        # Third GAT layer with residual
        x = self.gat3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x + residual)

        # Final projection
        x = self.final_proj(x)
        return x

class DimensionPredictor(nn.Module):
    """
    Advanced dimension line predictor using multiple interaction mechanisms.
    """
    def __init__(self, node_embedding_dim, hidden_dim=256):
        super(DimensionPredictor, self).__init__()

        # Multiple ways to combine node embeddings
        self.concat_mlp = nn.Sequential(
            nn.Linear(2 * node_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Hadamard (element-wise) product
        self.hadamard_mlp = nn.Sequential(
            nn.Linear(node_embedding_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        # Difference features
        self.diff_mlp = nn.Sequential(
            nn.Linear(node_embedding_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        # Geometric features (distance, angle, etc.)
        self.geometric_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # distance, angle, relative_x, relative_y
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )

        # Combine all features
        total_features = hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4
        self.final_mlp = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_embeddings, edge_candidates, node_positions):
        # node_embeddings: [num_nodes, embedding_dim]
        # edge_candidates: [2, num_edges]
        # node_positions: [num_nodes, 2] - normalized x, y coordinates

        start_embeddings = node_embeddings[edge_candidates[0]]  # [num_edges, embedding_dim]
        end_embeddings = node_embeddings[edge_candidates[1]]    # [num_edges, embedding_dim]

        start_positions = node_positions[edge_candidates[0]]    # [num_edges, 2]
        end_positions = node_positions[edge_candidates[1]]      # [num_edges, 2]

        # 1. Concatenation features
        concat_features = torch.cat([start_embeddings, end_embeddings], dim=1)
        concat_out = self.concat_mlp(concat_features)

        # 2. Hadamard product features
        hadamard_features = start_embeddings * end_embeddings
        hadamard_out = self.hadamard_mlp(hadamard_features)

        # 3. Difference features
        diff_features = torch.abs(start_embeddings - end_embeddings)
        diff_out = self.diff_mlp(diff_features)

        # 4. Geometric features
        relative_pos = end_positions - start_positions
        distances = torch.norm(relative_pos, dim=1, keepdim=True)
        angles = torch.atan2(relative_pos[:, 1:2], relative_pos[:, 0:1])

        geometric_features = torch.cat([
            distances, angles, relative_pos
        ], dim=1)
        geometric_out = self.geometric_mlp(geometric_features)

        # Combine all features
        combined_features = torch.cat([
            concat_out, hadamard_out, diff_out, geometric_out
        ], dim=1)

        # Final prediction
        logits = self.final_mlp(combined_features)
        return logits.squeeze(-1)

class DimensioningModel(nn.Module):
    """
    Complete model for dimension line prediction with advanced geometric understanding.
    """
    def __init__(self, encoder, predictor):
        super(DimensioningModel, self).__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x, edge_index, edge_candidates, edge_attr=None, node_positions=None):
        # 1. Get embeddings for all nodes in the graph
        node_embeddings = self.encoder(x, edge_index, edge_attr)

        # 2. Extract position features (first 2 dimensions of node features)
        if node_positions is None:
            node_positions = x[:, :2]  # Assume first 2 features are x, y coordinates

        # 3. Use the embeddings to predict the scores for candidate edges
        edge_logits = self.predictor(node_embeddings, edge_candidates, node_positions)
        return edge_logits

def build_model(node_feature_dim, edge_feature_dim, hidden_dim, embedding_dim):
    """
    Build the complete dimensioning model with advanced architecture.
    """
    encoder = GeometricEncoder(
        node_in_channels=node_feature_dim,
        edge_in_channels=edge_feature_dim,
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_heads=4
    )
    predictor = DimensionPredictor(
        node_embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )
    model = DimensioningModel(encoder, predictor)
    return model

class SmartNegativeSampler:
    """
    Intelligent negative sampling strategy for dimension line prediction.
    """
    def __init__(self, strategy='mixed'):
        self.strategy = strategy

    def sample_negatives(self, positive_edges, node_positions, num_nodes, num_negatives=None):
        """
        Sample negative edges using domain knowledge.
        """
        if num_negatives is None:
            num_negatives = len(positive_edges)

        positive_set = set(map(tuple, positive_edges))
        negatives = []

        # Strategy 1: Random sampling (baseline)
        random_negatives = self._random_sampling(positive_set, num_nodes, num_negatives // 3)
        negatives.extend(random_negatives)

        # Strategy 2: Distance-based hard negatives
        distance_negatives = self._distance_based_sampling(
            positive_edges, node_positions, positive_set, num_negatives // 3
        )
        negatives.extend(distance_negatives)

        # Strategy 3: Geometric pattern negatives
        pattern_negatives = self._pattern_based_sampling(
            positive_edges, node_positions, positive_set, num_negatives - len(negatives)
        )
        negatives.extend(pattern_negatives)

        return negatives[:num_negatives]

    def _random_sampling(self, positive_set, num_nodes, num_samples):
        negatives = []
        max_tries = num_samples * 10
        tries = 0

        while len(negatives) < num_samples and tries < max_tries:
            u, v = torch.randint(0, num_nodes, (2,)).tolist()
            if u != v:
                edge = tuple(sorted([u, v]))
                if edge not in positive_set:
                    negatives.append(list(edge))
            tries += 1

        return negatives

    def _distance_based_sampling(self, positive_edges, node_positions, positive_set, num_samples):
        """Sample negatives based on distance patterns from positives."""
        negatives = []

        if len(positive_edges) == 0:
            return negatives

        # Calculate distances of positive edges
        pos_distances = []
        for edge in positive_edges:
            u, v = edge
            dist = torch.norm(node_positions[u] - node_positions[v]).item()
            pos_distances.append(dist)

        mean_pos_dist = np.mean(pos_distances)
        std_pos_dist = np.std(pos_distances)

        # Sample negatives with similar distances but different node pairs
        num_nodes = len(node_positions)
        attempts = 0
        max_attempts = num_samples * 20

        while len(negatives) < num_samples and attempts < max_attempts:
            u, v = torch.randint(0, num_nodes, (2,)).tolist()
            if u != v:
                edge = tuple(sorted([u, v]))
                if edge not in positive_set:
                    dist = torch.norm(node_positions[u] - node_positions[v]).item()
                    # Accept if distance is within reasonable range of positive distances
                    if abs(dist - mean_pos_dist) < 2 * std_pos_dist:
                        negatives.append(list(edge))
            attempts += 1

        return negatives

    def _pattern_based_sampling(self, positive_edges, node_positions, positive_set, num_samples):
        """Sample negatives that break common geometric patterns."""
        negatives = []

        if len(positive_edges) == 0:
            return negatives

        # Find nodes that are endpoints of positive edges
        positive_nodes = set()
        for edge in positive_edges:
            positive_nodes.update(edge)

        positive_nodes = list(positive_nodes)
        num_nodes = len(node_positions)

        attempts = 0
        max_attempts = num_samples * 15

        while len(negatives) < num_samples and attempts < max_attempts:
            # Strategy: connect nodes that are not typically connected in technical drawings
            if len(positive_nodes) >= 2:
                # Sometimes sample from positive nodes (harder negatives)
                if torch.rand(1).item() < 0.7:
                    u = torch.randint(0, len(positive_nodes), (1,)).item()
                    v = torch.randint(0, len(positive_nodes), (1,)).item()
                    u, v = positive_nodes[u], positive_nodes[v]
                else:
                    u = torch.randint(0, num_nodes, (1,)).item()
                    v = torch.randint(0, num_nodes, (1,)).item()
            else:
                u = torch.randint(0, num_nodes, (1,)).item()
                v = torch.randint(0, num_nodes, (1,)).item()

            if u != v:
                edge = tuple(sorted([u, v]))
                if edge not in positive_set:
                    negatives.append(list(edge))
            attempts += 1

        return negatives

