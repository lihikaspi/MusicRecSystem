import duckdb
import torch
from torch_geometric.data import HeteroData
import numpy as np

class GraphBuilder:
    """
    Class to construct the graph used for the GNN model.
    Assumes preprocessing has already filtered and encoded all IDs.
    Uses item_train_idx for contiguous item IDs in training, but keeps original item_idx for mapping back.
    """
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def _build_graph(self):
        # Load edges, replace item_idx with item_train_idx where available
        edges_df = self.con.execute("""
            SELECT user_idx,
                   COALESCE(item_train_idx, -1) AS item_train_idx,
                   item_idx AS original_item_idx,
                   edge_count, edge_avg_played_ratio, edge_type, edge_weight
            FROM agg_edges_artist_album
            JOIN agg_edges USING(item_idx)
        """).fetch_df()

        # Filter out edges pointing to items not in train (optional: keep them with -1 if needed)
        edges_df = edges_df[edges_df['item_train_idx'] >= 0].copy()

        # Edge index
        edge_index_np = np.vstack((edges_df['user_idx'].values, edges_df['item_train_idx'].values))
        edge_index = torch.from_numpy(edge_index_np).long()

        # Edge attributes
        edge_attr = torch.tensor(
            edges_df[['edge_type', 'edge_count', 'edge_avg_played_ratio']].fillna(0).values,
            dtype=torch.float
        )

        # Edge weights
        edge_weight_init = torch.tensor(edges_df['edge_weight'].fillna(0).values, dtype=torch.float)

        # Load item node features
        item_embeddings_df = self.con.execute("""
            SELECT item_train_idx, item_normalized_embed, artist_idx, album_idx, item_idx AS original_item_idx
            FROM agg_edges_artist_album
            WHERE item_train_idx >= 0
            ORDER BY item_train_idx
        """).fetch_df()

        # Build HeteroData
        data = HeteroData()
        data['user', 'interacts', 'item'].edge_index = edge_index
        data['user', 'interacts', 'item'].edge_attr = edge_attr
        data['user', 'interacts', 'item'].edge_weight_init = edge_weight_init

        data['item'].x = torch.tensor(np.vstack(item_embeddings_df['item_normalized_embed'].values), dtype=torch.float)
        data['item'].artist_id = torch.tensor(item_embeddings_df['artist_idx'].values, dtype=torch.long)
        data['item'].album_id = torch.tensor(item_embeddings_df['album_idx'].values, dtype=torch.long)
        data['item'].item_idx = torch.tensor(item_embeddings_df['original_item_idx'].values, dtype=torch.long)  # keep original for mapping back
        data['item'].num_nodes = len(item_embeddings_df)

        # Users
        num_users = edges_df['user_idx'].max() + 1
        data['user'].user_id = torch.arange(num_users, dtype=torch.long)
        data['user'].num_nodes = num_users

        print(f"[INFO] Graph created: num_users={num_users}, num_items={len(item_embeddings_df)}")
        print(f"[DEBUG] Edge index min/max: {edge_index.min().item()}/{edge_index.max().item()}")
        print(f"[DEBUG] Total nodes (users + items): {num_users + len(item_embeddings_df)}")

        return data

    def save_graph(self, output_path: str):
        data = self._build_graph()
        torch.save(data, output_path)
        print(f"[INFO] Graph saved to {output_path}")
