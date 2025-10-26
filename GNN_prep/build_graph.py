import duckdb
import torch
from torch_geometric.data import HeteroData
import numpy as np
import json
from pathlib import Path

class GraphBuilder:
    """
    Class to construct the graph used for the GNN model.
    Assigns local contiguous user/item indices and stores mappings.
    """
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def _build_graph(self):
        """
        Builds the graph using the edges file and assigns local indices.
        """
        # === 1️⃣ Load raw edges ===
        edges_df = self.con.execute("""
            SELECT user_idx, item_idx, edge_count, edge_avg_played_ratio, edge_type, edge_weight
            FROM agg_edges
        """).fetch_df()

        # === 2️⃣ Create local ID mappings ===
        unique_users = edges_df['user_idx'].unique()
        unique_items = edges_df['item_idx'].unique()

        user_id_map = {uid: i for i, uid in enumerate(unique_users)}
        item_id_map = {iid: i for i, iid in enumerate(unique_items)}

        inv_user_id_map = {v: k for k, v in user_id_map.items()}
        inv_item_id_map = {v: k for k, v in item_id_map.items()}

        # === 3️⃣ Apply mappings ===
        edges_df['user_local'] = edges_df['user_idx'].map(user_id_map)
        edges_df['item_local'] = edges_df['item_idx'].map(item_id_map)

        # === 4️⃣ Build edge index ===
        edge_index_np = np.vstack((edges_df['user_local'].values, edges_df['item_local'].values))
        edge_index = torch.from_numpy(edge_index_np).long()

        # === 5️⃣ Build edge attributes ===
        edge_attr = torch.tensor(
            edges_df[['edge_type', 'edge_count', 'edge_avg_played_ratio']].fillna(0).values,
            dtype=torch.float
        )

        edge_weight_init = torch.tensor(
            edges_df['edge_weight'].fillna(0).values,
            dtype=torch.float
        )

        # === 6️⃣ Load item embeddings ===
        item_embeddings_df = self.con.execute("""
            SELECT item_idx, item_normalized_embed, artist_idx, album_idx
            FROM agg_edges_artist_album
            GROUP BY item_idx, item_normalized_embed, artist_idx, album_idx
        """).fetch_df()

        # Filter only items that appear in the local mapping
        item_embeddings_df = item_embeddings_df[item_embeddings_df['item_idx'].isin(unique_items)]

        # Re-map item indices
        item_embeddings_df['item_local'] = item_embeddings_df['item_idx'].map(item_id_map)

        embeddings = np.vstack(item_embeddings_df['item_normalized_embed'].values)
        item_x = torch.tensor(embeddings, dtype=torch.float)

        artist_ids = torch.tensor(item_embeddings_df['artist_idx'].values, dtype=torch.long)
        album_ids = torch.tensor(item_embeddings_df['album_idx'].values, dtype=torch.long)

        # === 7️⃣ Create HeteroData graph ===
        data = HeteroData()
        data['user', 'interacts', 'item'].edge_index = edge_index
        data['user', 'interacts', 'item'].edge_attr = edge_attr
        data['user', 'interacts', 'item'].edge_weight_init = edge_weight_init

        data['item'].x = item_x
        data['item'].artist_id = artist_ids
        data['item'].album_id = album_ids

        num_users = len(unique_users)
        data['user'].user_id = torch.arange(num_users, dtype=torch.long)

        print(f"[INFO] HeteroData graph created: {num_users} users, {len(unique_items)} items")
        print(f"[INFO] Edge count: {edge_index.shape[1]}")

        return data, user_id_map, item_id_map, inv_user_id_map, inv_item_id_map

    def save_graph(self, output_dir):
        """
        Runs the graph construction pipeline and saves:
            1. The graph (graph.pt)
            2. The ID mappings (as JSON)
        """
        data, user_map, item_map, inv_user_map, inv_item_map = self._build_graph()

        graph_path = Path(output_dir) / "graph.pt"
        torch.save(data, graph_path)

        with open(Path(output_dir) / "user_id_map.json", "w") as f:
            json.dump({int(k): int(v) for k, v in user_map.items()}, f)

        with open(Path(output_dir) / "item_id_map.json", "w") as f:
            json.dump({int(k): int(v) for k, v in item_map.items()}, f)

        with open(Path(output_dir) / "inv_user_id_map.json", "w") as f:
            json.dump({int(k): int(v) for k, v in inv_user_map.items()}, f)

        with open(Path(output_dir) / "inv_item_id_map.json", "w") as f:
            json.dump({int(k): int(v) for k, v in inv_item_map.items()}, f)

        print(f"[SUCCESS] Graph saved to {graph_path}")
        print(f"[SUCCESS] Mapping files saved to {output_dir}")
