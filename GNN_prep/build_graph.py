import duckdb
import torch
from torch_geometric.data import HeteroData
import numpy as np

class GraphBuilder:
    """
    Class to construct the graph used for the GNN model
    """
    def __init__(self, con: duckdb.DuckDBPyConnection):
        """
        Args:
            con: DuckDB connection
        """
        self.con = con


    def _build_graph(self):
        """
        Builds the graph using the edges file
        """
        # Query all edges from the existing temporary table
        query = """
                SELECT user_idx, item_idx, edge_count, edge_avg_played_ratio, edge_weight, edge_type
                FROM agg_edges_event_type
                """

        edges_df = self.con.execute(query).fetch_df()

        # Edge indices
        edge_index_np = np.vstack((edges_df['user_idx'].values, edges_df['item_idx'].values))
        edge_index = torch.from_numpy(edge_index_np).long()

        # Fixed-size edge feature vector: [edge_count, edge_avg_played_ratio, event_weight]
        edge_attr = torch.tensor(
            edges_df[['edge_type', 'edge_count', 'edge_avg_played_ratio', 'edge_weight']].fillna(0).values,
            dtype=torch.float
        )

        # HeteroData graph
        data = HeteroData()
        data['user', 'interacts', 'item'].edge_index = edge_index
        data['user', 'interacts', 'item'].edge_attr = edge_attr

        # Item node features: only normalized embedding
        item_embeddings_df = self.con.execute("""
                                              SELECT item_idx, item_normalized_embed, artist_idx, album_idx
                                              FROM agg_edges_event_type
                                              GROUP BY item_idx, item_normalized_embed, artist_idx, album_idx
                                              """).fetch_df()

        embeddings = np.vstack(item_embeddings_df['item_normalized_embed'].values)
        data['item'].x = torch.tensor(embeddings, dtype=torch.float)

        # Store artist and album IDs separately for use as indices in GNN
        data['item'].artist_id = torch.tensor(item_embeddings_df['artist_idx'].values, dtype=torch.long)
        data['item'].album_id = torch.tensor(item_embeddings_df['album_idx'].values, dtype=torch.long)

        # Users: store IDs for learnable embeddings
        num_users = edges_df['user_idx'].max() + 1
        data['user'].user_id = torch.arange(num_users, dtype=torch.long)

        return data

    def save_graph(self, output_path):
        """
        runs the graph construction pipeline:
            1. build the graph
            2. save the graph

        Args:
            output_path: path to save the graph
        """
        data = self._build_graph()
        torch.save(data, output_path)
        print(f"Graph saved to {output_path}")
