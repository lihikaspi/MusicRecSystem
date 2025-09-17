import duckdb
import torch
from torch_geometric.data import HeteroData

class GraphBuilder:
    def __init__(self, con: duckdb.DuckDBPyConnection, edges_file):
        self.con = con
        self.edges_file = edges_file


    def build_graph(self):
        # Query all edges from the existing temporary table
        query = """
                SELECT user_idx, item_idx, edge_count, edge_avg_played_ratio, edge_weight
                FROM agg_edges
                """

        edges_df = self.con.execute(query).fetch_df()

        # Edge indices
        edge_index = torch.tensor(
            [edges_df['user_idx'].values, edges_df['item_idx'].values],
            dtype=torch.long
        )

        # Fixed-size edge feature vector: [edge_count, edge_avg_played_ratio, event_weight]
        edge_attr = torch.tensor(
            edges_df[['edge_count', 'edge_avg_played_ratio', 'edge_weight']].fillna(0).values,
            dtype=torch.float
        )

        # HeteroData graph
        data = HeteroData()
        data['user', 'interacts', 'item'].edge_index = edge_index
        data['user', 'interacts', 'item'].edge_attr = edge_attr

        # Item node features: only normalized embedding
        item_embeddings_df = self.con.execute("""
                                              SELECT item_idx, edge_normalized_embed, artist_id, album_id
                                              FROM agg_edges
                                              GROUP BY item_idx, edge_normalized_embed, artist_id, album_id
                                              """).fetch_df()

        data['item'].x = torch.tensor(item_embeddings_df['edge_normalized_embed'].to_list(), dtype=torch.float)

        # Store artist and album IDs separately for use as indices in GNN
        data['item'].artist_id = torch.tensor(item_embeddings_df['artist_id'].values, dtype=torch.long)
        data['item'].album_id = torch.tensor(item_embeddings_df['album_id'].values, dtype=torch.long)

        # Users: store IDs for learnable embeddings
        num_users = edges_df['user_idx'].max() + 1
        data['user'].user_id = torch.arange(num_users, dtype=torch.long)

        return data

    def save_graph(self, output_path):
        data = self.build_graph()
        torch.save(data, output_path)
        print(f"Graph saved to {output_path}")
