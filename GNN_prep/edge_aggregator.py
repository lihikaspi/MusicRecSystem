import duckdb
from config import WEIGHTS, EDGE_TYPE_MAPPING

class EdgeAggregator:
    def __init__(self, con: duckdb.DuckDBPyConnection, events_path, weights, embeddings_path):
        self.con = con
        self.events_path = events_path
        self.weights = weights
        self.embeddings_path = embeddings_path


    def aggregate_edges(self, output_path):
        case_expr = "CASE e.event_type\n"
        for etype, weight in self.weights.items():
            case_expr += f"    WHEN '{etype}' THEN {weight}\n"
        case_expr += "END AS event_weight"

        query = f"""
            CREATE TEMPORARY TABLE agg_edges AS
            SELECT 
                e.user_inx, 
                e.item_idx, 
                e.event_type, 
                COUNT(*) AS edge_count,
                AVG(CASE WHEN e.event_type = 'listen' THEN e.played_ratio_pct END) AS edge_avg_played_ratio,
                {case_expr},
                emb.embed AS edge_embed, emb.normalized_embed AS edge_normalized_embed
            FROM read_parquet('{self.events_path}') e
            LEFT JOIN read_parquet('{self.embeddings_path}') emb
                ON e.item_id = emb.item_id
            GROUP BY e.user_inx, e.item_idx, e.event_type, emb.item_id, emb.embed, emb.normalized_embed
        """
        self.con.execute(query)
        self.con.execute(f"COPY (SELECT * FROM agg_edges) TO '{output_path}' (FORMAT PARQUET)")
        print(f"Edge data saved to {output_path}")



