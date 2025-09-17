import duckdb

class EdgeAssembler:
    def __init__(self, con: duckdb.DuckDBPyConnection, events_path, weights: dict, embeddings_path,
                 album_mapping_path, artist_mapping_path, event_type_mapping):
        self.con = con
        self.events_path = events_path
        self.weights = weights
        self.embeddings_path = embeddings_path
        self.album_mapping_path = album_mapping_path
        self.artist_mapping_path = artist_mapping_path
        self.event_type_mapping = event_type_mapping


    def aggregate_edges(self):
        case_expr = "CASE e.event_type\n"
        for etype, weight in self.weights.items():
            case_expr += f"    WHEN '{etype}' THEN {weight}\n"
        case_expr += "END AS edge_weight"

        query = f"""
            CREATE TEMPORARY TABLE agg_edges AS
            SELECT 
                e.user_idx, 
                e.item_idx, 
                e.event_type, 
                COUNT(*) AS edge_count,
                COALESCE(AVG(CASE WHEN e.event_type = 'listen' THEN e.played_ratio_pct END), 0) AS edge_avg_played_ratio,
                {case_expr},
                 emb.normalized_embed AS item_normalized_embed
            FROM read_parquet('{self.events_path}') e
            LEFT JOIN read_parquet('{self.embeddings_path}') emb
                ON e.item_id = emb.item_id
            GROUP BY e.user_idx, e.item_idx, e.event_type, emb.item_id, emb.embed, emb.normalized_embed
        """
        self.con.execute(query)
        print("Temporary table 'agg_edges' created.")


    def add_song_metadata(self):
        query = f"""
            CREATE TEMPORARY TABLE agg_edges_full AS
            WITH 
                -- Encode artists
                artist_index AS (
                    SELECT artist_id, ROW_NUMBER() OVER (ORDER BY artist_id) AS artist_idx
                    FROM (SELECT DISTINCT artist_id FROM read_parquet('{self.artist_mapping_path}'))
                ),
                -- Encode albums
                album_index AS (
                    SELECT album_id, ROW_NUMBER() OVER (ORDER BY album_id) AS album_idx
                    FROM (SELECT DISTINCT album_id FROM read_parquet('{self.album_mapping_path}'))
                ),
                -- Join artist mapping with encoded artist indices
                song_artist_meta AS (
                    SELECT s.item_id, s.artist_id, a.artist_idx
                    FROM read_parquet('{self.artist_mapping_path}') s
                    LEFT JOIN artist_index a USING (artist_id)
                ),
                -- Join album mapping with encoded album indices
                song_album_meta AS (
                    SELECT s.item_id, s.album_id, b.album_idx
                    FROM read_parquet('{self.album_mapping_path}') s
                    LEFT JOIN album_index b USING (album_id)
                )
            SELECT 
                ae.*,
                COALESCE(am.artist_id, 0) AS artist_id,
                COALESCE(am.artist_idx, 0) AS artist_idx,
                COALESCE(al.album_id, 0) AS album_id,
                COALESCE(al.album_idx, 0) AS album_idx
            FROM agg_edges ae
            LEFT JOIN song_artist_meta am
                ON ae.item_idx = am.item_id
            LEFT JOIN song_album_meta al
                ON ae.item_idx = al.item_id
        """
        self.con.execute(query)
        print("Temporary table 'agg_edges_full' created.")


    def add_event_type_cat(self):
        case_event_type = "CASE e.event_type\n"
        for etype, cat in self.event_type_mapping.items():
            case_event_type += f"    WHEN '{etype}' THEN {cat}\n"
        case_event_type += "    ELSE 0\nEND AS edge_type"

        # Query
        query = f"""
            CREATE TEMPORARY TABLE agg_edges_categorical AS
            SELECT e.*, {case_event_type}
            FROM agg_edges_full e
        """
        self.con.execute(query)
        print("Temporary table 'agg_edges_categorical' created.")


    def assemble_edges(self, output_path):
        self.aggregate_edges()
        self.add_song_metadata()
        self.add_event_type_cat()

        # self.con.execute(f"COPY (SELECT * FROM agg_edges_categorical) TO '{output_path}' (FORMAT PARQUET)")
        # print(f"Edge data saved to {output_path}")




