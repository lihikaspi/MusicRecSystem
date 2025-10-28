import duckdb

class EdgeAssembler:
    """
    Class to assemble the data for the graph construction
    """
    def __init__(self, con: duckdb.DuckDBPyConnection, train_path: str, weights: dict, embeddings_path: str,
                 album_mapping_path: str, artist_mapping_path: str, event_type_mapping: dict):
        """
        Args:
            con: DuckDB connection
            train_path: Path to the multi-events file
            weights: Dictionary of edge-type weights
            embeddings_path: Path to the embeddings file
            album_mapping_path: Path to the song-album mapping file
            artist_mapping_path: Path to the song-artist mapping file
            event_type_mapping: map of event names to categories
        """
        self.con = con
        self.train_path = train_path
        self.weights = weights
        self.embeddings_path = embeddings_path
        self.album_mapping_path = album_mapping_path
        self.artist_mapping_path = artist_mapping_path
        self.event_type_mapping = event_type_mapping

    def _filter_cancelled_events(self):
        """
        Filters train events to cancel out likes/unlikes and dislikes/undislikes.
        - Keeps all listen events.
        - Removes likes followed by unlikes and dislikes followed by undislikes.
        Creates a temporary table 'filtered_events' for aggregation.
        """
        cancel_query = f"""
        CREATE TEMPORARY TABLE filtered_events AS
        SELECT *
        FROM read_parquet('{self.train_path}') e
        WHERE e.event_type = 'listen'
           OR (e.event_type = 'like'
               AND NOT EXISTS (
                   SELECT 1 FROM read_parquet('{self.train_path}') x
                   WHERE x.user_idx = e.user_idx
                     AND x.item_idx = e.item_idx
                     AND x.event_type = 'unlike'
                     AND x.timestamp > e.timestamp
               )
           )
           OR (e.event_type = 'dislike'
               AND NOT EXISTS (
                   SELECT 1 FROM read_parquet('{self.train_path}') x
                   WHERE x.user_idx = e.user_idx
                     AND x.item_idx = e.item_idx
                     AND x.event_type = 'undislike'
                     AND x.timestamp > e.timestamp
               )
           )
        """
        self.con.execute(cancel_query)
        print("Finished filtering cancelled events")


    def _aggregate_edges(self):
        """
        Aggregates the interactions by user-song-event, adding interactions counter, event-type weights and the audio embeddings.
        Creates a temporary table 'agg_edges' in the DuckDB memory.
        """
        case_weight = "CASE e.event_type\n"
        for etype, weight in self.weights.items():
            case_weight += f"    WHEN '{etype}' THEN {weight}\n"
        case_weight += "END AS edge_weight"

        case_event_type = "CASE e.event_type\n"
        for etype, cat in self.event_type_mapping.items():
            case_event_type += f"    WHEN '{etype}' THEN {cat}\n"
        case_event_type += "    ELSE 0\nEND AS edge_type"

        query = f"""
            CREATE TEMPORARY TABLE agg_edges AS
            SELECT 
                e.uid,
                e.user_idx, 
                e.item_id,
                e.item_idx, 
                e.event_type, 
                COUNT(*) AS edge_count,
                {case_weight},
                {case_event_type},
                AVG(
                    CASE 
                        WHEN e.event_type = 'listen' THEN e.played_ratio_pct
                        WHEN e.event_type = 'like' THEN 1
                        WHEN e.event_type = 'dislike' THEN 0
                        ELSE 0.5
                    END
                ) AS edge_avg_played_ratio,
                emb.normalized_embed AS item_normalized_embed
            FROM read_parquet('{self.train_path}') e
            LEFT JOIN read_parquet('{self.embeddings_path}') emb
                ON e.item_id = emb.item_id
            GROUP BY e.uid, e.user_idx, e.item_id, e.item_idx, e.event_type, emb.item_id, emb.embed, emb.normalized_embed
        """
        self.con.execute(query)
        print("Finished aggregating the edges")


    def _prepare_train_item_map(self):
        """
        Create a mapping from item_idx to continuous train indices for items in the training set.
        """
        self.con.execute(f"""
            CREATE TEMPORARY TABLE train_items AS
            SELECT DISTINCT item_idx
            FROM read_parquet('{self.train_path}')
            WHERE split = 'train'
        """)

        self.con.execute("""
                         CREATE
                         TEMPORARY TABLE train_item_map AS
                         SELECT item_idx, ROW_NUMBER() OVER (ORDER BY item_idx) - 1 AS item_train_idx
                         FROM train_items
                         """)
        print("[INFO] Prepared train item re-index mapping")


    def _prepare_artist_album_metadata(self):
        """
        Prepare artist and album metadata tables with encoded indices.
        """
        self.con.execute(f"""
            CREATE TEMPORARY TABLE artist_index AS
            SELECT artist_id, ROW_NUMBER() OVER (ORDER BY artist_id) AS artist_idx
            FROM (SELECT DISTINCT artist_id FROM read_parquet('{self.artist_mapping_path}'))
        """)

        self.con.execute(f"""
            CREATE TEMPORARY TABLE album_index AS
            SELECT album_id, ROW_NUMBER() OVER (ORDER BY album_id) AS album_idx
            FROM (SELECT DISTINCT album_id FROM read_parquet('{self.album_mapping_path}'))
        """)

        self.con.execute(f"""
            CREATE TEMPORARY TABLE song_artist_meta AS
            SELECT s.item_id, s.artist_id, a.artist_idx
            FROM read_parquet('{self.artist_mapping_path}') s
            LEFT JOIN artist_index a USING (artist_id)
        """)

        self.con.execute(f"""
            CREATE TEMPORARY TABLE song_album_meta AS
            SELECT s.item_id, s.album_id, b.album_idx
            FROM read_parquet('{self.album_mapping_path}') s
            LEFT JOIN album_index b USING (album_id)
        """)
        print("[INFO] Prepared artist and album metadata")


    def _merge_edges_metadata(self):
        """
        Merge aggregated edges with metadata, assign train indices, and deduplicate by item_idx.
        """
        self.con.execute("""
                         CREATE
                         TEMPORARY TABLE merged AS
                         SELECT ae.item_idx,
                                COALESCE(tm.item_train_idx, -1) AS item_train_idx,
                                ae.item_normalized_embed,
                                COALESCE(am.artist_id, 0)       AS artist_id,
                                COALESCE(am.artist_idx, 0)      AS artist_idx,
                                COALESCE(al.album_id, 0)        AS album_id,
                                COALESCE(al.album_idx, 0)       AS album_idx
                         FROM agg_edges ae
                                  LEFT JOIN song_artist_meta am ON ae.item_id = am.item_id
                                  LEFT JOIN song_album_meta al ON ae.item_id = al.item_id
                                  LEFT JOIN train_item_map tm ON ae.item_idx = tm.item_idx
                         """)

        self.con.execute("""
                         CREATE
                         TEMPORARY TABLE agg_edges_artist_album AS
                         SELECT item_idx,
                                item_train_idx,
                                ANY_VALUE(item_normalized_embed) AS item_normalized_embed,
                                MAX(artist_idx)                  AS artist_idx,
                                MAX(album_idx)                   AS album_idx,
                                MAX(artist_id)                   AS artist_id,
                                MAX(album_id)                    AS album_id
                         FROM merged
                         GROUP BY item_idx, item_train_idx
                         """)
        print("[INFO] Merged edges and metadata, deduplicated, and re-indexed items for train")


    def _add_song_metadata(self):
        self._prepare_train_item_map()
        self._prepare_artist_album_metadata()
        self._merge_edges_metadata()


    def assemble_edges(self, output_path: str = None):
        """
        Runs the edge assembler pipeline:
            1. aggregate the edges
            2. add artist and album info

        Add an output path to save the filtered file.

        Args:
            output_path: path to output file, default: None
        """
        # self._filter_cancelled_events()
        self._aggregate_edges()
        self._add_song_metadata()

        if output_path is not None:
            self.con.execute(f"COPY (SELECT * FROM agg_edges_artist_album) TO '{output_path}' (FORMAT PARQUET)")
            print(f"Edge data saved to {output_path}")