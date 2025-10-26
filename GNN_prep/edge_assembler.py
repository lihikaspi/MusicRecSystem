import duckdb

class EdgeAssembler:
    """
    Class to assemble the data for the graph construction
    """
    def __init__(self, con: duckdb.DuckDBPyConnection, events_path: str, weights: dict, embeddings_path: str,
                 album_mapping_path: str, artist_mapping_path: str, event_type_mapping: dict):
        """
        Args:
            con: DuckDB connection
            events_path: Path to the multi-events file
            weights: Dictionary of edge-type weights
            embeddings_path: Path to the embeddings file
            album_mapping_path: Path to the song-album mapping file
            artist_mapping_path: Path to the song-artist mapping file
            event_type_mapping: map of event names to categories
        """
        self.con = con
        self.events_path = events_path
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
        FROM read_parquet('{self.events_path}') e
        WHERE e.event_type = 'listen'
           OR (e.event_type = 'like'
               AND NOT EXISTS (
                   SELECT 1 FROM read_parquet('{self.events_path}') x
                   WHERE x.user_idx = e.user_idx
                     AND x.item_idx = e.item_idx
                     AND x.event_type = 'unlike'
                     AND x.timestamp > e.timestamp
               )
           )
           OR (e.event_type = 'dislike'
               AND NOT EXISTS (
                   SELECT 1 FROM read_parquet('{self.events_path}') x
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
                e.user_idx, 
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
            FROM read_parquet('{self.events_path}') e
            LEFT JOIN read_parquet('{self.embeddings_path}') emb
                ON e.item_id = emb.item_id
            GROUP BY e.user_idx, e.item_idx, e.event_type, emb.item_id, emb.embed, emb.normalized_embed
        """
        self.con.execute(query)
        print("Finished aggregating the edges")


    def _add_song_metadata(self):
        """
        Adds album and artist info to the aggregated edges table
        Creates a temporary table 'agg_edges_artist_album' in the DuckDB memory.
        """
        query = f"""
            CREATE TEMPORARY TABLE agg_edges_artist_album AS
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
        print("Added album and artist information")


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
            self.con.execute(f"COPY (SELECT * FROM agg_edges_event_type) TO '{output_path}' (FORMAT PARQUET)")
            print(f"Edge data saved to {output_path}")