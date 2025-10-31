import duckdb
from config import Config

class EventProcessor:
    """
    Class for the pre-process of the multi-event file
    """
    def __init__(self, con: duckdb.DuckDBPyConnection, config: Config):
        """
        Args:
            con: duckdb connection
            config: global Config object
        """
        self.con = con
        self.embeddings_path = config.paths.audio_embeddings_file
        self.multi_event_path = config.paths.raw_multi_event_file
        self.low_threshold = config.preprocessing.low_interaction_threshold
        self.high_threshold = config.preprocessing.high_interaction_threshold
        self.split_ratios = config.preprocessing.split_ratios
        self.split_paths = config.paths.split_paths
        self.cold_start_songs_path = config.paths.cold_start_songs_file


    def _compute_active_users(self):
        """
        finds all the users that have more interactions than the given threshold.
        creates a temporary table 'active_users' in the DuckDB memory.
        """
        query = f"""
            CREATE TEMPORARY TABLE active_users AS
            SELECT uid
            FROM read_parquet('{self.multi_event_path}')
            GROUP BY uid
            HAVING COUNT(*) >= {self.low_threshold}
            AND COUNT(*) <= {self.high_threshold}
        """
        self.con.execute(query)
        print("Found all active users")


    def _filter_multi_event_file(self):
        """
        filters the multi-event file out of songs without provided embeddings and non-active users.
        creates a temporary table 'filtered_event' in the DuckDB memory.
        """
        query = f"""
            CREATE TEMPORARY TABLE filtered_events AS
            SELECT e.*
            FROM read_parquet('{self.multi_event_path}') e
            INNER JOIN read_parquet('{self.embeddings_path}') emb
                ON e.item_id = emb.item_id
            INNER JOIN active_users au
                ON e.uid = au.uid
            WHERE e.uid IS NOT NULL AND e.item_id IS NOT NULL
        """
        self.con.execute(query)
        print("Finished filtering the multi-event interactions")


    def _encode_user_ids(self):
        """
        encodes the given user IDs into GNN-ready IDs (continuous integers).
        creates a temporary table 'events_with_idx' in the DuckDB memory.
        """
        query = """
            CREATE TEMPORARY TABLE events_with_idx AS
            WITH 
            user_index AS (
                SELECT uid, ROW_NUMBER() OVER (ORDER BY uid) - 1 AS user_id
                FROM (SELECT DISTINCT uid FROM filtered_events)
            )
            SELECT e.*, u.user_id
            FROM filtered_events e
            JOIN user_index u USING (uid)
            """
        self.con.execute(query)
        print("Created user indices")


    def filter_events(self, low_threshold: int = None, high_threshold: int = None, output_path:str = None ):
        """
        runs the multi-event filtering pipeline:
            1. find active users with more interactions than the given threshold
            2. filter out songs and users
            3. encode user IDs

        Add a threshold value to override the config-defined threshold
        and/or an output path to save the filtered file as a parquet.

        Args:
            low_threshold: lower threshold for interactions, default: none
            high_threshold: higher threshold for interactions, default: none
            output_path: path to save the filtered file, default: none
        """
        if low_threshold is not None:
            self.low_threshold = low_threshold

        if high_threshold is not None:
            self.high_threshold = high_threshold

        self._compute_active_users()
        self._filter_multi_event_file()
        self._encode_user_ids()

        if output_path is not None:
            self.con.execute(f"COPY (SELECT * FROM events_with_idx) TO '{output_path}' (FORMAT PARQUET)")
            print(f'Filtered multi event file saved to {output_path}')


    def _split_data(self):
        """
        splits the filtered multi-event file into train, validation and test sets
        """
        query = f"""
            CREATE TEMPORARY TABLE split_data AS
            WITH ordered AS (
                SELECT e.*,
                       ROW_NUMBER() OVER (PARTITION BY e.user_id ORDER BY e.timestamp) AS rn,
                       COUNT(*) OVER (PARTITION BY e.user_id) AS total_events
                FROM events_with_idx e
            )
            SELECT o.*,
                   CASE 
                       WHEN o.rn <= {self.split_ratios['train']} * o.total_events THEN 'train'
                       WHEN o.rn <= ({self.split_ratios['train']} + {self.split_ratios['val']}) * o.total_events THEN 'val'
                       ELSE 'test'
                   END AS split
            FROM ordered o
            ORDER BY o.user_id, o.rn
        """

        self.con.execute(query)
        print(f"\nData was split into:\n"
              f"{self.split_ratios['train'] * 100}% train set\n"
              f"{self.split_ratios['val'] * 100}% validation set\n"
              f"{self.split_ratios['test'] * 100}% test set\n")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='train') TO '{self.split_paths['train']}' (FORMAT PARQUET)")
        print(f"Train data saved to {self.split_paths['train']}")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='val') TO '{self.split_paths['val']}' (FORMAT PARQUET)")
        print(f"Validation data saved to {self.split_paths['val']}")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='test') TO '{self.split_paths['test']}' (FORMAT PARQUET)")
        print(f"Test data saved to {self.split_paths['test']}")


    def _save_cold_start_songs(self):
        """
        save the audio embeddings and song IDs of songs not in the train set for evaluation purposes.
        """
        self.con.execute("""
            CREATE TEMPORARY TABLE test_items AS
            SELECT DISTINCT item_id 
            FROM split_data 
            WHERE split = 'test'
        """)

        self.con.execute(f"""
            CREATE TEMPORARY TABLE cold_start_songs AS
            SELECT d.item_id, emb.normalized_embed
            FROM split_data d
            LEFT JOIN read_parquet('{self.embeddings_path}') emb
                ON d.item_id = emb.item_id
            LEFT JOIN test_items t
                ON d.item_id = t.item_id
            WHERE d.split IN ('train', 'val') 
              AND t.item_id IS NULL
        """)

        self.con.execute(f"""COPY (SELECT * FROM cold_start_songs) TO '{self.cold_start_songs_path}' (FORMAT PARQUET)""")
        print(f'Cold start songs file saved to {self.cold_start_songs_path}')


    def split_data(self, split_ratios: dict = None):
        """
        splits the filtered multi-event file into train, validation and test sets and
        saves the embeddings of the cold-start songs.

        Args:
            split_ratios: dictionary of split ratios, default: none
        """
        if split_ratios is not None:
            self.split_ratios = split_ratios

        self._split_data()
        self._save_cold_start_songs()

