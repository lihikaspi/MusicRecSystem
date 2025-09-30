import duckdb

class EventProcessor:
    """
    Class for the pre-process of the multi-event file
    """
    def __init__(self, con: duckdb.DuckDBPyConnection, embeddings_path: str, multi_event_path: str):
        """
        Args:
            con: duckdb connection
            embeddings_path: path to embeddings file
            multi_event_path: path to multi-event file
        """
        self.con = con
        self.embeddings_path = embeddings_path
        self.multi_event_path = multi_event_path


    def _compute_active_users(self, threshold: int):
        """
        finds all the users that have more interactions than the given threshold.
        creates a temporary table 'active_users' in the DuckDB memory.

        Args:
            threshold: threshold for the amount of interactions
        """
        query = f"""
            CREATE TEMPORARY TABLE active_users AS
            SELECT uid
            FROM read_parquet('{self.multi_event_path}')
            GROUP BY uid
            HAVING COUNT(*) >= {threshold}
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


    def _encode_ids(self):
        """
        encodes the given user IDs and songs IDs into GNN-ready IDs (continuous integers).
        creates a temporary table 'events_with_idx' in the DuckDB memory.
        """
        query = """
            CREATE TEMPORARY TABLE events_with_idx AS
            WITH 
            user_index AS (
                SELECT uid, ROW_NUMBER() OVER (ORDER BY uid) - 1 AS user_idx
                FROM (SELECT DISTINCT uid FROM filtered_events)
            ),
            song_index AS (
                SELECT item_id, ROW_NUMBER() OVER (ORDER BY item_id) - 1 AS item_idx
                FROM (SELECT DISTINCT item_id FROM filtered_events)
            )
           SELECT e.*, u.user_idx, s.item_idx
           FROM filtered_events e
                JOIN user_index u USING (uid)
                JOIN song_index s USING (item_id)
        """
        self.con.execute(query)
        print("Created user and song indices")


    def filter_events(self, interactions_threshold: int, output_path:str = None ):
        """
        runs the multi-event filtering pipeline:
            1. find active users with more interactions than the given threshold
            2. filter out songs and users
            3. encode user and song IDs

        Add an output path to save the filtered file.

        Args:
            interactions_threshold: threshold for interactions
            output_path: path to save the filtered file, default: None
        """
        self._compute_active_users(interactions_threshold)
        self._filter_multi_event_file()
        self._encode_ids()

        if output_path is not None:
            self.con.execute(f"COPY (SELECT * FROM events_with_idx) TO '{output_path}' (FORMAT PARQUET)")
            print(f'Filtered multi event file saved to {output_path}')

    def split_data(self, split_ratios: dict, split_paths: dict):
        """
        splits the filtered multi-event file into train, validation and test sets.

        Args:
            split_ratios: dictionary with keys 'train', 'valid', 'test' containing the ratio for each set
            split_paths: dictionary with keys 'train', 'valid', 'test' containing the save paths
        """
        query = f"""
            CREATE TEMPORARY TABLE split_data AS
            WITH ordered AS (
                SELECT e.*,
                       ROW_NUMBER() OVER (PARTITION BY e.user_idx ORDER BY e.timestamp) AS rn,
                       COUNT(*) OVER (PARTITION BY e.user_idx) AS total_events
                FROM events_with_idx e
            )
            SELECT o.*,
                   CASE 
                       WHEN o.rn <= {split_ratios['train']} * o.total_events THEN 'train'
                       WHEN o.rn <= ({split_ratios['train']} + {split_ratios['val']}) * o.total_events THEN 'val'
                       ELSE 'test'
                   END AS split
            FROM ordered o
            ORDER BY o.user_idx, o.timestamp
        """
        self.con.execute(query)
        print(f"\nData was split into:\n"
              f"{split_ratios['train'] * 100}% train set\n"
              f"{split_ratios['val'] * 100}% validation set\n"
              f"{split_ratios['test'] * 100}% test set\n")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='train') TO '{split_paths['train']}' (FORMAT PARQUET)")
        print(f"Train data saved to {split_paths['train']}")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='val') TO '{split_paths['val']}' (FORMAT PARQUET)")
        print(f"Validation data saved to {split_paths['val']}")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='test') TO '{split_paths['test']}' (FORMAT PARQUET)")
        print(f"Test data saved to {split_paths['test']}")
