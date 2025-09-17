import duckdb

class EventProcessor:
    def __init__(self, con: duckdb.DuckDBPyConnection, embeddings_path, multi_event_path):
        self.con = con
        self.embeddings_path = embeddings_path
        self.multi_event_path = multi_event_path


    def compute_active_users(self, threshold):
        query = f"""
            CREATE TEMPORARY TABLE active_users AS
            SELECT uid
            FROM read_parquet('{self.multi_event_path}')
            GROUP BY uid
            HAVING COUNT(*) >= {threshold}
        """
        self.con.execute(query)
        print("Temporary table 'active_users' created.")


    def filter_multi_event_file(self):
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
        print("Temporary table 'filtered_events' created.")


    def encode_ids(self):
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
        print("Temporary table 'events_with_idx' created.")


    def save_filtered_events(self, output_path, interactions_threshold):
        self.compute_active_users(interactions_threshold)
        self.filter_multi_event_file()
        self.encode_ids()

        # self.con.execute(f"COPY (SELECT * FROM events_with_idx) TO '{output_path}' (FORMAT PARQUET)")
        # print(f'saved filtered multi event file to {output_path}')

    def split_data(self, split_ratios: dict, split_paths: dict):
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
        print(f"Data was split to {split_ratios['train'] * 100}% train set, "
              f"{split_ratios['val'] * 100}% validation set, "
              f"{split_ratios['test'] * 100}% test set")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='train') TO '{split_paths['train']}' (FORMAT PARQUET)")
        print(f"Train data saved to {split_paths['train']}")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='val') TO '{split_paths['val']}' (FORMAT PARQUET)")
        print(f"Validation data saved to {split_paths['val']}")

        self.con.execute(f"COPY (SELECT * FROM split_data WHERE split='test') TO '{split_paths['test']}' (FORMAT PARQUET)")
        print(f"Test data saved to {split_paths['test']}")





            
