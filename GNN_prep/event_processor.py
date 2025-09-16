import os
import duckdb
from config import (PROCESSED_LISTENS_FILE, WEIGHTS, RAW_LISTENS_FILE, RAW_LIKES_FILE, RAW_DISLIKES_FILE, RAW_UNLIKES_FILE, RAW_UNDISLIKES_FILE,
                    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, PROCESSED_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE)

class EventProcessor:
    def __init__(self, con: duckdb.DuckDBPyConnection, embeddings_path):
        self.con = con
        self.embeddings_path = embeddings_path

    def compute_active_users(self, multi_event_path, threshold):
        query = f"""
            CREATE TEMPORARY TABLE active_users AS
            SELECT uid
            FROM read_parquet('{multi_event_path}')
            GROUP BY uid
            HAVING COUNT(*) >= {threshold}
        """
        self.con.execute(query)
        print("Temporary table 'active_users' created.")

    def filter_single_event_file(self, file_path, save_listens=False):
        file_name = os.path.basename(file_path)
        table_name = file_name.replace(".parquet", "")
        query = f"""
            SELECT e.*
            FROM read_parquet('{file_path}') e
            INNER JOIN read_parquet('{self.embeddings_path}') emb
                ON e.item_id = emb.item_id
            INNER JOIN active_users au
                ON e.uid = au.uid
            WHERE e.uid IS NOT NULL AND e.item_id IS NOT NULL
        """
        if save_listens:
            self.con.execute(f"COPY ({query}) TO '{PROCESSED_LISTENS_FILE}' (FORMAT PARQUET)")
            print(f"Filtered listens saved to {PROCESSED_LISTENS_FILE}")

        self.con.execute(f"CREATE TEMPORARY TABLE {table_name} AS {query}")
        print(f"Filtered {file_name} loaded into memory as '{table_name}'.")

    def filter_all_events(self, raw_files):
        for file_path in raw_files:
            save_listens = os.path.basename(file_path) == "listens.parquet"
            self.filter_single_event_file(file_path, save_listens=save_listens)

    def create_union_tables(self):
        query = """
        CREATE TEMPORARY TABLE interactions AS
        SELECT uid, item_id, timestamp, 'listen' AS event_type FROM listens
        UNION ALL
        SELECT uid, item_id, timestamp, 'like' AS event_type FROM likes
        UNION ALL
        SELECT uid, item_id, timestamp, 'dislike' AS event_type FROM dislikes
        UNION ALL
        SELECT uid, item_id, timestamp, 'unlike' AS event_type FROM unlikes
        UNION ALL
        SELECT uid, item_id, timestamp, 'undislike' AS event_type FROM undislikes
        """
        self.con.execute(query)
        print("Union table 'interactions' created.")

    def split_data(self):
        query ="""
        CREATE TEMPORARY TABLE interactions_split AS
        WITH ordered AS (
            SELECT
                user_id,
                song_id,
                timestamp,
                interaction_type,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS rn,
                COUNT(*) OVER (PARTITION BY user_id) AS total_events
            FROM interactions
        )
        SELECT
            user_id,
            song_id,
            timestamp,
            interaction_type,
            CASE
                WHEN rn <= {TRAIN_RATIO} * total_events THEN 'train'
                ELSE 'test'
            END AS split
        FROM ordered
        ORDER BY user_id, timestamp;
        """

        self.con.execute(query)

        # create train, test dbs
        query = """
        CREATE TEMPORARY TABLE train AS
        SELECT * FROM interactions_split WHERE split='train'
        """
        self.con.execute(query)
        query = """
        CREATE TEMPORARY TABLE test AS
        SELECT * FROM interactions_split WHERE split='test'
        """
        self.con.execute(query)

        print("Data split into train and test sets in 'interactions_split' table.")

        # Save to parquet files
        self.con.execute(f"COPY (SELECT * FROM interactions_split WHERE split='train') TO '{TRAIN_FILE}' (FORMAT PARQUET)")
        print(f"Train data saved to {TRAIN_FILE}")
        self.con.execute(f"COPY (SELECT * FROM interactions_split WHERE split='test') TO '{TEST_FILE}' (FORMAT PARQUET)")
        print(f"Test data saved to {TEST_FILE}")
        # Note: VAL_RATIO is set to 0.0, so no validation set is created.





            
