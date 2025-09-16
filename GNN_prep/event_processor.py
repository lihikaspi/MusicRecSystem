import os
import duckdb
from config import PROCESSED_LISTENS_FILE, WEIGHTS

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
