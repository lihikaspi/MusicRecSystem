import os
import duckdb
from config import (
    RAW_DATA_FILES,
    RAW_MULTI_EVENT_FILE,
    EMBEDDINGS_FILE,
    PROCESSED_LISTENS_FILE,
    INTERACTIONS_FILE,
    WEIGHTS,
    INTERACTION_THRESHOLD
)

def compute_user_interactions_num(con, multi_event_path):
    """
    Create a temporary table active_users in DuckDB memory
    containing only users with interactions >= INTERACTION_THRESHOLD.
    """
    query = f"""
        CREATE TEMPORARY TABLE active_users AS
        SELECT uid
        FROM read_parquet('{multi_event_path}')
        GROUP BY uid
        HAVING COUNT(*) >= {INTERACTION_THRESHOLD}
    """
    con.execute(query)
    print("Temporary table 'active_users' created in DuckDB memory.")


def process_single_event_files(con, embeddings_path):
    """
    Filter single-event files by embeddings and active users,
    add weight column. Only 'listens.parquet' is saved to disk,
    the rest stay in-memory for merging.
    """
    in_memory_tables = {}

    for file_path in RAW_DATA_FILES:
        file_name = os.path.basename(file_path)
        weight = WEIGHTS[file_name]

        query = f"""
            SELECT e.*, {weight} AS event_type_weight
            FROM read_parquet('{file_path}') e
            INNER JOIN read_parquet('{embeddings_path}') emb
                ON e.item_id = emb.item_id
            INNER JOIN active_users au
                ON e.uid = au.uid
            WHERE e.uid IS NOT NULL AND e.item_id IS NOT NULL
        """

        if file_name == "listens.parquet":
            # Save filtered listens file to disk
            con.execute(f"COPY ({query}) TO '{PROCESSED_LISTENS_FILE}' (FORMAT PARQUET)")
            print(f"Filtered listens saved to {PROCESSED_LISTENS_FILE}")
            # Also register as in-memory table for merging
            con.execute(f"CREATE TEMPORARY TABLE listens AS {query}")
        else:
            # Keep other single-event files in memory
            table_name = file_name.replace(".parquet", "")
            con.execute(f"CREATE TEMPORARY TABLE {table_name} AS {query}")
            in_memory_tables[table_name] = True
            print(f"Filtered {file_name} loaded into memory as table '{table_name}'")

    return in_memory_tables


def merge_single_event_files(con):
    """
    Merge all single-event files (common columns only) into one Parquet file.
    """
    # Common columns: uid, timestamp, item_id, is_organic, event_type_weight
    common_columns = "uid, timestamp, item_id, is_organic, event_type_weight"

    # All tables to merge
    tables_to_merge = ["listens", "likes", "dislikes", "unlikes", "undislikes"]

    union_queries = [f"SELECT {common_columns} FROM {tbl}" for tbl in tables_to_merge]
    full_query = " UNION ALL ".join(union_queries)

    con.execute(f"COPY ({full_query}) TO '{INTERACTIONS_FILE}' (FORMAT PARQUET)")
    print(f"All single-event files merged (common columns) into {INTERACTIONS_FILE}")


def main():
    con = duckdb.connect()

    # Create in-memory table for active users
    compute_user_interactions_num(con, RAW_MULTI_EVENT_FILE)

    # Process single-event files
    process_single_event_files(con, EMBEDDINGS_FILE)

    # Merge all single-event files into one Parquet
    merge_single_event_files(con)


if __name__ == "__main__":
    main()
