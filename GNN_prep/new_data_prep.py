import os
import duckdb
from config import (
    RAW_DATA_FILES,
    RAW_MULTI_EVENT_FILE,
    EMBEDDINGS_FILE,
    PROCESSED_LISTENS_FILE,
    INTERACTIONS_FILE,
    WEIGHTS,
    INTERACTION_THRESHOLD,
    EDGE_TYPE_MAPPING
)


def compute_active_users(con, multi_event_path):
    """
    Create a temporary table 'active_users' for users with interactions >= INTERACTION_THRESHOLD
    """
    query = f"""
        CREATE TEMPORARY TABLE active_users AS
        SELECT uid
        FROM read_parquet('{multi_event_path}')
        GROUP BY uid
        HAVING COUNT(*) >= {INTERACTION_THRESHOLD}
    """
    con.execute(query)
    print("Temporary table 'active_users' created.")


def process_single_event_files(con, embeddings_path):
    """
    Filter single-event files by embeddings and active users.
    Save listens.parquet to disk, load all tables in memory.
    """
    in_memory_tables = {}

    for file_path in RAW_DATA_FILES:
        file_name = os.path.basename(file_path)
        table_name = file_name.replace(".parquet", "")
        weight = WEIGHTS[file_name]

        query = f"""
            SELECT e.*
            FROM read_parquet('{file_path}') e
            INNER JOIN read_parquet('{embeddings_path}') emb
                ON e.item_id = emb.item_id
            INNER JOIN active_users au
                ON e.uid = au.uid
            WHERE e.uid IS NOT NULL AND e.item_id IS NOT NULL
        """

        # Save filtered listens to disk
        if file_name == "listens.parquet":
            con.execute(f"COPY ({query}) TO '{PROCESSED_LISTENS_FILE}' (FORMAT PARQUET)")
            print(f"Filtered listens saved to {PROCESSED_LISTENS_FILE}")

        # Create temporary table for all event types
        con.execute(f"CREATE TEMPORARY TABLE {table_name} AS {query}")
        in_memory_tables[table_name] = True
        print(f"Filtered {file_name} loaded into memory as '{table_name}'.")

    return in_memory_tables


def aggregate_edges(con):
    """
    Aggregate interactions per user-song-event_type.
    Handle cancellations and add event_type_id from EDGE_TYPE_MAPPING.
    """
    # Likes minus unlikes
    con.execute(f"""
        CREATE TEMPORARY TABLE like_edges AS
        SELECT l.uid, l.item_id AS song_id,
               'like' AS event_type,
               GREATEST(COUNT(*) - COALESCE(u.cnt,0),0) AS edge_count,
               {WEIGHTS['likes.parquet']} AS edge_weight,
               {EDGE_TYPE_MAPPING['like']} AS event_type_id
        FROM likes l
        LEFT JOIN (
            SELECT uid, item_id, COUNT(*) AS cnt
            FROM unlikes
            GROUP BY uid, item_id
        ) u
        ON l.uid = u.uid AND l.item_id = u.item_id
        GROUP BY l.uid, l.item_id
    """)

    # Dislikes minus undislikes
    con.execute(f"""
        CREATE TEMPORARY TABLE dislike_edges AS
        SELECT d.uid, d.item_id AS song_id,
               'dislike' AS event_type,
               GREATEST(COUNT(*) - COALESCE(u.cnt,0),0) AS edge_count,
               {WEIGHTS['dislikes.parquet']} AS edge_weight,
               {EDGE_TYPE_MAPPING['dislike']} AS event_type_id
        FROM dislikes d
        LEFT JOIN (
            SELECT uid, item_id, COUNT(*) AS cnt
            FROM undislikes
            GROUP BY uid, item_id
        ) u
        ON d.uid = u.uid AND d.item_id = u.item_id
        GROUP BY d.uid, d.item_id
    """)

    # Listens (no cancellation)
    con.execute(f"""
        CREATE TEMPORARY TABLE listen_edges AS
        SELECT uid, item_id AS song_id,
               'listen' AS event_type,
               COUNT(*) AS edge_count,
               {WEIGHTS['listens.parquet']} AS edge_weight,
               {EDGE_TYPE_MAPPING['listen']} AS event_type_id
        FROM listens
        GROUP BY uid, item_id
    """)

    # Unlikes alone (remaining after cancellation)
    con.execute(f"""
        CREATE TEMPORARY TABLE unlike_edges AS
        SELECT uid, item_id AS song_id,
               'unlike' AS event_type,
               COUNT(*) AS edge_count,
               {WEIGHTS['unlikes.parquet']} AS edge_weight,
               {EDGE_TYPE_MAPPING['unlike']} AS event_type_id
        FROM unlikes
        GROUP BY uid, item_id
    """)

    # Undislikes alone
    con.execute(f"""
        CREATE TEMPORARY TABLE undislike_edges AS
        SELECT uid, item_id AS song_id,
               'undislike' AS event_type,
               COUNT(*) AS edge_count,
               {WEIGHTS['undislikes.parquet']} AS edge_weight,
               {EDGE_TYPE_MAPPING['undislike']} AS event_type_id
        FROM undislikes
        GROUP BY uid, item_id
    """)


def merge_all_edges(con):
    """
    Merge all per-event-type edges into a single Parquet file.
    """
    tables = ["listen_edges", "like_edges", "unlike_edges", "dislike_edges", "undislike_edges"]
    union_query = " UNION ALL ".join([f"SELECT * FROM {t}" for t in tables])

    con.execute(f"""
        COPY ({union_query}) TO '{INTERACTIONS_FILE}' (FORMAT PARQUET)
    """)
    print(f"All per-event-type edges saved to {INTERACTIONS_FILE}.")


def main():
    con = duckdb.connect()

    # Step 1: Create active users table
    compute_active_users(con, RAW_MULTI_EVENT_FILE)

    # Step 2: Filter single-event files and save filtered listens
    process_single_event_files(con, EMBEDDINGS_FILE)

    # Step 3: Aggregate edges per user-song-event_type and add event_type_id
    aggregate_edges(con)

    # Step 4: Merge all edges into final Parquet
    merge_all_edges(con)


if __name__ == "__main__":
    main()
