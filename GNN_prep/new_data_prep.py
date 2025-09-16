import duckdb
from ..config import (
    RAW_DATA_FILES,
    RAW_MULTI_EVENT_FILE,
    EMBEDDINGS_FILE,
    INTERACTIONS_FILE,
    INTERACTION_THRESHOLD,
    EVENT_TYPES
)
from event_processor import EventProcessor
from edge_aggregator import EdgeAggregator


def main():
    con = duckdb.connect()

    # Step 1: Compute active users
    processor = EventProcessor(con, EMBEDDINGS_FILE)
    processor.compute_active_users(RAW_MULTI_EVENT_FILE, INTERACTION_THRESHOLD)

    # Step 2: Filter all single-event files
    processor.filter_all_events(RAW_DATA_FILES)

    # Step 2.5: Create union table
    processor.create_union_tables()

    processor.split_data()

    # Step 3: Aggregate edges
    aggregator = EdgeAggregator(con)
    aggregator.aggregate_all_edges(EVENT_TYPES)

    # Step 4: Merge all edges into final Parquet
    aggregator.merge_edges(INTERACTIONS_FILE, EVENT_TYPES)


if __name__ == "__main__":
    main()
