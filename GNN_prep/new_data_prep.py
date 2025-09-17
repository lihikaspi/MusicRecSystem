import duckdb
from config import (
    RAW_DATA_FILES,
    RAW_MULTI_EVENT_FILE,
    EMBEDDINGS_FILE,
    INTERACTIONS_FILE,
    INTERACTION_THRESHOLD,
    EVENT_TYPES, SPLIT_RATIOS, SPLIT_PATHS
)
from event_processor import EventProcessor
from edge_aggregator import EdgeAggregator


def main():
    con = duckdb.connect()

    processor = EventProcessor(con, EMBEDDINGS_FILE, RAW_MULTI_EVENT_FILE)
    processor.save_filtered_events(INTERACTIONS_FILE, INTERACTION_THRESHOLD)

    processor.split_data(SPLIT_RATIOS, SPLIT_PATHS)

    # Step 3: Aggregate edges
    aggregator = EdgeAggregator(con)
    aggregator.aggregate_all_edges(EVENT_TYPES)

    # Step 4: Merge all edges into final Parquet
    aggregator.merge_edges(INTERACTIONS_FILE, EVENT_TYPES)


if __name__ == "__main__":
    main()
