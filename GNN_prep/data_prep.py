import duckdb
from config import (
    RAW_MULTI_EVENT_FILE, EMBEDDINGS_FILE, INTERACTIONS_FILE,
    INTERACTION_THRESHOLD, SPLIT_RATIOS, SPLIT_PATHS, TRAIN_FILE, WEIGHTS, EDGES_FILE
)
from event_processor import EventProcessor
from edge_aggregator import EdgeAggregator


def main():
    con = duckdb.connect()

    processor = EventProcessor(con, EMBEDDINGS_FILE, RAW_MULTI_EVENT_FILE)
    processor.save_filtered_events(INTERACTIONS_FILE, INTERACTION_THRESHOLD)
    processor.split_data(SPLIT_RATIOS, SPLIT_PATHS)

    aggregator = EdgeAggregator(con, TRAIN_FILE, WEIGHTS, EMBEDDINGS_FILE)
    aggregator.aggregate_edges(EDGES_FILE)


if __name__ == "__main__":
    main()
