import duckdb
import os
from GNN_prep.event_processor import EventProcessor
from GNN_prep.edge_assembler import EdgeAssembler
from GNN_prep.build_graph import GraphBuilder
from config import (
    RAW_MULTI_EVENT_FILE, AUDIO_EMBEDDINGS_FILE, INTERACTIONS_FILE,INTERACTION_THRESHOLD,
    SPLIT_RATIOS, SPLIT_PATHS, TRAIN_SET_FILE, WEIGHTS, TRAIN_EDGES_FILE, TRAIN_GRAPH_FILE,
    ALBUM_MAPPING_FILE, ARTIST_MAPPING_FILE, PROCESSED_DIR, EDGE_TYPE_MAPPING
)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    con = duckdb.connect()

    print('---------- EVENT PROCESSOR ----------')
    processor = EventProcessor(con, AUDIO_EMBEDDINGS_FILE, RAW_MULTI_EVENT_FILE)
    processor.filter_events(INTERACTION_THRESHOLD)
    # processor.filter_events(INTERACTION_THRESHOLD, INTERACTIONS_FILE)
    processor.split_data(SPLIT_RATIOS, SPLIT_PATHS)
    print()

    print('---------- EDGE ASSEMBLER ----------')
    aggregator = EdgeAssembler(con, TRAIN_SET_FILE, WEIGHTS, AUDIO_EMBEDDINGS_FILE,
                               ALBUM_MAPPING_FILE, ARTIST_MAPPING_FILE, EDGE_TYPE_MAPPING)
    aggregator.assemble_edges()
    # aggregator.assemble_edges(EDGES_FILE)
    print()

    print('---------- GRAPH BUILDER ----------')
    graph_builder = GraphBuilder(con)
    graph_builder.save_graph(TRAIN_GRAPH_FILE)

    con.close()


if __name__ == "__main__":
    main()
