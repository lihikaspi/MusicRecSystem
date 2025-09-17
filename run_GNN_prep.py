import duckdb
from GNN_prep.event_processor import EventProcessor
from GNN_prep.edge_assembler import EdgeAssembler
from GNN_prep.build_graph import GraphBuilder
from config import (
    RAW_MULTI_EVENT_FILE, EMBEDDINGS_FILE, INTERACTIONS_FILE,INTERACTION_THRESHOLD,
    SPLIT_RATIOS, SPLIT_PATHS, TRAIN_FILE, WEIGHTS, EDGES_FILE, GRAPH_FILE,
    ALBUM_MAPPING_FILE, ARTIST_MAPPING_FILE
)


def main():
    con = duckdb.connect()

    processor = EventProcessor(con, EMBEDDINGS_FILE, RAW_MULTI_EVENT_FILE)
    processor.save_filtered_events(INTERACTIONS_FILE, INTERACTION_THRESHOLD)
    processor.split_data(SPLIT_RATIOS, SPLIT_PATHS)

    aggregator = EdgeAssembler(con, TRAIN_FILE, WEIGHTS, EMBEDDINGS_FILE,
                               ALBUM_MAPPING_FILE, ARTIST_MAPPING_FILE)
    aggregator.assemble_edges(EDGES_FILE)

    graph_builder = GraphBuilder(con, EDGES_FILE)
    graph_builder.save_graph(GRAPH_FILE)

    con.close()


if __name__ == "__main__":
    main()
