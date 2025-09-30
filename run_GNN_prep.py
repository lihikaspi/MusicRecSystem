import duckdb
import os
from GNN_prep.event_processor import EventProcessor
from GNN_prep.edge_assembler import EdgeAssembler
from GNN_prep.build_graph import GraphBuilder
from config import config


def main():
    os.makedirs(config.paths.processed_dir, exist_ok=True)

    con = duckdb.connect()

    print('---------- EVENT PROCESSOR ----------')
    processor = EventProcessor(con, config.paths.audio_embeddings_file, config.paths.raw_multi_event_file)
    processor.filter_events(config.preprocessing.interaction_threshold)
    # processor.filter_events(config.preprocessing.interaction_threshold, config.paths.interactions_file)
    processor.split_data(config.preprocessing.split_ratios, config.preprocessing.split_paths)
    print()

    print('---------- EDGE ASSEMBLER ----------')
    aggregator = EdgeAssembler(con, config.paths.train_set_file, config.preprocessing.weights,
                               config.paths.audio_embeddings_file, config.paths.album_mapping_file,
                               config.paths.artist_mapping_file, config.preprocessing.edge_type_mapping)
    aggregator.assemble_edges()
    # aggregator.assemble_edges(config.paths.train_edges_file)
    print()

    print('---------- GRAPH BUILDER ----------')
    graph_builder = GraphBuilder(con)
    graph_builder.save_graph(config.paths.train_graph_file)

    con.close()


if __name__ == "__main__":
    main()
