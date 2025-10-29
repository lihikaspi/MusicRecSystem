import duckdb
import os
from GNN_prep.event_processor import EventProcessor
from GNN_prep.edge_assembler import EdgeAssembler
from GNN_prep.build_graph import GraphBuilder
from config import config


def check_prev_files():
    """
    check for the files created in the previous stage.
    if at least one file is missing raises FileNotFoundError
    """
    needed = [config.paths.raw_multi_event_file, config.paths.audio_embeddings_file,
              config.paths.artist_mapping_file, config.paths.album_mapping_file]
    fail = False
    for file in needed:
        if not os.path.exists(file):
            print("Couldn't find file: {}".format(file))
            fail = True
    if fail:
        raise FileNotFoundError("Needed files are missing, run previous stage to create the needed files!")
    else:
        print("All needed files are present! starting Preprocessing ... ")


def main():
    con = duckdb.connect()

    print('---------- EVENT PROCESSOR ----------')
    processor = EventProcessor(con, config)
    processor.filter_events()
    # processor.filter_events(output_path=config.paths.interactions_file)
    processor.split_data()

    print('\n---------- EDGE ASSEMBLER ----------')
    aggregator = EdgeAssembler(con, config)
    aggregator.assemble_edges()
    # aggregator.assemble_edges(config.paths.train_edges_file)

    print('\n---------- GRAPH BUILDER ----------')
    graph_builder = GraphBuilder(con)
    graph_builder.build_graph(config.paths.train_graph_file)

    con.close()


if __name__ == "__main__":
    check_prev_files()
    main()
