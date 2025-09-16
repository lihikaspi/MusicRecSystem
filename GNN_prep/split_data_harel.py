import os
import duckdb
from config import PROCESSED_DIR, INTERACTIONS_FILE
from event_processor import EventProcessor
from edge_aggregator import EdgeAggregator

def split_data():
