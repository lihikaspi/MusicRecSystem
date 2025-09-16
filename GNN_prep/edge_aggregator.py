import duckdb
from ..config import WEIGHTS, EDGE_TYPE_MAPPING

class EdgeAggregator:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def create_edge_table(self, event_table, event_type, opposite_table=None):
        cancellation_sql = ""
        if opposite_table:
            cancellation_sql = f"""
                LEFT JOIN (
                    SELECT uid, item_id, COUNT(*) AS cnt
                    FROM {opposite_table}
                    GROUP BY uid, item_id
                ) opp
                ON e.uid = opp.uid AND e.item_id = opp.item_id
            """

        sql = f"""
            CREATE TEMPORARY TABLE {event_type}_edges AS
            SELECT e.uid, e.item_id AS song_id,
                   '{event_type}' AS event_type,
                   GREATEST(COUNT(*) - COALESCE(opp.cnt,0),0) AS edge_count,
                   {WEIGHTS['{event_table}.parquet']} AS edge_weight,
                   {EDGE_TYPE_MAPPING['{event_type}']} AS event_type_id
            FROM {event_table} e
            {cancellation_sql}
            GROUP BY e.uid, e.item_id
        """
        self.con.execute(sql)
        print(f"Aggregated edges for event type '{event_type}'.")

    def aggregate_all_edges(self, event_config):
        for e in event_config:
            self.create_edge_table(e["table"], e["table"], opposite_table=e["opposite"])

    def merge_edges(self, output_file, event_config):
        edge_tables = [f"{e['table']}_edges" for e in event_config]
        union_query = " UNION ALL ".join([f"SELECT * FROM {t}" for t in edge_tables])
        self.con.execute(f"COPY ({union_query}) TO '{output_file}' (FORMAT PARQUET)")
        print(f"All per-event-type edges saved to {output_file}.")
