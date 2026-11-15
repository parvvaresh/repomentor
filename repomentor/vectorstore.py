"""
Where chunk vectors are searched.

`NumpyStore` scores every chunk with one matrix product; it is exact and
plenty fast for a few repositories. `PgVectorStore` keeps vectors in
PostgreSQL with the pgvector extension and an HNSW index, for deployments
with many repositories or large ones. Both return a full score array so
the retrieval code does not care which one is used; pgvector fills only
its nearest candidates and leaves the rest at -inf.
"""

import numpy as np


# How many nearest neighbours pgvector returns per query
PGVECTOR_CANDIDATES = 200


class NumpyStore:
    name = "numpy"

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def scores(self, vector: np.ndarray, count: int) -> np.ndarray:
        return self.embeddings @ vector


class PgVectorStore:
    name = "pgvector"

    def __init__(self, database_url: str, index_key: str, dimension: int):
        self.database_url = database_url
        self.index_key = index_key
        self.dimension = dimension
        self.table = f"repomentor_chunks_{dimension}"

    def _connect(self):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as error:
            raise RuntimeError(
                "VECTOR_STORE=pgvector needs the psycopg and pgvector "
                "packages: pip install 'psycopg[binary]' pgvector"
            ) from error

        connection = psycopg.connect(self.database_url, autocommit=True)
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        return connection

    def _ensure_table(self, connection) -> None:
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            "index_key text NOT NULL, "
            "position integer NOT NULL, "
            "chunk_id text, "
            f"embedding vector({self.dimension}) NOT NULL, "
            "PRIMARY KEY (index_key, position))"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
            "USING hnsw (embedding vector_cosine_ops)"
        )

    def replace(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        """Store the vectors of one index, replacing any earlier version."""

        with self._connect() as connection:
            self._ensure_table(connection)

            with connection.transaction():
                connection.execute(
                    f"DELETE FROM {self.table} WHERE index_key = %s",
                    (self.index_key,),
                )

                with connection.cursor() as cursor:
                    cursor.executemany(
                        f"INSERT INTO {self.table} "
                        "(index_key, position, chunk_id, embedding) "
                        "VALUES (%s, %s, %s, %s)",
                        [
                            (self.index_key, position, chunk_id, vector)
                            for position, (chunk_id, vector)
                            in enumerate(zip(chunk_ids, embeddings))
                        ],
                    )

    def delete(self) -> None:
        with self._connect() as connection:
            self._ensure_table(connection)
            connection.execute(
                f"DELETE FROM {self.table} WHERE index_key = %s",
                (self.index_key,),
            )

    def scores(self, vector: np.ndarray, count: int) -> np.ndarray:
        result = np.full(count, -np.inf)

        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT position, 1 - (embedding <=> %s) FROM {self.table} "
                "WHERE index_key = %s ORDER BY embedding <=> %s LIMIT %s",
                (vector, self.index_key, vector, PGVECTOR_CANDIDATES),
            ).fetchall()

        for position, score in rows:
            if position < count:
                result[position] = score

        return result
