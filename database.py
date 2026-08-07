import sqlite3

DATABASE_NAME = "RelevanceIQ.db"

def create_table():

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patent_match_relevance_results (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        publication_number TEXT,
        title TEXT,
        abstract TEXT,
        independent_claim TEXT,
        all_claims TEXT, 
        product_description TEXT,
        relevance_framework TEXT, 
        relevance TEXT,
        confidence INTEGER,
        rationale TEXT,
        relevance_framework_only TEXT,
        confidence_framework_only INTEGER,
        rationale_framework_only TEXT

    )
    """)

    conn.commit()
    conn.close()

def save_result(result):

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO patent_match_relevance_results(

            publication_number,
            title,
            abstract,
            independent_claim,
            all_claims,
            product_description,
            relevance_framework,
            relevance,
            confidence,
            rationale,
            relevance_framework_only,
            confidence_framework_only,
            rationale_framework_only

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (

            result["publication_number"],
            result["title"],
            result["abstract"],
            result["independent_claim"],
            result["all_claims"],
            result["product_description"],
            result["relevance_framework"],
            result["relevance"],
            result["confidence"],
            result["rationale"],
            result["relevance_framework_only"],
            result["confidence_framework_only"],
            result["rationale_framework_only"]

        )
    )

    conn.commit()
    conn.close()


def fetch_all_results():

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM patent_match_relevance_results
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return rows

import pandas as pd

create_table()

conn = sqlite3.connect(DATABASE_NAME)
df = pd.read_sql_query(
    "SELECT * FROM patent_match_relevance_results",
    conn
)

print(df)

conn.close()