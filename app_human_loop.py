import streamlit as st
import sqlite3
import uuid
from langgraph.types import Command
import pandas as pd

# TEST_MODE = True

from graph_3_node import workflow
from database import create_table, save_result

st.set_page_config(
    page_title="RelevanceIQ",
    page_icon="Icon.png",
    layout="centered"
)

st.title("RelevanceIQ")
st.subheader("Patent Relevance Analysis - Auto and Relevance Framework- Human Loop")

create_table()

# Session State
# -----------------------------
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "df" not in st.session_state:
    st.session_state.df = None

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "results" not in st.session_state:
    st.session_state.results = []

if "product_description" not in st.session_state:
    st.session_state.product_description = ""

if "relevance_framework" not in st.session_state:
    st.session_state.relevance_framework = ""

#-----------------------------
## Inputs 

product_description = st.text_area(
    "Enter Product Description",
    height=200
)

uploaded_file = st.file_uploader(
    "Upload Patent Excel File",
    type=["xlsx"]
)

relevance_framework = st.text_area(
    "Relevance Framework",
    height=250
)

run = st.button("Run Analysis")

if run:

    if uploaded_file is None:
        st.warning("Please upload an Excel file.")
        st.stop()

    if not product_description.strip():
        st.warning("Please enter the product description.")
        st.stop()

    if not relevance_framework.strip():
        st.warning("Please provide the relevance framework.")
        st.stop()


    st.session_state.df = pd.read_excel(uploaded_file)

    st.session_state.product_description = product_description
    st.session_state.relevance_framework = relevance_framework

    st.session_state.results = []
    st.session_state.current_index = 0
    st.session_state.analysis_started = True

    st.session_state.session_id = str(uuid.uuid4())

    st.rerun()

# Process one patent at a time
# ----------------------------------------------------

if st.session_state.analysis_started:

    df = st.session_state.df
    index = st.session_state.current_index

    # Finished all patents
    # -----------------------------
    if index >= len(df):
        
        st.success("Analysis Completed Successfully!")

        result_df = pd.DataFrame(st.session_state.results)

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.download_button(
            "Download Results",
            result_df.to_csv(index=False),
            "RelevanceIQ_Results.csv",
            "text/csv"
        )

    else:

        progress_bar = st.progress(index / len(df))

        status = st.empty()

        status.write(
            f"Processing Patent {index + 1} of {len(df)}"
        )

        row = df.iloc[index]

        config = {
            "configurable": {
                "thread_id":   f"{st.session_state.session_id}_patent_{index}"
            }
        }

        # Get current graph state
        snapshot = workflow.get_state(config)

        # First execution for this patent
        if not snapshot or not snapshot.values:
            state = {

                "product_description": st.session_state.product_description,

                "publication_number": row["Publication Number"],

                "title": row["title"],

                "abstract": row["Abstract"],

                "independent_claim": row["Independent Claim"],

                "all_claims": row["All Claims"],

                "relevance_framework": st.session_state.relevance_framework,

                "product_features": [],

                "patent_features": [],

                "comparison": [],

                "relevance": "",

                "confidence": 0,

                "rationale": "",

                "relevance_framework_only": "",

                "confidence_framework_only": 0,

                "rationale_framework_only": "",

                "review_type": ""

            }

            workflow.invoke(
                state,
                config=config
            )

            snapshot = workflow.get_state(config)
            st.write(snapshot.values)

            print("Next:", snapshot.next)
            print("Interrupts:", snapshot.interrupts)

            if snapshot.interrupts:
                print(snapshot.interrupts[0].value)


        # Waiting for human review
        # if snapshot.next:
        if snapshot.interrupts:    

            review = snapshot.interrupts[0].value

            st.info(
                f"Review the extracted {review['review_type']} features"
            )

            edited = st.text_area(

                "Edit Features",

                value="\n".join(review["features"]),

                height=250,

                key=f"review_{index}"

            )

            if st.button(
                "Approve & Continue",
                key=f"approve_{index}"
            ):

                workflow.invoke(

                    Command(
                        resume=edited.split("\n")
                    ),

                    config=config

                )

                st.rerun()


        elif snapshot.next:       # Added 

            st.info("Continuing workflow...")
            st.rerun()        

        # Workflow finished
        # ---------------------------------------
        else:

            result = snapshot.values

            save_result(result)

            st.session_state.results.append({

                "Publication Number": result["publication_number"],

                "Title": result["title"],

                "Relevance": result["relevance"],

                "Confidence": result["confidence"],

                "Rationale": result["rationale"],

                "Relevance_Framework_Only": result["relevance_framework_only"],
                
                "Confidence_Framework_Only": result["confidence_framework_only"],

                "Rationale_Framework_Only": result["rationale_framework_only"]

            })

            progress_bar.progress(
                (index + 1) / len(df)
            )

            st.session_state.current_index += 1

            st.rerun()
            
# --------------------------------------------------
# View Database
# --------------------------------------------------

st.divider()

if st.button("View Database"):

    conn = sqlite3.connect("RelevanceIQ.db")

    database_df = pd.read_sql_query(

        "SELECT * FROM patent_match_relevance_results ORDER BY id DESC",
        conn

    )

    conn.close()

    st.dataframe(
        database_df,
        use_container_width=True
    )
