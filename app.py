import streamlit as st
import sqlite3
import uuid
from langgraph.types import Command
import pandas as pd
from node import llm, feature_parser

# TEST_MODE = True
from graph import workflow
from node import extract_product_features
from database import create_table, save_result

st.set_page_config(
    page_title="RelevanceIQ",
    page_icon="Icon.png",
    layout="centered"
)

# -----------------------------------
# Login
# -----------------------------------

def login():

    st.title("RelevanceIQ")
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        users = st.secrets["users"] 
        if username in users and password == users[username]:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.rerun()

        else:
            st.error("Invalid username or password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

st.title("RelevanceIQ")
st.subheader("Patent Relevance Analysis")

st.write(f"Logged in as: {st.session_state.username}")

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

if "primary_product_features" not in st.session_state:
    st.session_state.primary_product_features = []   

if "secondary_product_features" not in st.session_state:
    st.session_state.secondary_product_features = []

if "product_features_approved" not in st.session_state:
    st.session_state.product_features_approved = False

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

st.download_button(
    label="Download Input Data Template Sample",
    data=open("RelevanceIQ_Input_Data_Schema.xlsx", "rb").read(),
    file_name="RelevanceIQ_Input_Data_Schema.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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

    st.session_state.primary_product_features = []
    st.session_state.secondary_product_features = []
    st.session_state.product_features_approved = False

    st.session_state.analysis_started = True

    st.session_state.session_id = str(uuid.uuid4())

    st.rerun()

# ----------------------------------------------------
# EXTRACT PRODUCT FEATURES ONCE
# ----------------------------------------------------

if (
    st.session_state.analysis_started
    and not st.session_state.product_features_approved
):

    # First extraction
    if not st.session_state.primary_product_features and not st.session_state.secondary_product_features:

        st.info("Extracting product features for review...")

        product_prompt = f"""
You are an experienced patent analyst performing an FTO
(Freedom-to-Operate) technical analysis.

Extract the technical features explicitly disclosed in the
product description.

PRIMARY FEATURES:
- Core technical features and mechanisms.
- Features important for satisfying potential patent claim
  limitations.

SECONDARY FEATURES:
- Supporting, auxiliary, optional, control, communication,
  interface, or peripheral features.

Rules:
- Extract only explicitly disclosed features.
- Do not infer or invent features.
- Keep features concise and technically specific.
# - One feature per list item.

{feature_parser.get_format_instructions()}


Product Description:
{st.session_state.product_description}
"""

        from langchain_core.messages import HumanMessage
        from node import llm, feature_parser

        response = llm.invoke(
            [HumanMessage(content=product_prompt)]
        )

        result = feature_parser.parse(response.content)

        st.session_state.primary_product_features = (
            result.primary_features
        )

        st.session_state.secondary_product_features = (
            result.secondary_features
        )

        st.rerun()

    # Human approval
    st.subheader("Review Product Features")

    edited_primary = st.text_area(
        "Primary Product Features",
        value="\n".join(
            st.session_state.primary_product_features
        ),
        height=250,
        key="product_primary_review"
    )

    edited_secondary = st.text_area(
        "Secondary Product Features",
        value="\n".join(
            st.session_state.secondary_product_features
        ),
        height=150,
        key="product_secondary_review"
    )

    if st.button(
        "Approve Product Features",
        key="approve_product_features"
    ):

        st.session_state.primary_product_features = [
            x.strip()
            for x in edited_primary.split("\n")
            if x.strip()
        ]

        st.session_state.secondary_product_features = [
            x.strip()
            for x in edited_secondary.split("\n")
            if x.strip()
        ]

        st.session_state.product_features_approved = True

        st.rerun()

    st.stop()    

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
                # "product_features": [],
                "primary_product_features": st.session_state.primary_product_features,
                "secondary_product_features":  st.session_state.secondary_product_features,
                # "patent_features": [],
                "primary_patent_features": [],
                "secondary_patent_features": [],
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
            # st.write(snapshot.values)

            print("Next:", snapshot.next)
            print("Interrupts:", snapshot.interrupts)

            if snapshot.interrupts:
                print(snapshot.interrupts[0].value)

        # if snapshot.next:
        if snapshot.interrupts:

            review = snapshot.interrupts[0].value

            # The only expected interrupt is product feature approval
            if review["review_type"] == "product":

                st.info("Review the extracted product features")

                st.subheader("Primary Product Features")

                edited_primary = st.text_area(
                    "Primary Product Features",
                    value="\n".join(review["primary_features"]),
                    height=200,
                    key=f"primary_{index}_product"
                )

                st.subheader("Secondary Product Features")

                edited_secondary = st.text_area(
                    "Secondary Product Features",
                    value="\n".join(review["secondary_features"]),
                    height=200,
                    key=f"secondary_{index}_product"
                )

                if st.button(
                    "Approve Product Features",
                    key=f"approve_{index}_product"
                ):

                    approved_features = {
                        "primary_features": [
                            x.strip()
                            for x in edited_primary.split("\n")
                            if x.strip()
                        ],
                        "secondary_features": [
                            x.strip()
                            for x in edited_secondary.split("\n")
                            if x.strip()
                        ]
                    }

                    workflow.invoke(
                        Command(resume=approved_features),
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

                "Patent Primary Features": result["primary_patent_features"],
                "Patent Secondary Features": result["secondary_patent_features"],


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

# Logout
# --------------------------------------------------

if st.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.pop("username" , None)
    st.rerun()    
    
# --------------------------------------------------
# View Database
# --------------------------------------------------

st.divider()

if st.button("View Database"):

    conn = sqlite3.connect("RelevanceIQ.db")

    database_df = pd.read_sql_query(

        "SELECT * FROM patent_FTO_relevance_results ORDER BY id DESC",
        conn

    )

    conn.close()
    st.dataframe(
        database_df,
        use_container_width=True
    )
