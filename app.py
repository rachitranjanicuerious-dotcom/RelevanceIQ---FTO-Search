import streamlit as st
import sqlite3
import pandas as pd

# TEST_MODE = True

from graph_3_node import workflow

from database import create_table, save_result

st.set_page_config(
    page_title="RelevanceIQ",
    page_icon=" ",
    layout="centered"
)

st.title("RelevanceIQ")
st.subheader("Patent Relevance Analysis")

create_table()

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

    elif not product_description.strip():
        st.warning("Please enter the product description.")

    elif not relevance_framework.strip():
        st.warning("Please provide the relevance framework.")


    else:
      
        df = pd.read_excel(uploaded_file)
        results = []
        progress_bar = st.progress(0)

        status = st.empty()

        total_patents = len(df)

        for index, row in df.iterrows():

            status.write(
                 f"Processing Patent {index + 1} of {total_patents}"
            )

            state = {
                "product_description" : product_description,
                "publication_number"  : row["Publication Number"],
                "title"               : row["title"],
                "abstract"            : row["Abstract"],
                "independent_claim"   : row["Independent Claim"],
                "all_claims"          : row["All Claims"],
                "relevance_framework" : relevance_framework, 
                "product_features"     : [],
                "patent_features"     : [],
                "comparison"          : [],
                "relevance"           : " ",
                "rationale"           : " ",
                "confidence"          : 0

            }
            print(state)
            
            result = workflow.invoke(state)

            save_result(result)

            print("Workflow completed for:", result["publication_number"])

            results.append({
                "Publication Number"  : result["publication_number"],
                "Title": result["title"],
                "Relevance": result["relevance"],
                "Confidence": result["confidence"],
                "Rationale": result["rationale"]
                
            })

            progress_bar.progress(
            (index + 1) / total_patents
            )

        status.empty()

        st.success("Analysis Completed Successfully!")

        result_df = pd.DataFrame(results)

        st.dataframe(
            result_df,
            use_container_width = True
        )

        st.download_button(
            label = "Download Results",
            data = result_df.to_csv(index = False),
            file_name = 'RelevanceIQ_Results.csv',
            mime      = "text/csv"
        )

if st.button("View Database"):

    conn = sqlite3.connect("RelevanceIQ.db")

    df = pd.read_sql_query(
        "SELECT * FROM patent_results",
        conn
    )

    conn.close()

    st.dataframe(df, use_container_width=True)


            