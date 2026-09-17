import streamlit as st
import sqlite3
import uuid
import os
import textwrap

import pandas as pd
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from node import llm, feature_parser
from graph import workflow
from database import create_table, save_result

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RelevanceIQ | iCuerious",
    page_icon="Icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HTML HELPER
# ============================================================

def render_html(content):
    """
    Safely render custom HTML.

    textwrap.dedent() prevents Streamlit Markdown
    from interpreting indented HTML as a code block.
    """

    content = textwrap.dedent(content)

    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(
            content,
            unsafe_allow_html=True
        )


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    /* ========================================================
       iCUERIOUS / RELEVANCEIQ COLOR PALETTE
       ======================================================== */

    :root {
        --orange: #F4512A;
        --orange-dark: #D9401E;

        --navy: #071426;
        --navy-light: #101F32;

        --input-bg: #182536;
        --input-border: #3B4A5C;

        --white: #FFFFFF;

        --light-bg: #F7F7F7;
        --border: #D9DEE4;

        --text: #263238;
        --muted: #697586;
    }


    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: var(--light-bg);
        color: var(--text);
    }

    .main .block-container {
        max-width: 1400px;

        padding-top: 1.5rem;
        padding-bottom: 3rem;

        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #18283C;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    .sidebar-brand {
        padding: 12px 8px 25px 8px;
        border-bottom: 1px solid #29394C;
        margin-bottom: 20px;
    }

    .sidebar-brand-main {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -1px;

        color: #FFFFFF;
    }

    .sidebar-brand-main span {
        color: var(--orange);
    }

    .sidebar-brand-sub {
        font-size: 11px;
        color: #B8C2CE !important;
        margin-top: 3px;
        letter-spacing: 0.5px;
    }

    .sidebar-section {
        font-size: 11px;
        color: #8997A8 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px; 
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .sidebar-user {
        background: var(--navy-light);
        border: 1px solid #26384C;
        border-radius: 5px;
        padding: 14px;
        margin-top: 25px;
    }

    .sidebar-user-label {
        font-size: 10px;
        color: #8997A8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .sidebar-user-name {
        font-size: 14px;
        color: #FFFFFF !important;
        margin-top: 4px;
        word-break: break-word;
    }


    /* ========================================================
       SIDEBAR RADIO
       ======================================================== */

    section[data-testid="stSidebar"]
    div[role="radiogroup"] {
        gap: 5px;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        background: transparent;

        border-radius: 3px;

        padding: 7px 10px;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label:hover {
        background: #14253A;
    }


    /* ========================================================
       TOP HEADER
       ======================================================== */

    .top-header {
        background: #FFFFFF;
        border-bottom: 1px solid #DEDEDE;
        padding: 18px 28px;
        margin: -1.5rem -3rem 30px -3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-name {
        font-size: 26px;
        font-weight: 700;
        color: #1D1D1D;
        letter-spacing: -1px;
    }

    .brand-name span {
        color: var(--orange);
    }

    .brand-tagline {
        font-size: 10px;
        color: #666666;
        letter-spacing: 0.5px;
        margin-top: -3px;
    }

    .product-name {
        font-size: 20px;
        font-weight: 600;
        color: var(--navy);
        border-left: 2px solid var(--orange);
        padding-left: 15px;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background: var(--navy);
        border-radius: 4px;
        padding: 35px 42px;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";

        position: absolute;
        right: -100px;
        top: -120px;
        width: 350px;
        height: 350px;

        border: 1px solid rgba(244,81,42,0.25);
        border-radius: 50%;
    }

    .hero-kicker {
        color: var(--orange) !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    .hero-title {
        color: #FFFFFF !important;

        font-size: 34px;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .hero-description {
        color: #C5CED8 !important;

        font-size: 14px;
        max-width: 760px;
        line-height: 1.7;
    }


    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-header {
        display: flex;

        align-items: center;
        gap: 12px;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .section-number {
        width: 30px;
        height: 30px;

        background: var(--orange);
        color: #FFFFFF !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        border-radius: 50%;
        font-size: 13px;
        flex-shrink: 0;
    }

    .section-title {
        color: var(--navy) !important;
        font-size: 20px;
        font-weight: 600;
    }

    .section-description {
        color: var(--muted) !important;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 15px;
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .info-card {
        background: #FFFFFF;

        border: 1px solid var(--border);

        border-top: 3px solid var(--orange);

        border-radius: 4px;

        padding: 20px;

        height: 100%;
    }

    .info-card-title {
        font-size: 13px;

        color: var(--navy) !important;

        font-weight: 600;

        text-transform: uppercase;

        letter-spacing: 0.6px;
    }

    .info-card-value {
        font-size: 24px;

        font-weight: 600;

        color: var(--orange) !important;

        margin-top: 8px;
    }


    /* ========================================================
       TEXT AREAS
       
       IMPORTANT:
       USER ENTERED TEXT = WHITE
       BACKGROUND = DARK
       ======================================================== */

    div[data-testid="stTextArea"] {
        width: 100%;
    }

    div[data-testid="stTextArea"] > div {
        background-color: var(--input-bg) !important;

        border-radius: 4px !important;

        border: 1px solid var(--input-border) !important;
    }

    div[data-testid="stTextArea"] > div:focus-within {
        border: 1px solid var(--orange) !important;

        box-shadow:
            0 0 0 1px
            rgba(244,81,42,0.15) !important;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: var(--input-bg) !important;

        color: #FFFFFF !important;

        -webkit-text-fill-color: #FFFFFF !important;

        caret-color: #FFFFFF !important;

        border: none !important;

        font-size: 14px !important;

        line-height: 1.6 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        color: #FFFFFF !important;

        -webkit-text-fill-color: #FFFFFF !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #AEB9C5 !important;

        -webkit-text-fill-color: #AEB9C5 !important;

        opacity: 1 !important;
    }


    /* ========================================================
       BASEWEB TEXTAREA
       Extra protection for Streamlit versions
       ======================================================== */

    div[data-baseweb="textarea"] {
        background-color: var(--input-bg) !important;

        border-radius: 4px !important;

        border: 1px solid var(--input-border) !important;
    }

    div[data-baseweb="textarea"]:focus-within {
        border: 1px solid var(--orange) !important;

        box-shadow:
            0 0 0 1px
            rgba(244,81,42,0.15) !important;
    }

    div[data-baseweb="textarea"] textarea {
        background-color: var(--input-bg) !important;

        color: #FFFFFF !important;

        -webkit-text-fill-color: #FFFFFF !important;

        caret-color: #FFFFFF !important;

        border: none !important;

        font-size: 14px !important;

        line-height: 1.6 !important;
    }

    div[data-baseweb="textarea"] textarea::placeholder {
        color: #AEB9C5 !important;

        -webkit-text-fill-color: #AEB9C5 !important;

        opacity: 1 !important;
    }


    /* ========================================================
       TEXT INPUT / LOGIN INPUT
       ======================================================== */

    div[data-baseweb="input"] {
        background-color: var(--input-bg) !important;

        border: 1px solid var(--input-border) !important;

        border-radius: 4px !important;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1px solid var(--orange) !important;

        box-shadow:
            0 0 0 1px
            rgba(244,81,42,0.15) !important;
    }

    div[data-baseweb="input"] input {
        background-color: var(--input-bg) !important;

        color: #FFFFFF !important;

        -webkit-text-fill-color: #FFFFFF !important;

        caret-color: #FFFFFF !important;

        border: none !important;

        font-size: 14px !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #AEB9C5 !important;

        -webkit-text-fill-color: #AEB9C5 !important;

        opacity: 1 !important;
    }


    /* ========================================================
       ALL INPUT / TEXTAREA ELEMENTS
       ======================================================== */

    input,
    textarea {
        color: #FFFFFF !important;

        -webkit-text-fill-color: #FFFFFF !important;
    }


    /* ========================================================
       INPUT LABELS
       ======================================================== */

    label[data-testid="stWidgetLabel"] p {
        color: var(--navy) !important;

        font-weight: 600 !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: var(--orange) !important;

        color: #FFFFFF !important;

        border: 1px solid var(--orange) !important;

        border-radius: 3px !important;

        padding: 0.55rem 1.5rem !important;

        font-weight: 600 !important;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--orange-dark) !important;

        border-color: var(--orange-dark) !important;

        color: #FFFFFF !important;
    }

    .stButton > button:focus {
        color: #FFFFFF !important;

        border-color: var(--orange) !important;

        box-shadow:
            0 0 0 2px
            rgba(244,81,42,0.2) !important;
    }


    /* ========================================================
       DOWNLOAD BUTTON
       ======================================================== */

    .stDownloadButton > button {
        background-color: #FFFFFF !important;

        color: var(--navy) !important;

        border: 1px solid #BFC6CE !important;

        border-radius: 3px !important;

        font-weight: 600 !important;
    }

    .stDownloadButton > button:hover {
        background-color: #F4F5F6 !important;

        color: var(--navy) !important;

        border-color: var(--orange) !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    section[data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;

        border: 1px dashed #BFC6CE !important;

        border-radius: 4px !important;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: var(--navy) !important;
    }

    section[data-testid="stFileUploaderDropzone"] button {
        background: #FFFFFF !important;

        color: var(--navy) !important;

        border: 1px solid #BFC6CE !important;
    }


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    div[data-testid="stProgress"] > div > div {
        background-color: var(--orange) !important;
    }


    /* ========================================================
       RESULT BADGES
       ONLY H / M+ / L
       ======================================================== */

    .rating-H {
        display: inline-block;

        background: var(--orange);

        color: #FFFFFF !important;

        padding: 5px 14px;

        border-radius: 3px;

        font-weight: 700;

        font-size: 13px;
    }

    .rating-MPLUS {
        display: inline-block;

        background: #D96D31;

        color: #FFFFFF !important;

        padding: 5px 14px;

        border-radius: 3px;

        font-weight: 700;

        font-size: 13px;
    }

    .rating-L {
        display: inline-block;

        background: #73808C;

        color: #FFFFFF !important;

        padding: 5px 14px;

        border-radius: 3px;

        font-weight: 700;

        font-size: 13px;
    }

    .rating-NL {
        display: inline-block;

        background: #59636D;

        color: #FFFFFF !important;

        padding: 5px 14px;

        border-radius: 3px;

        font-weight: 700;

        font-size: 13px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-card {
        background: #FFFFFF;

        border: 1px solid var(--border);

        border-left: 4px solid var(--orange);

        padding: 15px 20px;

        border-radius: 3px;

        margin: 15px 0;
    }

    .status-title {
        font-weight: 600;

        color: var(--navy) !important;
    }

    .status-text {
        font-size: 13px;

        color: #68727D !important;
    }


    /* ========================================================
       INFO BOX
       ======================================================== */

    .input-info {
        background: #FFFFFF;

        border: 1px solid var(--border);

        padding: 14px 18px;

        border-radius: 3px;

        font-size: 12px;

        color: #697586 !important;

        line-height: 1.6;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);

        border-radius: 4px;

        overflow: hidden;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 4px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        border-top: 1px solid #DEDEDE;

        margin-top: 50px;

        padding-top: 20px;

        text-align: center;

        color: #7B858F !important;

        font-size: 11px;
    }


    /* ========================================================
       STREAMLIT CHROME
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """
)

# ============================================================
# LOGIN
# ============================================================

def login():

    render_html(
        """
        <div style="
            max-width:520px;
            margin:70px auto 25px auto;
            background:#FFFFFF;
            border:1px solid #DEDEDE;
            border-top:4px solid #F4512A;
            padding:45px;
            border-radius:4px;
        ">

            <div style="
                font-size:34px;
                font-weight:700;
                color:#071426;
            ">
                iCuerious<span style="color:#F4512A;">™</span>
            </div>

            <div style="
                font-size:11px;
                color:#777777;
                letter-spacing:1px;
                margin-top:-2px;
                margin-bottom:35px;
            ">
                FINDING WHAT MATTERS MOST
            </div>

            <div style="
                font-size:25px;
                font-weight:600;
                color:#071426;
                border-left:3px solid #F4512A;
                padding-left:12px;
                margin-bottom:8px;
            ">
                RelevanceIQ
            </div>

            <div style="
                font-size:13px;
                color:#6B7280;
                margin-bottom:5px;
            ">
                Patent Relevance Analysis
            </div>

        </div>
        """
    )
    username = st.text_input(
        "Username",
        placeholder="Enter username",
        key="login_username"
    )
    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password",
        key="login_password"
    )
    if st.button(
        "Login",
        use_container_width=True,
        key="login_button"
    ):

        users = st.secrets["users"]

        if username in users and password == users[username]:

            st.session_state.logged_in = True

            st.session_state.username = username

            st.rerun()

        else:

            st.error(
                "Invalid username or password"
            )


# ============================================================
# SESSION INITIALIZATION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    login()

    st.stop()


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


if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = False


# ============================================================
# DATABASE
# ============================================================

create_table()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-brand-main">
                iCuerious<span>™</span>
            </div>

            <div class="sidebar-brand-sub">
                FINDING WHAT MATTERS MOST
            </div>

        </div>
        """
    )


    render_html(
        """
        <div class="sidebar-section">
            Application
        </div>
        """
    )


    page = st.radio(
        "Navigation",
        [
            "Analysis",
            "Results",
            "Database"
        ],
        label_visibility="collapsed"
    )


    render_html(
        """
        <div class="sidebar-section">
            Tool
        </div>

        <div style="
            color:#FFFFFF;
            font-size:15px;
            font-weight:600;
        ">
            RelevanceIQ
        </div>

        <div style="
            color:#8997A8;
            font-size:11px;
            margin-top:4px;
            line-height:1.5;
        ">
            Patent relevance and FTO analysis
        </div>
        """
    )


    render_html(
        f"""
        <div class="sidebar-user">

            <div class="sidebar-user-label">
                Logged in as
            </div>

            <div class="sidebar-user-name">
                {st.session_state.username}
            </div>

        </div>
        """
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    if st.button(
        "Logout",
        use_container_width=True,
        key="logout_button"
    ):

        st.session_state.logged_in = False

        st.session_state.pop(
            "username",
            None
        )

        st.rerun()


# ============================================================
# TOP HEADER
# ============================================================

render_html(
    """
    <div class="top-header">

        <div>

            <div class="brand-name">
                iCuerious<span>™</span>
            </div>

            <div class="brand-tagline">
                FINDING WHAT MATTERS MOST
            </div>

        </div>

        <div class="product-name">
            RelevanceIQ
        </div>

    </div>
    """
)

# ============================================================
# DATABASE PAGE
# ============================================================

if page == "Database":

    render_html(
        """
        <div class="hero">

            <div class="hero-kicker">
                RelevanceIQ
            </div>

            <div class="hero-title">
                Analysis Database
            </div>

            <div class="hero-description">
                Review previously generated patent relevance
                analyses stored by the application.
            </div>

        </div>
        """
    )


    conn = sqlite3.connect(
        "RelevanceIQ.db"
    )


    try:

        database_df = pd.read_sql_query(
            """
            SELECT *
            FROM FTO_relevance_results
            ORDER BY id DESC
            """,
            conn
        )

    except Exception as e:

        database_df = pd.DataFrame()

        st.error(
            f"Unable to load database records: {e}"
        )

    finally:

        conn.close()


    if not database_df.empty:

    
        # Number of records
        st.write(
            f"**{len(database_df)} records found**"
        )

        st.dataframe(
            database_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Database",
            database_df.to_csv(index=False),
            "RelevanceIQ_Database.csv",
            "text/csv",
            key="download_database"
        )
    
    else:

        st.info(
            "No database records are available."
        )


    st.stop()


# ============================================================
# RESULTS PAGE
# ============================================================

if page == "Results":

    render_html(
        """
        <div class="hero">

            <div class="hero-kicker">
                RelevanceIQ
            </div>

            <div class="hero-title">
                Analysis Results
            </div>

            <div class="hero-description">
                Patent relevance results generated from the
                current analysis session.
            </div>

        </div>
        """
    )


    if st.session_state.results:

        result_df = pd.DataFrame(
            st.session_state.results
        )


        # ====================================================
        # MAIN RELEVANCE COUNTS
        # ONLY:
        # H
        # M+
        # L

        h_count = len(
            result_df[
                result_df["Relevance"] == "H"
            ]
        )


        mplus_count = len(
            result_df[
                result_df["Relevance"] == "M+"
            ]
        )


        l_count = len(
            result_df[
                result_df["Relevance"] == "L"
            ]
        )


        # Framework-only NL count
        nl_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "NL"
            ]
        )


        # ----------------------------------------------------
        # MAIN METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        Patents Analysed
                    </div>

                    <div class="info-card-value">
                        {len(result_df)}
                    </div>

                </div>
                """
            )


        with col2:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        Highly Relevant
                    </div>

                    <div class="info-card-value">
                        {h_count}
                    </div>

                </div>
                """
            )


        with col3:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        M+
                    </div>

                    <div class="info-card-value">
                        {mplus_count}
                    </div>

                </div>
                """
            )


        with col4:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        L
                    </div>

                    <div class="info-card-value">
                        {l_count}
                    </div>

                </div>
                """
            )


        # ----------------------------------------------------
        # FRAMEWORK-ONLY SUMMARY
        # ----------------------------------------------------

        render_html(
            """
            <div class="section-header">

                <div class="section-number">
                    F
                </div>

                <div class="section-title">
                    Framework-Only Classification
                </div>

            </div>

            <div class="section-description">
                Classification based specifically on the
                user-provided relevance framework.
            </div>
            """
        )


        framework_col1, framework_col2, framework_col3, framework_col4 = (
            st.columns(4)
        )


        framework_h_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "H"
            ]
        )


        framework_mplus_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "M+"
            ]
        )


        framework_l_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "L"
            ]
        )


        framework_nl_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "NL"
            ]
        )


        with framework_col1:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        H
                    </div>

                    <div class="info-card-value">
                        {framework_h_count}
                    </div>

                </div>
                """
            )


        with framework_col2:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        M+
                    </div>

                    <div class="info-card-value">
                        {framework_mplus_count}
                    </div>

                </div>
                """
            )


        with framework_col3:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        L
                    </div>

                    <div class="info-card-value">
                        {framework_l_count}
                    </div>

                </div>
                """
            )


        with framework_col4:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        NL
                    </div>

                    <div class="info-card-value">
                        {framework_nl_count}
                    </div>

                </div>
                """
            )


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        render_html(
            """
            <div class="section-header">

                <div class="section-number">
                    R
                </div>

                <div class="section-title">
                    Patent Results
                </div>

            </div>
            """
        )


        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.download_button(
            "Download Results",
            result_df.to_csv(index=False),
            "RelevanceIQ_Results.csv",
            "text/csv",
            key="download_results"
        )

    else:

        st.info(
            "No analysis results are available yet. "
            "Run an analysis first."
        )

    st.stop()


# ============================================================
# ANALYSIS PAGE
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="hero-kicker">
            Patent Intelligence • FTO
        </div>

        <div class="hero-title">
            Patent Relevance Analysis
        </div>

        <div class="hero-description">
            Evaluate patents against a target product using
            a defined relevance framework and human-reviewed
            technical feature extraction.
        </div>

    </div>
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

render_html(
    """
    <div class="section-header">

        <div class="section-number">
            1
        </div>

        <div class="section-title">
            Target Product
        </div>

    </div>

    <div class="section-description">
        Describe the technical product that you want to analyze.
    </div>
    """
)


product_description = st.text_area(
    "Product Description",

    value=st.session_state.product_description,

    height=180,

    placeholder=(
        "Enter the technical description of the target product..."
    ),

    label_visibility="collapsed",

    key="product_description_input"
)


# ============================================================
# PATENT FILE
# ============================================================

render_html(
    """
    <div class="section-header">

        <div class="section-number">
            2
        </div>

        <div class="section-title">
            Patent Dataset
        </div>

    </div>

    <div class="section-description">
        Upload the Excel file containing the patents to be evaluated.
    </div>
    """
)


uploaded_file = st.file_uploader(
    "Upload Patent Excel File",

    type=["xlsx"],

    label_visibility="collapsed",

    key="patent_file"
)


col1, col2 = st.columns([3, 1])


with col1:

    render_html(
        """
        <div class="input-info">

            Expected fields include
            <b>Publication Number</b>,
            <b>Title</b>,
            <b>Abstract</b>,
            <b>Independent Claim</b> and
            <b>All Claims</b>.

        </div>
        """
    )


with col2:

    template_path = (
        "RelevanceIQ_Input_Data_Schema.xlsx"
    )


    if os.path.exists(template_path):

        with open(
            template_path,
            "rb"
        ) as f:

            st.download_button(
                "Download Template",

                f.read(),

                "RelevanceIQ_Input_Data_Schema.xlsx",

                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                use_container_width=True,

                key="download_template"
            )


# ============================================================
# FRAMEWORK
# ============================================================

render_html(
    """
    <div class="section-header">

        <div class="section-number">
            3
        </div>

        <div class="section-title">
            Relevance Framework
        </div>

    </div>

    <div class="section-description">
        Define the criteria that will be used to determine
        patent relevance.
    </div>
    """
)


relevance_framework = st.text_area(
    "Relevance Framework",

    value=st.session_state.relevance_framework,

    height=220,

    placeholder=(
        "Enter the relevance framework used to classify "
        "patents as H, M+, L or NR..."
    ),

    label_visibility="collapsed",

    key="relevance_framework_input"
)


# ============================================================
# RUN ANALYSIS
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)


run_col1, run_col2, run_col3 = st.columns(
    [1, 2, 1]
)


with run_col2:

    run = st.button(
        "Run Relevance Analysis",

        use_container_width=True,

        key="run_analysis_button"
    )


if run:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if uploaded_file is None:

        st.warning(
            "Please upload an Excel file containing the patents."
        )

        st.stop()


    if not product_description.strip():

        st.warning(
            "Please enter the product description."
        )

        st.stop()


    if not relevance_framework.strip():

        st.warning(
            "Please provide the relevance framework."
        )

        st.stop()


    # --------------------------------------------------------
    # READ EXCEL
    # --------------------------------------------------------

    try:

        uploaded_df = pd.read_excel(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to read the Excel file: {e}"
        )

        st.stop()


    if uploaded_df.empty:

        st.warning(
            "The uploaded Excel file does not contain "
            "any patent rows."
        )

        st.stop()


    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "Publication Number",
        "Title", ## changed
        "Abstract",
        "Independent Claim",
        "All Claims"
    ]


    missing_columns = [
        column

        for column in required_columns

        if column not in uploaded_df.columns
    ]


    if missing_columns:

        st.error(
            "The uploaded Excel file is missing the following "
            f"required columns: {', '.join(missing_columns)}"
        )

        st.stop()


    # --------------------------------------------------------
    # SAVE SESSION DATA
    # --------------------------------------------------------

    st.session_state.df = uploaded_df


    st.session_state.product_description = (
        product_description
    )


    st.session_state.relevance_framework = (
        relevance_framework
    )


    st.session_state.results = []


    st.session_state.current_index = 0


    st.session_state.primary_product_features = []


    st.session_state.secondary_product_features = []


    st.session_state.product_features_approved = False


    st.session_state.analysis_started = True


    st.session_state.session_id = str(
        uuid.uuid4()
    )


    st.rerun()


# ============================================================
# PRODUCT FEATURE EXTRACTION
# ============================================================

if (
    st.session_state.analysis_started
    and not st.session_state.product_features_approved
):

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    if (
        not st.session_state.primary_product_features
        and not st.session_state.secondary_product_features
    ):

        render_html(
            """
            <div class="status-card">

                <div class="status-title">
                    Extracting Product Features
                </div>

                <div class="status-text">

                    RelevanceIQ is identifying the primary
                    and secondary technical features of the
                    target product for human review.

                </div>

            </div>
            """
        )


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

{feature_parser.get_format_instructions()}

Product Description:
{st.session_state.product_description}
"""


        response = llm.invoke(
            [
                HumanMessage(
                    content=product_prompt
                )
            ]
        )


        result = feature_parser.parse(
            response.content
        )


        st.session_state.primary_product_features = (
            result.primary_features
        )


        st.session_state.secondary_product_features = (
            result.secondary_features
        )


        st.rerun()


    # --------------------------------------------------------
    # HUMAN REVIEW
    # --------------------------------------------------------

    render_html(
        """
        <div class="section-header">

            <div class="section-number">
                ✓
            </div>

            <div class="section-title">
                Review Product Features
            </div>

        </div>

        <div class="section-description">

            Review and edit the extracted features before
            continuing with patent analysis.

        </div>
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        edited_primary = st.text_area(
            "Primary Product Features",

            value="\n".join(
                st.session_state.primary_product_features
            ),

            height=250,

            key="product_primary_review"
        )


    with col2:

        edited_secondary = st.text_area(
            "Secondary Product Features",

            value="\n".join(
                st.session_state.secondary_product_features
            ),

            height=250,

            key="product_secondary_review"
        )


    approve_col1, approve_col2, approve_col3 = st.columns(
        [1, 2, 1]
    )


    with approve_col2:

        if st.button(
            "Approve Product Features",

            use_container_width=True,

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


# ============================================================
# PATENT PROCESSING
# ============================================================

if st.session_state.analysis_started:

    df = st.session_state.df

    index = st.session_state.current_index


    # ========================================================
    # COMPLETED
    # ========================================================

    if index >= len(df):

        render_html(
            """
            <div class="hero">

                <div class="hero-kicker">
                    Analysis Complete
                </div>

                <div class="hero-title">
                    Patent Analysis Completed
                </div>

                <div class="hero-description">

                    The selected patent dataset has been
                    processed successfully.

                </div>

            </div>
            """
        )


        result_df = pd.DataFrame(
            st.session_state.results
        )


        # ----------------------------------------------------
        # MAIN RELEVANCE COUNTS
        #
        # ONLY H / M+ / L
        # ----------------------------------------------------

        h_count = len(
            result_df[
                result_df["Relevance"] == "H"
            ]
        )


        mplus_count = len(
            result_df[
                result_df["Relevance"] == "M+"
            ]
        )


        l_count = len(
            result_df[
                result_df["Relevance"] == "L"
            ]
        )


        # ----------------------------------------------------
        # FRAMEWORK-ONLY COUNTS
        #
        # H / M+ / L / NR
        # ----------------------------------------------------

        framework_h_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "H"
            ]
        )


        framework_mplus_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "M+"
            ]
        )


        framework_l_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "L"
            ]
        )


        framework_nl_count = len(
            result_df[
                result_df[
                    "Relevance_Framework_Only"
                ] == "NR"
            ]
        )


        # ====================================================
        # SUMMARY METRICS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        Patents
                    </div>

                    <div class="info-card-value">
                        {len(result_df)}
                    </div>

                </div>
                """
            )


        with col2:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        H
                    </div>

                    <div class="info-card-value">
                        {h_count}
                    </div>

                </div>
                """
            )


        with col3:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        M+
                    </div>

                    <div class="info-card-value">
                        {mplus_count}
                    </div>

                </div>
                """
            )


        with col4:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        L
                    </div>

                    <div class="info-card-value">
                        {l_count}
                    </div>

                </div>
                """
            )


        # ====================================================
        # FRAMEWORK ONLY SUMMARY
        # ====================================================

        render_html(
            """
            <div class="section-header">

                <div class="section-number">
                    F
                </div>

                <div class="section-title">
                    Framework-Only Results
                </div>

            </div>

            <div class="section-description">

                Classification based only on the
                user-provided relevance framework.

            </div>
            """
        )


        framework_col1, framework_col2, framework_col3, framework_col4 = (
            st.columns(4)
        )


        with framework_col1:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        H
                    </div>

                    <div class="info-card-value">
                        {framework_h_count}
                    </div>

                </div>
                """
            )


        with framework_col2:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        M+
                    </div>

                    <div class="info-card-value">
                        {framework_mplus_count}
                    </div>

                </div>
                """
            )


        with framework_col3:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        L
                    </div>

                    <div class="info-card-value">
                        {framework_l_count}
                    </div>

                </div>
                """
            )


        with framework_col4:

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        NR
                    </div>

                    <div class="info-card-value">
                        {framework_nl_count}
                    </div>

                </div>
                """
            )


        # ====================================================
        # RESULTS
        # ====================================================

        render_html(
            """
            <div class="section-header">

                <div class="section-number">
                    R
                </div>

                <div class="section-title">
                    Results
                </div>

            </div>
            """
        )


        st.dataframe(
            result_df,

            use_container_width=True,

            hide_index=True
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.download_button(
            "Download Results",

            result_df.to_csv(index=False),

            "RelevanceIQ_Results.csv",

            "text/csv",

            key="download_final_results"
        )


    # ========================================================
    # PROCESS CURRENT PATENT
    # ========================================================

    else:

        try:
            progress_value = (
                index / len(df)
            )


            st.progress(
                progress_value
            )


            render_html(
                f"""
                <div class="status-card">

                    <div class="status-title">
                        Patent {index + 1} of {len(df)}
                    </div>

                    <div class="status-text">

                        RelevanceIQ is processing the current patent.

                    </div>

                </div>
                """
            )


            row = df.iloc[index]


            config = {
                "configurable": {
                    "thread_id":
                    f"{st.session_state.session_id}_patent_{index}"
                }
            }


            snapshot = workflow.get_state(
                config
            )


            # ====================================================
            # FIRST EXECUTION
            # ====================================================

            if not snapshot or not snapshot.values:

                state = {

                    "product_description":
                        st.session_state.product_description,

                    "publication_number":
                        str(row["Publication Number"]),

                    "title":
                        row["Title"], ## changed

                    "abstract":
                        row["Abstract"],

                    "independent_claim":
                        row["Independent Claim"],

                    "all_claims":
                        row["All Claims"],

                    "relevance_framework":
                        st.session_state.relevance_framework,

                    "primary_product_features":
                        st.session_state.primary_product_features,

                    "secondary_product_features":
                        st.session_state.secondary_product_features,

                    "primary_patent_features":
                        [],

                    "secondary_patent_features":
                        [],

                    "comparison":
                        [],

                    "relevance":
                        "",

                    "confidence":
                        0,

                    "rationale":
                        "",

                    "reasoning":
                        "",    
            
                    "relevance_framework_only":
                        "",

                    "confidence_framework_only":
                        0,

                    "rationale_framework_only":
                        "",

                    "reasoning_framework_only":
                        "",    

                    "review_type":
                        ""
                }


                workflow.invoke(
                    state,
                    config=config
                )


                snapshot = workflow.get_state(
                    config
                )


            # ====================================================
            # HUMAN INTERRUPT
            # ====================================================

            if snapshot.interrupts:

                review = snapshot.interrupts[0].value


                if review["review_type"] == "product":

                    render_html(
                        """
                        <div class="section-header">

                            <div class="section-number">
                                !
                            </div>

                            <div class="section-title">
                                Human Review Required
                            </div>

                        </div>

                        <div class="section-description">

                            Review the extracted product features
                            before patent relevance analysis continues.

                        </div>
                        """
                    )


                    col1, col2 = st.columns(2)


                    with col1:

                        edited_primary = st.text_area(
                            "Primary Product Features",

                            value="\n".join(
                                review["primary_features"]
                            ),

                            height=230,

                            key=f"primary_{index}_product"
                        )


                    with col2:

                        edited_secondary = st.text_area(
                            "Secondary Product Features",

                            value="\n".join(
                                review["secondary_features"]
                            ),

                            height=230,

                            key=f"secondary_{index}_product"
                        )


                    if st.button(
                        "Approve & Continue",

                        key=f"approve_{index}_product",

                        use_container_width=True
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
                            Command(
                                resume=approved_features
                            ),

                            config=config
                        )


                        st.rerun()


            # ====================================================
            # CONTINUE WORKFLOW
            # ====================================================

            elif snapshot.next:

                st.info(
                    "Continuing patent analysis..."
                )

                st.rerun()


            # ====================================================
            # WORKFLOW FINISHED
            # ====================================================

            else:

                result = snapshot.values


                save_result(
                    result
                )

                st.session_state.results.append({

                    "Publication Number":
                        result["publication_number"],

                    "Title":
                        result["title"],

                    "Patent Primary Features":
                        result["primary_patent_features"],

                    "Patent Secondary Features":
                        result["secondary_patent_features"],

                    "Relevance":
                        result["relevance"],

                    "Confidence":
                        result["confidence"],

                    "Rationale":
                        result["rationale"],

                    "Reasoning":
                            result["reasoning"],    

                    "Relevance_Framework_Only":
                        result["relevance_framework_only"],

                    "Confidence_Framework_Only":
                        result["confidence_framework_only"],

                    "Rationale_Framework_Only":
                        result["rationale_framework_only"] ,

                    "Reasoning_Framework_Only":
                        result["reasoning_framework_only"]    

                })


                st.session_state.current_index += 1


                st.rerun()


        except Exception as e:

            st.session_state.analysis_started = False
            st.session_state.analysis_error = True

            st.error(
                f"Analysis stopped at Patent {index + 1}: "
                f"{row['Publication Number']}"
            )

            st.warning(
                f"Error: {str(e)}"
            )

            st.info(
                "Correct the issue and click Resume Analysis "
                "to continue from this patent."
            )

if st.session_state.get("analysis_error", False):

    if st.button(
        "▶ Resume Analysis",
        use_container_width=True,
        key="resume_analysis"
    ):

        st.session_state.analysis_error = False
        st.session_state.analysis_started = True

        st.rerun()



# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">

        RelevanceIQ &nbsp; | &nbsp;
        iCuerious Patent Intelligence

        <br><br>

        Patent relevance analysis • FTO support •
        Technology intelligence

    </div>
    """
)