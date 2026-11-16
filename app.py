import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------
# CONFIG
# --------------------------------

st.set_page_config(
    page_title="SAP Public Sector Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SAP Public Sector Dashboard")

# --------------------------------
# OPENAI
# --------------------------------

OPENAI_API_KEY = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------
# DATA LOADING
# --------------------------------

@st.cache_data
def load_data(file):

    ugovori = pd.read_excel(
        file,
        sheet_name="Уговори"
    )

    okvirni = pd.read_excel(
        file,
        sheet_name="Оквирни споразуми"
    )

    ugovori["Tip"] = "Ugovor"
    okvirni["Tip"] = "Okvirni sporazum"

    ugovori = ugovori.rename(
        columns={
            "Наручилац": "Klijent",
            "Предмет набавке": "Predmet",
            "Назив уговорне стране": "Partner",
            "Укупни износ": "Vrednost",
            "Датум закључења": "Datum",
            "Трајање (месеци)": "Trajanje"
        }
    )

    okvirni = okvirni.rename(
        columns={
            "Наручилац": "Klijent",
            "Назив предмета": "Predmet",
            "Уговорна страна": "Partner",
            "Вред. О.С.": "Vrednost",
            "Закључен": "Datum",
            "Трајање (месец)": "Trajanje"
        }
    )

    cols = [
        "Klijent",
        "Predmet",
        "Partner",
        "Vrednost",
        "Datum",
        "Trajanje",
        "Tip"
    ]

    ugovori = ugovori[cols]
    okvirni = okvirni[cols]

    df = pd.concat(
        [ugovori, okvirni],
        ignore_index=True
    )

    df["Datum"] = pd.to_datetime(
        df["Datum"],
        errors="coerce"
    )

    df["Trajanje"] = pd.to_numeric(
        df["Trajanje"],
        errors="coerce"
    )

    df["Istek"] = (
        df["Datum"] +
        pd.to_timedelta(
            df["Trajanje"].fillna(0) * 30,
            unit="D"
        )
    )

    return df


# --------------------------------
# RAG
# --------------------------------

@st.cache_resource
def create_vector_store(df):

    docs = []

    for _, row in df.iterrows():

        text = f"""
Klijent: {row['Klijent']}
Partner: {row['Partner']}
Predmet: {row['Predmet']}
Vrednost: {row['Vrednost']}
Datum: {row['Datum']}
Istek: {row['Istek']}
Tip: {row['Tip']}
"""

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "partner": str(row["Partner"]),
                    "klijent": str(row["Klijent"])
                }
            )
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(
        docs,
        embeddings
    )


# --------------------------------
# FILE UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload SAP Excel",
    type=["xlsx"]
)

if not uploaded_file:
    st.info("Upload Excel fajl.")
    st.stop()

df = load_data(uploaded_file)

vectorstore = create_vector_store(df)

# --------------------------------
# SIDEBAR FILTERS
# --------------------------------

st.sidebar.header("Filteri")

partner_filter = st.sidebar.multiselect(
    "Partner",
    sorted(df["Partner"].dropna().unique())
)

client_filter = st.sidebar.multiselect(
    "Klijent",
    sorted(df["Klijent"].dropna().unique())
)

filtered = df.copy()

if partner_filter:
    filtered = filtered[
        filtered["Partner"].isin(
            partner_filter
        )
    ]

if client_filter:
    filtered = filtered[
        filtered["Klijent"].isin(
            client_filter
        )
    ]

# --------------------------------
# KPIs
# --------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Broj ugovora",
    len(filtered)
)

col2.metric(
    "Ukupna vrednost",
    f"{filtered['Vrednost'].sum():,.0f} RSD"
)

col3.metric(
    "Klijenti",
    filtered["Klijent"].nunique()
)

col4.metric(
    "Partneri",
    filtered["Partner"].nunique()
)

st.divider()

# --------------------------------
# TABS
# --------------------------------

tabs = st.tabs([
    "Partneri",
    "Klijenti",
    "Top ugovori",
    "Tender radar",
    "Podaci",
    "AI Chat"
])

# --------------------------------
# PARTNERI
# --------------------------------

with tabs[0]:

    partner_summary = (
        filtered
        .groupby("Partner")
        .agg(
            Broj_ugovora=("Predmet", "count"),
            Ukupna_vrednost=("Vrednost", "sum"),
            Broj_klijenata=("Klijent", "nunique")
        )
        .reset_index()
        .sort_values(
            "Ukupna_vrednost",
            ascending=False
        )
    )

    st.dataframe(
        partner_summary,
        use_container_width=True
    )

    fig = px.bar(
        partner_summary.head(15),
        x="Partner",
        y="Ukupna_vrednost",
        title="Top partneri"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------
# KLIJENTI
# --------------------------------

with tabs[1]:

    client_summary = (
        filtered
        .groupby("Klijent")
        .agg(
            Broj_ugovora=("Predmet", "count"),
            Ukupna_vrednost=("Vrednost", "sum")
        )
        .reset_index()
        .sort_values(
            "Ukupna_vrednost",
            ascending=False
        )
    )

    st.dataframe(
        client_summary,
        use_container_width=True
    )

# --------------------------------
# TOP UGOVORI
# --------------------------------

with tabs[2]:

    st.dataframe(
        filtered
        .sort_values(
            "Vrednost",
            ascending=False
        )
        .head(25),
        use_container_width=True
    )

# --------------------------------
# TENDER RADAR
# --------------------------------

with tabs[3]:

    today = pd.Timestamp.today()

    radar = filtered[
        filtered["Istek"].notna()
    ].copy()

    radar["Dana_do_isteka"] = (
        radar["Istek"] - today
    ).dt.days

    radar = radar.sort_values(
        "Dana_do_isteka"
    )

    st.dataframe(
        radar[
            [
                "Klijent",
                "Partner",
                "Predmet",
                "Istek",
                "Dana_do_isteka"
            ]
        ],
        use_container_width=True
    )

# --------------------------------
# PODACI
# --------------------------------

with tabs[4]:

    st.dataframe(
        filtered,
        use_container_width=True
    )

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Export CSV",
        data=csv,
        file_name="sap_export.csv",
        mime="text/csv"
    )

# --------------------------------
# AI CHAT
# --------------------------------

with tabs[5]:

    st.subheader(
        "🤖 SAP Tender AI Assistant"
    )

    st.markdown("""
Primer pitanja:

- Koje ugovore ima Prointer?
- Ko dominira u EPS-u?
- Koji ugovori ističu narednih 6 meseci?
- Ko ima najviše SAP poslova?
- Prikaži SAP HCM projekte.
- Kolika je vrednost poslova Flanera?
""")

    question = st.text_input(
        "Postavi pitanje"
    )

    if question:

        if not client:
            st.error(
                "Unesi OpenAI API ključ."
            )
        else:

            docs = vectorstore.similarity_search(
                question,
                k=20
            )

            context = "\n\n".join(
                [
                    d.page_content
                    for d in docs
                ]
            )

            prompt = f"""
Ti si senior SAP market analyst za javni sektor Srbije.

Koristi isključivo informacije koje se nalaze u kontekstu.

Ako nema dovoljno podataka reci da odgovor nije pronađen.

KONTEKST:

{context}

PITANJE:

{question}
"""

            with st.spinner(
                "Analiziram..."
            ):

                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system",
                            "content": "Ti si SAP market analyst."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1
                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                st.success(answer)
