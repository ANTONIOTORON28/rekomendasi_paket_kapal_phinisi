import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Rekomendasi Kapal Phinisi",
    page_icon="🚢",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        "dataset_kapal_preprocessing.csv"
    )

    df.columns = df.columns.str.strip()

    return df


# ==========================================
# LOAD MODEL SBERT
# ==========================================
@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# ==========================================
# CREATE EMBEDDING
# ==========================================
@st.cache_resource
def create_embeddings(
    data
):

    texts = data[
        "processed_text"
    ].fillna("").astype(str).tolist()

    embeddings = model.encode(
        texts
    )

    return embeddings


# ==========================================
# LOAD EVERYTHING
# ==========================================
df = load_data()

model = load_model()

embeddings = create_embeddings(
    df
)


# ==========================================
# COSINE SIMILARITY
# ==========================================
similarity_matrix = cosine_similarity(
    embeddings
)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df["nama_kapal"],
    columns=df["nama_kapal"]
)


# ==========================================
# RECOMMEND FUNCTION
# ==========================================
def rekomendasi_kapal(
    nama_kapal,
    top_n=5
):

    similarity_scores = similarity_df[
        nama_kapal
    ].sort_values(
        ascending=False
    )

    similarity_scores = similarity_scores[
        1:top_n+1
    ]

    hasil = []

    for kapal, score in similarity_scores.items():

        kapal_data = df[
            df["nama_kapal"] == kapal
        ].iloc[0]

        hasil.append({

            "nama_kapal":
            kapal,

            "kategori":
            kapal_data.get(
                "kategori",
                "-"
            ),

            "harga":
            kapal_data.get(
                "harga",
                "-"
            ),

            "kapasitas":
            kapal_data.get(
                "kapasitas",
                "-"
            ),

            "cabin":
            kapal_data.get(
                "cabin",
                "-"
            ),

            "image_url":
            kapal_data.get(
                "image_url",
                ""
            ),

            "similarity":
            round(
                score,
                4
            )

        })

    return hasil


# ==========================================
# HEADER
# ==========================================
st.title(
    "🚢 Sistem Rekomendasi Kapal Phinisi"
)

st.write(
    """
    Sistem rekomendasi kapal wisata
    menggunakan Sentence-BERT dan
    Cosine Similarity.
    """
)


# ==========================================
# USER INPUT
# ==========================================
selected_kapal = st.selectbox(

    "Pilih Kapal:",

    sorted(
        df["nama_kapal"].dropna().unique()
    )

)

top_n = st.slider(

    "Jumlah Rekomendasi",

    min_value=1,

    max_value=10,

    value=5

)


# ==========================================
# BUTTON
# ==========================================
if st.button(
    "Cari Rekomendasi"
):

    hasil = rekomendasi_kapal(

        selected_kapal,

        top_n

    )

    st.subheader(
        "Hasil Rekomendasi"
    )


    for item in hasil:

        col1, col2 = st.columns(
            [1, 3]
        )

        with col1:

            if (
                item["image_url"]
                and str(
                    item["image_url"]
                ).startswith("http")
            ):

                st.image(
                    item["image_url"],
                    use_container_width=True
                )

        with col2:

            st.markdown(
                f"""
                ### {item['nama_kapal']}

                **Kategori:** {item['kategori']}

                **Harga:** {item['harga']}

                **Kapasitas:** {item['kapasitas']}

                **Cabin:** {item['cabin']}

                **Similarity Score:** {item['similarity']}
                """
            )

            st.divider()