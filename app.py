import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Rekomendasi Paket Wisata Phinisi",
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
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# ==========================================
# LOAD OBJECT
# ==========================================
df = load_data()

model = load_model()


# ==========================================
# CREATE EMBEDDING
# ==========================================
@st.cache_resource
def create_embeddings():

    texts = df[
        "processed_text"
    ].fillna("").astype(str).tolist()

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings


embeddings = create_embeddings()


# ==========================================
# EVALUATION METRICS
# ==========================================
def precision_at_k(
    relevant_items,
    recommended_items,
    k
):

    recommended_k = recommended_items[:k]

    hit_count = len(
        set(
            recommended_k
        ).intersection(
            set(
                relevant_items
            )
        )
    )

    return hit_count / k


def recall_at_k(
    relevant_items,
    recommended_items,
    k
):

    if len(
        relevant_items
    ) == 0:

        return 0


    recommended_k = recommended_items[:k]

    hit_count = len(
        set(
            recommended_k
        ).intersection(
            set(
                relevant_items
            )
        )
    )

    return hit_count / len(
        relevant_items
    )


def average_precision_at_k(
    relevant_items,
    recommended_items,
    k
):

    score = 0

    hit_count = 0

    recommended_k = recommended_items[:k]


    for i, item in enumerate(

        recommended_k,

        start=1

    ):

        if item in relevant_items:

            hit_count += 1

            score += (
                hit_count / i
            )


    if hit_count == 0:

        return 0


    return score / min(

        len(
            relevant_items
        ),

        k

    )


def evaluate_model(
    selected_category,
    recommended_items,
    k
):

    relevant_items = df[
        df[
            "kategori"
        ] == selected_category
    ][
        "nama_kapal"
    ].tolist()


    precision = precision_at_k(

        relevant_items,

        recommended_items,

        k

    )


    recall = recall_at_k(

        relevant_items,

        recommended_items,

        k

    )


    map_score = average_precision_at_k(

        relevant_items,

        recommended_items,

        k

    )


    return (

        precision,

        recall,

        map_score

    )


# ==========================================
# UI HEADER
# ==========================================
st.title(
    "🚢 Sistem Rekomendasi Paket Wisata Phinisi"
)

st.write(
    """
    Cari paket wisata terbaik
    berdasarkan jenis trip
    dan kebutuhan perjalanan Anda.
    """
)

st.divider()


# ==========================================
# INPUT
# ==========================================
col1, col2 = st.columns(
    2
)


with col1:

    selected_paket = st.selectbox(

        "Pilih Paket Wisata",

        [

            "Private Trip",

            "Open Trip",

            "Family Trip",

            "Honeymoon",

            "Diving Trip",

            "Luxury Trip"

        ]

    )


with col2:

    top_n = st.slider(

        "Top Recommendation",

        1,

        10,

        5

    )


user_desc = st.text_area(

    "Deskripsikan kebutuhan perjalanan",

    placeholder="""
Contoh:
private trip dengan snorkeling,
jacuzzi, spa, sunset dinner,
chef pribadi
"""

)


# ==========================================
# SEARCH
# ==========================================
if st.button(
    "🔍 Cari Rekomendasi"
):

    if user_desc.strip() == "":

        st.warning(
            "Silakan isi deskripsi perjalanan."
        )

        st.stop()


    # ======================================
    # USER QUERY
    # ======================================
    query = (

        selected_paket

        + " " +

        user_desc

    )


    # ======================================
    # SBERT
    # ======================================
    query_embedding = model.encode(
        [query]
    )


    # ======================================
    # COSINE SIMILARITY
    # ======================================
    scores = cosine_similarity(

        query_embedding,

        embeddings

    )[0]


    # ======================================
    # RANKING
    # ======================================
    top_indices = scores.argsort()[
        ::-1
    ][
        :top_n
    ]


    # ======================================
    # LIST RECOMMENDED
    # ======================================
    recommended_names = []


    for idx in top_indices:

        kapal_name = df.iloc[
            idx
        ][
            "nama_kapal"
        ]


        recommended_names.append(
            kapal_name
        )


    # ======================================
    # EVALUATION
    # ======================================
    precision, recall, map_score = evaluate_model(

        selected_paket,

        recommended_names,

        top_n

    )


    # ======================================
    # SHOW METRICS
    # ======================================
    st.subheader(
        "📊 Evaluasi Model"
    )


    m1, m2, m3 = st.columns(
        3
    )


    with m1:

        st.metric(

            "Precision",

            round(
                precision,
                4
            )

        )


    with m2:

        st.metric(

            "Recall",

            round(
                recall,
                4
            )

        )


    with m3:

        st.metric(

            "MAP",

            round(
                map_score,
                4
            )

        )


    st.divider()


    # ======================================
    # SHOW RECOMMENDATION
    # ======================================
    st.subheader(
        "✨ Paket Terbaik Untuk Anda"
    )


    for rank, idx in enumerate(

        top_indices,

        start=1

    ):

        item = df.iloc[
            idx
        ]


        col_img, col_info = st.columns(
            [1, 3]
        )


        # ==============================
        # IMAGE
        # ==============================
        with col_img:

            img_url = str(

                item.get(

                    "image_url",

                    ""

                )

            )


            if img_url.startswith(
                "http"
            ):

                st.image(

                    img_url,

                    use_container_width=True

                )


        # ==============================
        # INFO
        # ==============================
        with col_info:

            st.markdown(
                f"""
                ## #{rank} {item.get('nama_kapal','-')}

                **Kategori:** {item.get('kategori','-')}

                **Harga:** {item.get('harga','-')}

                **Destinasi:** {item.get('destinasi','-')}

                **Cabin:** {item.get('cabin','-')}

                **Fasilitas:** {item.get('fasilitas','-')}

                **Layanan:** {item.get('layanan','-')}

                **Similarity Score:** {round(scores[idx],4)}
                """
            )

            st.divider()
