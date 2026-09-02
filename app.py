import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartShop AI",
    page_icon="🛍️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.product-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-bottom: 15px;
}

.price {
    font-size: 22px;
    font-weight: bold;
}

.rating {
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🛍️ SmartShop AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered E-Commerce Product Recommendation System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PRODUCT DATA
# =========================================================

products = {

    "product_name": [
        "Nike Air Max Running Shoes",
        "Adidas Sports Running Shoes",
        "Puma Casual Sneakers",
        "Nike Sports T-Shirt",
        "Adidas Cotton T-Shirt",
        "Puma Training T-Shirt",
        "Apple iPhone 15",
        "Samsung Galaxy S24",
        "OnePlus 12 Smartphone",
        "Apple MacBook Air",
        "Dell Inspiron Laptop",
        "HP Pavilion Laptop",
        "Sony Wireless Headphones",
        "Boat Bluetooth Headphones",
        "JBL Wireless Earbuds",
        "Samsung Smart TV",
        "LG 4K Ultra HD TV",
        "Sony Bravia Smart TV",
        "Nike Sports Backpack",
        "Adidas School Backpack"
    ],

    "category": [
        "Shoes",
        "Shoes",
        "Shoes",
        "Clothing",
        "Clothing",
        "Clothing",
        "Mobiles",
        "Mobiles",
        "Mobiles",
        "Laptops",
        "Laptops",
        "Laptops",
        "Headphones",
        "Headphones",
        "Headphones",
        "Television",
        "Television",
        "Television",
        "Bags",
        "Bags"
    ],

    "price": [
        4999,
        4299,
        2999,
        1499,
        999,
        1199,
        69999,
        74999,
        54999,
        89999,
        64999,
        59999,
        7999,
        2499,
        2999,
        44999,
        39999,
        59999,
        1999,
        1799
    ],

    "rating": [
        4.5,
        4.4,
        4.2,
        4.3,
        4.1,
        4.2,
        4.7,
        4.6,
        4.5,
        4.8,
        4.4,
        4.3,
        4.5,
        4.2,
        4.4,
        4.6,
        4.5,
        4.7,
        4.3,
        4.2
    ],

    "image": [
        "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
        "https://images.unsplash.com/photo-1552346154-21d32810aba3",
        "https://images.unsplash.com/photo-1549298916-b41d501d3772",
        "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab",
        "https://images.unsplash.com/photo-1503341504253-dff4815485f1",
        "https://images.unsplash.com/photo-1562157873-818bc0726f68",
        "https://images.unsplash.com/photo-1592899677977-9c10ca588bbd",
        "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf",
        "https://images.unsplash.com/photo-1598327105666-5b89351aff97",
        "https://images.unsplash.com/photo-1517336714731-489689fd1ca8",
        "https://images.unsplash.com/photo-1496181133206-80ce9b88a853",
        "https://images.unsplash.com/photo-1496180727794-817822f65950",
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",
        "https://images.unsplash.com/photo-1590658268037-6bf12165a8df",
        "https://images.unsplash.com/photo-1593359677879-a4bb92f829d1",
        "https://images.unsplash.com/photo-1567690187548-f07b1d7bf5a9",
        "https://images.unsplash.com/photo-1601944179066-29786cb9d32a",
        "https://images.unsplash.com/photo-1553062407-98eeb64c6a62",
        "https://images.unsplash.com/photo-1585914641050-7a0da59e4e67"
    ],

    "description": [

        "Nike running shoes comfortable sports footwear for running exercise gym and fitness",

        "Adidas running shoes lightweight sports footwear for running fitness and exercise",

        "Puma casual sneakers stylish comfortable shoes for daily wear walking and lifestyle",

        "Nike sports t-shirt comfortable athletic clothing for gym running and exercise",

        "Adidas cotton t-shirt soft comfortable casual clothing for everyday use",

        "Puma training t-shirt sports clothing for gym workout training and exercise",

        "Apple iPhone smartphone powerful processor excellent camera premium design and advanced features",

        "Samsung Galaxy smartphone powerful processor excellent camera Android features and premium display",

        "OnePlus smartphone fast processor excellent camera premium display and powerful performance",

        "Apple MacBook Air lightweight laptop powerful processor long battery life and premium design",

        "Dell Inspiron laptop suitable for students work office and everyday computing",

        "HP Pavilion laptop powerful performance suitable for students office work and entertainment",

        "Sony wireless headphones high quality sound noise cancellation Bluetooth and comfortable design",

        "Boat Bluetooth headphones affordable wireless headphones powerful sound Bluetooth connectivity",

        "JBL wireless earbuds compact Bluetooth earbuds clear sound comfortable fit and portable design",

        "Samsung smart television 4K display streaming smart features excellent picture quality",

        "LG 4K Ultra HD television smart TV features excellent picture quality streaming and entertainment",

        "Sony Bravia smart television 4K display streaming premium picture quality and smart features",

        "Nike sports backpack durable bag for gym travel college and daily use",

        "Adidas school backpack durable comfortable bag for students college and everyday use"
    ]
}


df = pd.DataFrame(products)


# =========================================================
# COMBINE TEXT FEATURES
# =========================================================

df["combined_features"] = (
    df["product_name"] + " " +
    df["category"] + " " +
    df["description"]
)


# =========================================================
# TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    df["combined_features"]
)


# =========================================================
# COSINE SIMILARITY
# =========================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_products(product_name, number_of_recommendations):

    product_index = df[
        df["product_name"] == product_name
    ].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[product_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[
        1:number_of_recommendations + 1
    ]:

        recommendations.append({
            "index": index,
            "similarity": score
        })

    return recommendations


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Search & Filters")

search_text = st.sidebar.text_input(
    "🔎 Search Product"
)

categories = ["All"] + sorted(
    df["category"].unique().tolist()
)

selected_category = st.sidebar.selectbox(
    "📂 Select Category",
    categories
)

number_of_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    1,
    10,
    5
)


# =========================================================
# FILTER PRODUCTS
# =========================================================

filtered_df = df.copy()

if search_text:

    filtered_df = filtered_df[
        filtered_df["product_name"]
        .str.contains(
            search_text,
            case=False,
            na=False
        )
    ]

if selected_category != "All":

    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]


# =========================================================
# PRODUCT SELECTION
# =========================================================

st.subheader("🔎 Choose a Product")

if len(filtered_df) == 0:

    st.warning(
        "No products found. Try another search."
    )

    st.stop()


selected_product = st.selectbox(
    "Select a product:",
    filtered_df["product_name"].tolist()
)


# =========================================================
# SELECTED PRODUCT DETAILS
# =========================================================

selected_data = df[
    df["product_name"] == selected_product
].iloc[0]


st.markdown("---")

st.subheader("📦 Selected Product")

col1, col2 = st.columns([1, 2])


with col1:

    st.image(
        selected_data["image"],
        use_container_width=True
    )


with col2:

    st.markdown(
        f"## {selected_data['product_name']}"
    )

    st.write(
        f"**Category:** {selected_data['category']}"
    )

    st.markdown(
        f"### ₹{selected_data['price']:,}"
    )

    st.write(
        f"⭐ **Rating:** {selected_data['rating']} / 5"
    )

    st.write(
        selected_data["description"]
    )


# =========================================================
# RECOMMENDATION BUTTON
# =========================================================

st.markdown("---")

if st.button(
    "✨ Get AI Recommendations",
    use_container_width=True
):

    recommendations = recommend_products(
        selected_product,
        number_of_recommendations
    )

    st.subheader("🛒 Recommended Products")

    for recommendation in recommendations:

        index = recommendation["index"]

        similarity = recommendation["similarity"]

        product = df.iloc[index]

        col1, col2 = st.columns(
            [1, 3]
        )

        with col1:

            st.image(
                product["image"],
                use_container_width=True
            )

        with col2:

            st.markdown(
                f"### {product['product_name']}"
            )

            st.write(
                f"**Category:** {product['category']}"
            )

            st.markdown(
                f"### ₹{product['price']:,}"
            )

            st.write(
                f"⭐ Rating: {product['rating']} / 5"
            )

            st.progress(
                float(similarity)
            )

            st.write(
                f"AI Similarity Score: "
                f"{similarity * 100:.2f}%"
            )

            st.write(
                product["description"]
            )

        st.divider()


# =========================================================
# PRODUCT CATALOG
# =========================================================

st.markdown("---")

st.subheader("🛍️ Product Catalog")

catalog_df = filtered_df[
    [
        "product_name",
        "category",
        "price",
        "rating"
    ]
].copy()

catalog_df["price"] = (
    "₹" +
    catalog_df["price"]
    .map(lambda x: f"{x:,}")
)

catalog_df["rating"] = (
    catalog_df["rating"].astype(str) +
    " ⭐"
)

st.dataframe(
    catalog_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ABOUT SECTION
# =========================================================

st.markdown("---")

st.subheader("🤖 How Does the AI Work?")

st.write(
    """
    This recommendation system uses Natural Language Processing
    (NLP) to understand product information.

    **1. TF-IDF**

    Product names, categories and descriptions are converted
    into numerical vectors using TF-IDF.

    **2. Cosine Similarity**

    The system compares these vectors and calculates how similar
    different products are.

    **3. Recommendation**

    Products with the highest similarity scores are recommended
    to the user.

    **Technologies Used:**

    Python • Streamlit • Pandas • Scikit-learn • NLP • TF-IDF
    • Cosine Similarity
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🛍️ SmartShop AI | E-Commerce Recommendation System"
)