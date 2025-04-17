
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Airline-Sentiment-2-w-AA.csv", encoding='latin1')
    df = df[['text', 'airline', 'airline_sentiment', 'negativereason']].copy()

    def clean_tweet(text):
        hashtags = re.findall(r"#(\w+)", text)
        mentions = re.findall(r"@(\w+)", text)
        text_cleaned = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|http\S+", "", text)
        text_cleaned = re.sub(r"[^A-Za-z\s]", "", text_cleaned).lower()
        return text_cleaned.strip(), hashtags, mentions

    df[['clean_text', 'hashtags', 'mentions']] = df['text'].apply(lambda x: pd.Series(clean_tweet(x)))
    df.dropna(subset=['clean_text'], inplace=True)
    df = df[df['clean_text'].str.strip() != ""]
    return df

# Run clustering
@st.cache_data
def run_clustering(df, num_clusters=5):
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(tfidf_matrix.toarray())
    df['pca_x'] = reduced_features[:, 0]
    df['pca_y'] = reduced_features[:, 1]
    return df

# Main app
st.title("Airline Sentiment Explorer ✈️")
df = load_data()
df_clustered = run_clustering(df)

st.sidebar.header("Filters")
selected_airline = st.sidebar.multiselect("Select Airline", options=df['airline'].unique(), default=df['airline'].unique())
selected_sentiment = st.sidebar.multiselect("Select Sentiment", options=df['airline_sentiment'].unique(), default=df['airline_sentiment'].unique())
selected_cluster = st.sidebar.multiselect("Select Cluster", options=sorted(df['cluster'].unique()), default=sorted(df['cluster'].unique()))

filtered_df = df_clustered[(df_clustered['airline'].isin(selected_airline)) &
                           (df_clustered['airline_sentiment'].isin(selected_sentiment)) &
                           (df_clustered['cluster'].isin(selected_cluster))]

st.subheader("Clustered Tweet Scatter Plot")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x='pca_x', y='pca_y', hue='cluster', style='airline', ax=ax)
st.pyplot(fig)

st.subheader("Explore the Data")
st.dataframe(filtered_df[['text', 'airline', 'airline_sentiment', 'negativereason', 'cluster']])

st.markdown("---")
st.caption("Built by Shubhankar Rana 🚀")
