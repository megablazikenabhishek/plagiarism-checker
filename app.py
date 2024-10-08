import streamlit as st
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure required NLTK resources are downloaded
nltk.download('all')

# Preprocessing function to clean the text
def preprocess_text(text):
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

# Function to calculate similarity between two texts using TF-IDF
def calculate_similarity_tfidf(text1, text2):
    # Preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform both texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0]  # Return the similarity score

# Streamlit UI
st.title('Text Comparison for Plagiarism Detection')

st.write("Paste the first text below:")
text1 = st.text_area("Enter first text here:", height=300)

st.write("Paste the second text below:")
text2 = st.text_area("Enter second text here:", height=300)

# Check plagiarism when user submits code
if st.button("Check for Plagiarism"):
    if text1.strip() and text2.strip():
        # Calculate similarity
        similarity = calculate_similarity_tfidf(text1, text2)
        
        # Display similarity result
        st.write(f"Plagiarism Similarity Score: {similarity:.2f}")
        
        # Show plagiarism warning if similarity is above a certain threshold
        if similarity > 0.8:  # Adjust this threshold if necessary
            st.error("Warning: The provided texts are very similar, possible plagiarism detected!")
        else:
            st.success("No significant plagiarism detected.")
        
        # Visualization
        st.subheader("Similarity Score Visualization")
        
        # Bar plot for similarity score
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=["Text 1 vs Text 2"], y=[similarity], palette="Blues", ax=ax)
        ax.axhline(y=0.8, color='r', linestyle='--', label='Plagiarism Threshold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Similarity Score')
        ax.set_title('Similarity Score between Texts')
        ax.legend()
        st.pyplot(fig)

        # Scatter Plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.scatter([1, 2], [similarity, similarity], color='blue')
        ax2.axhline(y=0.8, color='r', linestyle='--', label='Plagiarism Threshold')
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['Text 1', 'Text 2'])
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Scatter Plot of Similarity Scores')
        ax2.legend()
        st.pyplot(fig2)

        # Histogram
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.histplot([similarity], bins=10, color='blue', kde=True, ax=ax3)
        ax3.axvline(x=0.8, color='r', linestyle='--', label='Plagiarism Threshold')
        ax3.set_xlim(0, 1)
        ax3.set_xlabel('Similarity Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Histogram of Similarity Scores')
        ax3.legend()
        # st.pyplot(fig3)

    else:
        st.warning("Please paste both texts to check for plagiarism.")
