import streamlit as st
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Ensure required NLTK resources are downloaded
nltk.download('punkt')  # This is necessary for tokenization

# Preprocessing function to clean the code
def preprocess_code(code):
    # Remove comments
    code = re.sub(r'//.*?(\n|$)', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove extra whitespace and newlines
    code = ' '.join(code.split())
    # Tokenize the code into words
    tokens = nltk.word_tokenize(code)
    return ' '.join(tokens)

# Function to calculate similarity between two code files using TF-IDF
def calculate_similarity_tfidf(code1, code2):
    # Preprocess the code
    code1 = preprocess_code(code1)
    code2 = preprocess_code(code2)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform both code files
    tfidf_matrix = vectorizer.fit_transform([code1, code2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0]  # Return the similarity score

# Streamlit UI
st.title('Text Plagiarism Detection')

st.write("Paste your first Text below:")
# Input text area for user to paste code
user_code_1 = st.text_area("Enter your first Text here:", height=300)

st.write("Paste your second Text below:")
user_code_2 = st.text_area("Enter your second Text here:", height=300)

# List files in the 'dataset' folder
dataset_folder = "dataset"
files = [f for f in os.listdir(dataset_folder) if f.endswith('.cpp')]

# Read content of the dataset files
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Check plagiarism when user submits code
if st.button("Check for Plagiarism"):
    if user_code_1.strip() and user_code_2.strip():
        # Preprocess user input
        user_code_1 = preprocess_code(user_code_1)
        user_code_2 = preprocess_code(user_code_2)
        
        # Initialize variables to track similarities
        similarity_scores = []
        highest_similarity_1 = 0
        highest_similarity_2 = 0
        most_similar_file_1 = None
        most_similar_file_2 = None

        # Iterate over all files in the dataset and calculate similarity
        for file in files:
            dataset_code = read_file(os.path.join(dataset_folder, file))
            
            # Calculate similarity with both user-provided codes
            similarity_1 = calculate_similarity_tfidf(user_code_1, dataset_code)
            similarity_2 = calculate_similarity_tfidf(user_code_2, dataset_code)

            similarity_scores.append((similarity_1, similarity_2))

            # Update the most similar file and highest similarity score for the first code
            if similarity_1 > highest_similarity_1:
                highest_similarity_1 = similarity_1
                most_similar_file_1 = file
            
            # Update the most similar file and highest similarity score for the second code
            if similarity_2 > highest_similarity_2:
                highest_similarity_2 = similarity_2
                most_similar_file_2 = file
        
        # Display similarity results
        st.write(f"Most similar file to Code 1: {most_similar_file_1}, Similarity Score: {highest_similarity_1:.2f}")
        st.write(f"Most similar file to Code 2: {most_similar_file_2}, Similarity Score: {highest_similarity_2:.2f}")
        
        # Show plagiarism warnings if similarity is above a certain threshold
        if highest_similarity_1 > 0.8:
            st.error("Warning: The first code is very similar to an existing file, possible plagiarism detected!")
        else:
            st.success("No significant plagiarism detected for the first code.")
        
        if highest_similarity_2 > 0.8:
            st.error("Warning: The second code is very similar to an existing file, possible plagiarism detected!")
        else:
            st.success("No significant plagiarism detected for the second code.")
        
        # Visualization
        st.subheader("Similarity Scores Visualization")
        
        # Bar chart of similarity scores
        similarity_matrix = np.array(similarity_scores)
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        
        # Create horizontal bar plots
        index = np.arange(len(files))
        plt.barh(index, similarity_matrix[:, 0], color='skyblue', label='Code 1 Similarity', height=bar_width)
        plt.barh(index + bar_width, similarity_matrix[:, 1], color='lightgreen', label='Code 2 Similarity', height=bar_width)
        plt.xlabel('Similarity Score')
        plt.title('Similarity Scores between User Codes and Dataset Files')
        plt.axvline(x=0.8, color='red', linestyle='--', label='Plagiarism Threshold')
        plt.legend()
        plt.xlim(0, 1)
        
        # Clean y-ticks and limit the number displayed
        plt.yticks(index + bar_width / 2, [f'File {i + 1}' for i in range(len(files))])
        
        st.pyplot(plt)  # Display the plot
        
        # Heatmap of similarity scores
        st.subheader("Heatmap of Similarity Scores")
        sns.heatmap(similarity_matrix.T, annot=False, yticklabels=['Code 1', 'Code 2'], cmap='YlGnBu', cbar=True)
        plt.title('Heatmap of Similarity Scores')
        plt.xlabel('Dataset Files')
        plt.ylabel('User Codes')
        st.pyplot(plt)  # Display the plot

        # Scatter plot for similarity scores
        st.subheader("Scatter Plot of Similarity Scores")
        plt.figure(figsize=(10, 6))
        plt.scatter(similarity_matrix[:, 0], similarity_matrix[:, 1], color='purple')
        plt.xlabel('Code 1 Similarity Score')
        plt.ylabel('Code 2 Similarity Score')
        plt.title('Scatter Plot of Similarity Scores between Code 1 and Code 2')
        plt.axhline(y=0.8, color='red', linestyle='--', label='Plagiarism Threshold for Code 2')
        plt.axvline(x=0.8, color='blue', linestyle='--', label='Plagiarism Threshold for Code 1')
        plt.legend()
        st.pyplot(plt)  # Display the plot

    else:
        st.warning("Please paste both codes to check for plagiarism.")
