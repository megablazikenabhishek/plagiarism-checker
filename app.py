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
st.title('Plagiarism Detection on C++ Code')

st.write("Paste your C++ code below:")

# Input text area for user to paste code
user_code = st.text_area("Enter your C++ code here:", height=300)

# List files in the 'dataset' folder
dataset_folder = "dataset"
files = [f for f in os.listdir(dataset_folder) if f.endswith('.cpp')]

# Read content of the dataset files
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Check plagiarism when user submits code
if st.button("Check for Plagiarism"):
    if user_code.strip():
        # Preprocess user input
        user_code = preprocess_code(user_code)
        
        # Initialize variables to track the most similar file
        most_similar_file = None
        highest_similarity = 0
        similarity_scores = []

        # Iterate over all files in the dataset and calculate similarity
        for file in files:
            dataset_code = read_file(os.path.join(dataset_folder, file))
            
            # Calculate similarity with user-provided code
            similarity = calculate_similarity_tfidf(user_code, dataset_code)
            similarity_scores.append(similarity)

            # Update the most similar file and highest similarity score
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_file = file
        
        # Display similarity result
        st.write(f"Most similar file in the dataset: {most_similar_file}")
        st.write(f"Plagiarism Similarity Score: {highest_similarity:.2f}")
        
        # Show plagiarism warning if similarity is above a certain threshold
        if highest_similarity > 0.8:  # Adjust this threshold if necessary
            st.error("Warning: The provided code is very similar to an existing file, possible plagiarism detected!")
        else:
            st.success("No significant plagiarism detected.")
        
        # Visualization
        st.subheader("Similarity Scores Visualization")
        
        # Bar chart of similarity scores
        plt.figure(figsize=(12, 6))  # Increase the figure width
        plt.barh(range(len(similarity_scores)), similarity_scores, color='skyblue')
        plt.xlabel('Similarity Score')
        plt.title('Similarity Scores between User Code and Dataset Files')
        plt.axvline(x=0.8, color='red', linestyle='--', label='Plagiarism Threshold')
        plt.legend()
        plt.xlim(0, 1)
        
        # Clean y-ticks and limit the number displayed
        yticks = range(0, len(similarity_scores), max(1, len(similarity_scores) // 10))  # Show every 10th file
        plt.yticks(yticks, [f'File {i + 1}' for i in yticks])
        
        st.pyplot(plt)  # Display the plot
        
        # Heatmap of similarity scores
        st.subheader("Heatmap of Similarity Scores")
        similarity_matrix = np.array(similarity_scores).reshape(1, -1)  # Reshape for heatmap
        plt.figure(figsize=(12, 2))  # Increase figure width for heatmap
        sns.heatmap(similarity_matrix, annot=False, yticklabels=['User Code'], cmap='YlGnBu', cbar=True)
        plt.title('Heatmap of Similarity Scores')
        plt.xlabel('Dataset Files')
        plt.ylabel('User Code')
        st.pyplot(plt)  # Display the plot

        # Pie chart for plagiarism detection overview
        st.subheader("Plagiarism Detection Overview")
        above_threshold = sum(score > 0.8 for score in similarity_scores)
        below_threshold = len(similarity_scores) - above_threshold
        
        sizes = [above_threshold, below_threshold]
        labels = ['Plagiarism Detected', 'No Plagiarism']
        colors = ['lightcoral', 'lightgreen']
        explode = (0.1, 0)  # explode 1st slice

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Plagiarism Detection Overview')
        st.pyplot(plt)  # Display the plot
    else:
        st.warning("Please paste some code to check for plagiarism.")
