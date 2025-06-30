import streamlit as st
import os
import requests
from io import BytesIO
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import logging
import yaml
import textwrap
import threading
import time
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from torch.autograd import Variable
import FER.transforms as transforms
from FER.models import VGG  # Import the model class as required
import numpy as np

# Load configuration
def load_config():
    with open("LLM/yaml-editor-online.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

# Function for MRI report analysis
def analyze_mri(image_file):
    api_url = "http://44.213.223.150:5000/predict"
    headers = {"Content-Type": "application/octet-stream"}
    response = requests.post(api_url, headers=headers, data=image_file)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("prediction", "No prediction available")
    else:
        return "Error analyzing MRI"

# Function to create and load the vector database
def create_vector_db(config):
    try:
        instructor_embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
        loader = CSVLoader(file_path="LLM/LLM_data_dementia.csv", source_column="prompt")
        data = loader.load()
        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
        vectordb.save_local("faiss_index")
        logging.info("Vector database successfully created and saved to 'faiss_index'.")
    except Exception as e:
        logging.error("Error creating and saving vector database:", exc_info=e)

# Function to handle the chat logic
def get_qa_chain(query, config, chat):
    try:
        instructor_embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
        vectordb = FAISS.load_local("faiss_index", instructor_embeddings, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever(score_threshold=0.7)
        context = retriever.get_relevant_documents(query)

        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        If the answer is not found in the context, kindly state "I don't know."
        CONTEXT: {context}
        QUESTION: {query}"""
        
        prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template).format(context=context, query=query)
        response = chat(prompt=prompt)
        markdown_response = to_markdown(response)
        return markdown_response
    except Exception as e:
        logging.error("Error getting QA chain:", exc_info=e)
        return "Error during QA chain processing"

# Function to convert to markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Define the model loading function for FER
def load_model():
    net = VGG('VGG19')
    checkpoint = torch.load('FER/FER2013_VGG19/PrivateTest_model.t7', map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return net

# Define the test image transformation function for FER
def transform_image(image):
    cut_size = 44
    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    return transform_test(image)

# Define class names for FER
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Convert RGB to grayscale for FER
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Main function to process the uploaded image and make predictions for FER
def process_image(uploaded_image):
    raw_img = Image.open(uploaded_image).convert('RGB')
    raw_img = np.array(raw_img)

    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)

    inputs = transform_image(img)
    
    # Load the pre-trained model
    net = load_model()

    # Prepare inputs for the model
    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = Variable(inputs, volatile=True)  # Set volatile to True only if using older PyTorch versions; otherwise omit

    # Forward pass through the model
    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # average over crops

    # Get prediction and scores
    score = F.softmax(outputs_avg, dim=0)
    _, predicted = torch.max(outputs_avg.data, 0)
    
    return score, predicted, raw_img

# Multithreaded function for FER image analysis
def fer_analysis_thread(uploaded_image, results_container):
    score, predicted, raw_img = process_image(uploaded_image)
    results_container["score"] = score
    results_container["predicted"] = predicted
    results_container["image"] = raw_img

# Multithreaded function for MRI analysis
def mri_analysis_thread(uploaded_image, results_container):
    image_bytes = uploaded_image.read()
    results_container["image"] = image_bytes
    results_container["prediction"] = analyze_mri(image_bytes)

# Multithreaded function for chatbot interaction
def chatbot_thread(user_query, config, chat, responses_container):
    response = get_qa_chain(user_query, config, chat)
    responses_container.append((user_query, response))

# Multithreaded function for video analysis
def video_analysis_thread(uploaded_video, results_container):
    video_bytes = uploaded_video.read()
    time.sleep(3)  # Simulate processing time
    results_container["video"] = "Video processed successfully!"

# Streamlit interface
def main():
    st.title("AI Features Interface")

    # Load the config
    config = load_config()
    llm = GoogleGenerativeAI(google_api_key=config["google_api_key"], model=config.get("model", "gemini-1.5-flash"), temperature=0.7)
    chat = llm
    create_vector_db(config)

    # Initialize session state
    if "queries_and_answers" not in st.session_state:
        st.session_state.queries_and_answers = []
    if "mri_results" not in st.session_state:
        st.session_state.mri_results = {}
    if "video_results" not in st.session_state:
        st.session_state.video_results = {}
    if "fer_results" not in st.session_state:
        st.session_state.fer_results = {}

    # Display features in parallel
    tab1, tab2, tab3, tab4 = st.tabs(["MRI Report Analyzer", "Chatbot", "Video Upload", "Facial Emotion Recognition"])

    # MRI Report Analyzer
    with tab1:
        st.header("Upload MRI Image for Analysis")
        uploaded_image = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image:
            thread = threading.Thread(target=mri_analysis_thread, args=(uploaded_image, st.session_state.mri_results))
            thread.start()
            thread.join()
            st.image(st.session_state.mri_results.get("image"), caption="Uploaded MRI Image", use_column_width=True)
            st.write(f"Prediction: {st.session_state.mri_results.get('prediction', 'Analyzing...')}")

    # Chatbot
    with tab2:
        st.header("Ask the Chatbot")

        if st.session_state.queries_and_answers:
            for idx, (query, answer) in enumerate(st.session_state.queries_and_answers):
                st.subheader(f"Query {idx + 1}")
                st.write(f"**Question**: {query}")
                st.write(f"**Answer**: {answer}")

        user_query = st.text_input("Ask a new question:")
        if user_query:
            responses = []
            thread = threading.Thread(target=chatbot_thread, args=(user_query, config, chat, responses))
            thread.start()
            thread.join()
            st.session_state.queries_and_answers.append((user_query, responses[0][1]))
            st.markdown(responses[0][1])

    # Video Upload
    with tab3:
        st.header("Upload Video for Analysis")
        uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"])
        
        if uploaded_video:
            thread = threading.Thread(target=video_analysis_thread, args=(uploaded_video, st.session_state.video_results))
            thread.start()
            thread.join()
            st.video(uploaded_video, format="video/mp4")
            st.write(f"Video Analysis Result: {st.session_state.video_results.get('video', 'Processing...')}")

    # FER (Facial Emotion Recognition)
    with tab4:
        st.header("Upload Image for Facial Emotion Recognition")
        uploaded_image = st.file_uploader("Choose an image for FER", type=["jpg", "jpeg", "png"])

        if uploaded_image:
            results_container = {}
            fer_thread = threading.Thread(target=fer_analysis_thread, args=(uploaded_image, results_container))
            fer_thread.start()
            fer_thread.join()
            
            st.image(results_container["image"], caption="Uploaded Image", use_column_width=True)
            st.write(f"Prediction: {class_names[results_container['predicted']]}")
            st.write(f"Confidence Score: {results_container['score'][results_container['predicted']].item():.2f}")
            
if __name__ == "__main__":
    main()