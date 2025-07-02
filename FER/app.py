import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from torch.autograd import Variable
import transforms as transforms
from models import VGG  # Import the model class as required

# Define the model loading function
def load_model():
    net = VGG('VGG19')
    checkpoint = torch.load('emotion.t7', map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return net

# Define the test image transformation function
def transform_image(image):
    cut_size = 44
    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    return transform_test(image)

# Define class names for output labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Convert RGB to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Main function to process the uploaded image and make predictions
def process_image(uploaded_image):
    # Ensure the uploaded image is loaded correctly
    raw_img = Image.open(uploaded_image).convert('RGB')
    raw_img = np.array(raw_img)

    # Convert image to grayscale
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    # Stack grayscale to RGB (3 channels)
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

# Streamlit app layout
st.title("Facial Emotion Recognition")
st.write("Upload an image to predict the facial emotion!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    # Process the image and get results
    try:
        score, predicted, raw_img = process_image(uploaded_image)

        # Plot the results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Input Image
        axes[0].imshow(raw_img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Classification Bar Chart
        ind = 0.1 + 0.6 * np.arange(len(class_names))
        width = 0.4
        color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
        for i in range(len(class_names)):
            axes[1].bar(ind[i], score.data.numpy()[i], width, color=color_list[i])
        axes[1].set_title("Classification Results")
        axes[1].set_xlabel("Expression Category")
        axes[1].set_ylabel("Classification Score")
        axes[1].set_xticks(ind)
        axes[1].set_xticklabels(class_names, rotation=45)

        # Emoji Display
        emoji_img = io.imread(f'images/emojis/{class_names[int(predicted.cpu().numpy())]}.png')
        axes[2].imshow(emoji_img)
        axes[2].set_title('Predicted Emoji')
        axes[2].axis('off')

        # Show the result
        st.pyplot(fig)

        st.write(f"**Predicted Expression:** {class_names[int(predicted.cpu().numpy())]}")

    except Exception as e:
        st.error(f"Error: {str(e)}")