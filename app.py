import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO
import time

# Set page config
st.set_page_config(page_title="AI Image Generator", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #00A9FF;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Function to set API key as environment variable
def set_api_key(api_key):
    os.environ['FAL_KEY'] = api_key

# Custom header
st.markdown("""
    <h1 style='text-align: center; color: #31333F;'>AI Image Generator</h1>
    <p style='text-align: center; color: #31333F;'>Create stunning images with AI</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Generation Parameters")

# API key input in sidebar
api_key = st.sidebar.text_input("Enter your FAL API Key:", type="password")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if api_key:
    set_api_key(api_key)
    import fal_client

    def generate_image(prompt, negative_prompt, image_size, num_inference_steps, guidance_scale, num_images, safety_tolerance):
        start_time = time.time()
        handler = fal_client.submit(
            "fal-ai/flux-pro",
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_size": image_size,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "safety_tolerance": safety_tolerance
            }
        )
        
        # Create a placeholder for the status message
        status_placeholder = st.empty()
        
        while True:
            elapsed_time = time.time() - start_time
            status_placeholder.info(f"Generating image... (Elapsed time: {elapsed_time:.2f} seconds)")
            
            try:
                result = handler.get(timeout=0.1)  # Try to get the result with a short timeout
                break  # If successful, break the loop
            except fal_client.RequestTimeoutError:
                time.sleep(0.5)  # Wait a bit before trying again
            except Exception as e:
                status_placeholder.error(f"An error occurred: {str(e)}")
                raise e
        
        total_time = time.time() - start_time
        status_placeholder.success(f"Image generated successfully! (Total time: {total_time:.2f} seconds)")
        return result

    # Main area
    prompt = st.text_area("Enter your prompt:", help="Describe the image you want to generate")
    
    # Sidebar parameters
    with st.sidebar.expander("Advanced Settings", expanded=False):
        negative_prompt = st.text_area("Negative prompt:", value="worst quality, low quality, bad quality", help="Describe what you don't want in the image")
        image_size = st.selectbox("Image size:", ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"], help="Choose the aspect ratio of the generated image")
        num_inference_steps = st.slider("Inference steps:", min_value=1, max_value=100, value=40, step=1, help="More steps generally result in better quality but take longer")
        guidance_scale = st.slider("Guidance scale:", min_value=1.0, max_value=10.0, value=9.0, step=0.5, help="How closely the image should follow the prompt. Higher values stick closer to the prompt")
        num_images = st.number_input("Number of images:", min_value=1, max_value=10, value=1, help="Number of images to generate in one go")
        safety_tolerance = st.selectbox("Safety tolerance:", ["1", "2", "3", "4", "5", "6"], index=5, help="6 is the most permissive, 1 is the most restrictive")

    # Generate button
    if st.button("Generate Image"):
        if api_key and prompt:
            try:
                result = generate_image(prompt, negative_prompt, image_size, num_inference_steps, guidance_scale, num_images, safety_tolerance)
                for idx, image_info in enumerate(result['images']):
                    image_url = image_info['url']
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption=f"Generated Image {idx+1}", use_column_width=True)
                    
                    # Add to history
                    st.session_state.history.append({
                        'prompt': prompt,
                        'image': img
                    })

                    # Download button
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    st.download_button(
                        label=f"Download Image {idx+1}",
                        data=img_byte_arr,
                        file_name=f"generated_image_{idx+1}.jpg",
                        mime="image/jpeg"
                    )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter your API key and a prompt.")

else:
    st.sidebar.error("Please enter your API key.")
