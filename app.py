import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO
import time
import uuid
from datetime import datetime

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
        
        # Create a placeholder for the status message
        status_placeholder = st.empty()
        status_placeholder.info("Generating image...")

        # Submit the request
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
        
        # Wait for the result
        result = handler.get()
        
        # Calculate total time
        total_time = time.time() - start_time
        status_placeholder.success(f"Image generated successfully! (Total time: {total_time:.2f} seconds)")
        
        return result

    # Main area
    prompt = st.text_area("Enter your prompt:", help="Describe the image you want to generate")
    
    # Sidebar parameters
    with st.sidebar.expander("Advanced Settings", expanded=False):
        negative_prompt = st.text_area("Negative prompt:", value="worst quality, low quality, bad quality", help="NOT WORKING AT THE MOMENT! Describe what you don't want in the image")
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
                
                # Display seed information
                seed = result.get('seed', 'unknown')
                if seed != 'unknown':
                    st.info(f"Seed used for generation: {seed}")
                
                for idx, image_info in enumerate(result['images']):
                    image_url = image_info['url']
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))
                    
                    # Create columns for image and info
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Display the image at 50% of its original size
                        width = img.width // 2
                        st.image(img, caption=f"Generated Image {idx+1}", width=width)
                    
                    with col2:
                        st.write(f"Image {idx+1} Info:")
                        st.write(f"Content Type: {image_info['content_type']}")
                        if 'has_nsfw_concepts' in result:
                            st.write(f"NSFW Content: {'Yes' if result['has_nsfw_concepts'][idx] else 'No'}")
                    
                    # Generate filename
                    generation_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
                    filename = f"image_{seed}_{generation_time}_{unique_id}.jpg"
                    
                    # Add to history
                    st.session_state.history.append({
                        'prompt': prompt,
                        'image': img,
                        'seed': seed,
                        'filename': filename,
                        'generation_time': generation_time
                    })

                    # Download button
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    st.download_button(
                        label=f"Download Image {idx+1}",
                        data=img_byte_arr,
                        file_name=filename,
                        mime="image/jpeg"
                    )
                
                # Display the prompt used
                st.write(f"Prompt used: {result.get('prompt', prompt)}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter your API key and a prompt.")

else:
    st.sidebar.error("Please enter your API key.")

# History display
if st.session_state.history:
    st.header("Generation History")
    for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5 items
        with st.expander(f"Generation {len(st.session_state.history)-i}: {item['prompt'][:50]}..."):
            col1, col2 = st.columns([2, 3])
            with col1:
                # Display historical images at 50% of their original size
                width = item['image'].width // 2
                st.image(item['image'], caption="Generated Image", width=width)
            with col2:
                st.write("**Prompt:**", item['prompt'])
                st.write("**Seed:**", item.get('seed', 'Unknown'))
                if 'generation_time' in item:
                    st.write("**Generated at:**", item['generation_time'])
                else:
                    st.write("**Generated at:** Not available")
                if 'download' not in item:
                    # Create a download button for the image
                    img_byte_arr = BytesIO()
                    item['image'].save(img_byte_arr, format='JPEG')
                    item['download'] = img_byte_arr.getvalue()
                st.download_button(
                    label="Download Image",
                    data=item['download'],
                    file_name=item.get('filename', f"generated_image_{i}.jpg"),
                    mime="image/jpeg"
                )

# Clear history button
if st.session_state.history:
    if st.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()
