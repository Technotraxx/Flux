import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO

# Function to set API key as environment variable
def set_api_key(api_key):
    os.environ['FAL_KEY'] = api_key

# Streamlit UI
st.title("Image Generation with FAL API")

# Sidebar for parameter control
st.sidebar.title("Generation Parameters")

# API key input in sidebar
api_key = st.sidebar.text_input("Enter your FAL API Key:", type="password")

# Set the API key
if api_key:
    set_api_key(api_key)
    import fal_client  # Importing fal_client after setting the API key

    # Function to generate image using the API
    def generate_image(prompt, negative_prompt, image_size, num_inference_steps, guidance_scale, num_images, safety_tolerance):
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
        result = handler.get()
        return result

    # Prompt input in main area
    prompt = st.text_area("Enter your prompt:")
    
    # Parameters in sidebar
    negative_prompt = st.sidebar.text_area("Enter your negative prompt:", value="worst quality, low quality, bad quality, deformed hands, deformed limbs, ugly, eye bags, small eyes, wrinkles, dark skin, logo, watermark, text, red color cast, tongue")
    image_size = st.sidebar.selectbox("Select image size:", ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"])
    num_inference_steps = st.sidebar.slider("Number of inference steps:", min_value=1, max_value=50, value=40, step=1)
    guidance_scale = st.sidebar.slider("Guidance scale:", min_value=1.0, max_value=20.0, value=9.0, step=0.5)
    num_images = st.sidebar.number_input("Number of images to generate:", min_value=1, max_value=10, value=1)
    safety_tolerance = st.sidebar.selectbox("Safety tolerance level:", ["1", "2", "3", "4", "5", "6"], index=5)  # Default to "6"

    # Generate button
    if st.button("Generate Image"):
        if api_key and prompt:
            with st.spinner("Generating image..."):
                result = generate_image(prompt, negative_prompt, image_size, num_inference_steps, guidance_scale, num_images, safety_tolerance)
                st.success("Image generated successfully!")
                image_url = result['images'][0]['url']
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Generated Image", use_column_width=True)

                # Download button
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                st.download_button(
                    label="Download Image",
                    data=img_byte_arr,
                    file_name="generated_image.jpg",
                    mime="image/jpeg"
                )
        else:
            st.error("Please enter your API key and a prompt.")
else:
    st.sidebar.error("Please enter your API key.")
