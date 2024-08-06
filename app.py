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

# API key input
api_key = st.text_input("Enter your FAL API Key:", type="password")

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

    # Prompt input
    prompt = st.text_area("Enter your prompt:")
    negative_prompt = st.text_area("Enter your negative prompt:", value="worst quality, low quality, bad quality, deformed hands, deformed limbs, ugly, eye bags, small eyes, wrinkles, dark skin, logo, watermark, text, red color cast, tongue")

    # Image generation parameters
    image_size = st.selectbox("Select image size:", ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"])
    num_inference_steps = st.number_input("Number of inference steps:", min_value=1, max_value=100, value=40)
    guidance_scale = st.number_input("Guidance scale:", min_value=1.0, max_value=10.0, value=9.0)
    num_images = st.number_input("Number of images to generate:", min_value=1, max_value=10, value=1)
    safety_tolerance = st.selectbox("Safety tolerance level:", ["1", "2", "3", "4", "5", "6"])

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
    st.error("Please enter your API key.")
