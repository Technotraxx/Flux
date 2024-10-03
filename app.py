import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO
import time
import uuid
from datetime import datetime

# Set page config
st.set_page_config(page_title="FLUX AI Image Generator", layout="wide", initial_sidebar_state="expanded")

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
    <h1 style='text-align: center; color: #31333F;'>FLUX AI Image Generator</h1>
    <p style='text-align: center; color: #31333F;'>Create stunning images with the FLUX Model Series by Black Forest Labs</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Generation Parameters")

# API key input in sidebar
api_key = st.sidebar.text_input("Enter your FAL API Key:", type="password")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize session state for current generation
if 'current_generation' not in st.session_state:
    st.session_state.current_generation = []

# Import fal_client only if API key is provided
if api_key:
    set_api_key(api_key)
    try:
        import fal_client
    except ImportError:
        st.error("The 'fal_client' module is not installed. Please install it using `pip install fal-client`.")
        st.stop()

    # Update the generate_image function to handle both Text-to-Image and Image-to-Image
    def generate_image(
        model,
        prompt,
        image_size,
        num_inference_steps,
        guidance_scale,
        num_images,
        safety_tolerance,
        enable_safety_checker,
        seed=None,
        image_url=None,
        strength=None,
        loras=None
    ):
        start_time = time.time()
        
        # Create a placeholder for the status message
        status_placeholder = st.empty()
        status_placeholder.info(f"Generating image using {model}...")
    
        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "enable_safety_checker": enable_safety_checker
        }
    
        # Add seed to payload if provided
        if seed:
            payload["seed"] = int(seed)
        
        # Add image_url and strength if provided (for Image-to-Image)
        if image_url:
            payload["image_url"] = image_url
            if strength:
                payload["strength"] = float(strength)
        
        # Add LoRA paths if provided
        if loras:
            lora_list = []
            for lora_path in loras:
                lora_list.append({"path": lora_path, "scale": 1.0})  # Default scale can be modified as needed
            payload["loras"] = lora_list
    
        # Submit the request
        handler = fal_client.submit(model, payload)
        
        # Wait for the result
        result = handler.get()
        
        # Calculate total time
        total_time = time.time() - start_time
        status_placeholder.success(f"Image generated successfully using {model}! (Total time: {total_time:.2f} seconds)")
        
        return result

    # Main area
    # Mode selection: Text-to-Image or Image-to-Image
    generation_mode = st.radio(
        "Select Generation Mode:",
        ("Text-to-Image", "Image-to-Image"),
        horizontal=True
    )

    # Prompt input (used in both modes)
    prompt = st.text_area("Enter your prompt:", help="Describe the image you want to generate")

    # Conditional inputs based on generation mode
    if generation_mode == "Image-to-Image":
        # Image upload
        uploaded_image = st.file_uploader("Upload an image for modification:", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            try:
                image = Image.open(uploaded_image)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_byte_arr = buffered.getvalue()
                
                # Upload the image using fal_client to get a URL
                image_url = fal_client.upload_file(uploaded_image)
            except Exception as e:
                st.error(f"Error processing the uploaded image: {e}")
                image_url = None
        else:
            image_url = None

        # LoRA paths input
        lora_input = st.text_area(
            "Enter LoRA Paths (one per line):",
            help="Provide URLs or file paths to LoRA weights. Enter one path per line."
        )
        lora_paths = [line.strip() for line in lora_input.split('\n') if line.strip()]
    else:
        image_url = None
        strength = None
        lora_paths = None

    # Sidebar parameters
    with st.sidebar.expander("Advanced Settings", expanded=False):
        model_options = [
            "fal-ai/flux-pro",
            "fal-ai/flux/dev",
            "fal-ai/flux-realism"
        ]
        if generation_mode == "Image-to-Image":
            model_options.append("fal-ai/flux-general/image-to-image")
        
        model = st.selectbox(
            "Choose AI Model:",
            model_options,
            help="Select the AI model for image generation"
        )
        image_size = st.selectbox(
            "Image size:",
            ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"],
            help="Choose the aspect ratio of the generated image"
        )
        num_inference_steps = st.slider(
            "Inference steps:",
            min_value=1,
            max_value=50,
            value=40,
            step=1,
            help="More steps generally result in better quality but take longer"
        )
        guidance_scale = st.slider(
            "Guidance scale:",
            min_value=1.0,
            max_value=20.0,
            value=9.0,
            step=0.5,
            help="How closely the image should follow the prompt. Higher values stick closer to the prompt"
        )
        num_images = st.number_input(
            "Number of images:",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of images to generate in one go"
        )
        safety_tolerance = st.selectbox(
            "Safety tolerance:",
            ["1", "2", "3", "4", "5", "6"],
            index=5,
            help="6 is the most permissive, 1 is the most restrictive"
        )
        enable_safety_checker = st.checkbox(
            "Enable safety checker",
            value=False,
            help="If unchecked, the safety checker will be disabled"
        )

        # Additional inputs for Image-to-Image
        if generation_mode == "Image-to-Image":
            strength = st.slider(
                "Strength:",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                help="Strength to use for image modification. 1.0 completely remakes the image while 0.0 preserves the original."
            )
    
    # Generate button
    if st.button("Generate Image"):
        if api_key and prompt and (generation_mode == "Text-to-Image" or (generation_mode == "Image-to-Image" and image_url)):
            try:
                # Convert seed to integer if provided, otherwise pass None
                seed_input = st.sidebar.text_input("Seed (optional):", help="Enter an integer for reproducible generation. Leave empty for random results.")
                seed_value = int(seed_input) if seed_input and seed_input.lstrip("-").isdigit() else None

                # Call generate_image with appropriate parameters
                result = generate_image(
                    model=model,
                    prompt=prompt,
                    image_size=image_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images=num_images,
                    safety_tolerance=safety_tolerance,
                    enable_safety_checker=enable_safety_checker,
                    seed=seed_value,
                    image_url=image_url if generation_mode == "Image-to-Image" else None,
                    strength=strength if generation_mode == "Image-to-Image" else None,
                    loras=lora_paths if generation_mode == "Image-to-Image" else None
                )
                
                # Display seed information
                used_seed = result.get('seed')
                if used_seed is not None:
                    st.info(f"Seed used for generation: {used_seed}")
                else:
                    st.info("No seed information available from the API.")
                
                # Clear previous generation
                st.session_state.current_generation = []
                
                for idx, image_info in enumerate(result['images']):
                    image_url_generated = image_info['url']
                    response = requests.get(image_url_generated)
                    img = Image.open(BytesIO(response.content))
    
                    # Generate filename
                    generation_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
                    filename = f"image_{used_seed}_{generation_time}_{unique_id}.jpg"
                    
                    # Store image and info in session state
                    st.session_state.current_generation.append({
                        'image': img,
                        'prompt': result.get('prompt', prompt),
                        'content_type': image_info.get('content_type', 'image/jpeg'),
                        'has_nsfw_concepts': result.get('has_nsfw_concepts', [False])[idx],
                        'filename': filename,
                        'seed': used_seed,
                        'generation_time': generation_time,
                        'enable_safety_checker': enable_safety_checker,
                        'model': model
                    })
                    
                    # Add to history
                    st.session_state.history.append(st.session_state.current_generation[-1])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter your API key, prompt, and upload an image for Image-to-Image generation.")
    
    # Display current generation
    if st.session_state.current_generation:
        st.header("Current Generation")
        for idx, item in enumerate(st.session_state.current_generation):
            col1, col2, col3 = st.columns([6, 1, 3])
            
            with col1:
                # Download button
                img_byte_arr = BytesIO()
                item['image'].save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                st.download_button(
                    label=f"Download Image {idx+1}",
                    data=img_byte_arr,
                    file_name=item['filename'],
                    mime="image/jpeg"
                )
                
                # Display the image at 100% of its original size
                st.image(item['image'], caption=f"Generated Image {idx+1}", use_column_width=True)
    
            with col2:
               st.markdown("<p>", unsafe_allow_html=True)
                
            with col3:
                 # Display the prompt used
                st.subheader("**Prompt used:**")
                st.code(item['prompt'])
                st.divider()
                 # Display additional info
                st.subheader(f"**Image {idx+1} Info:**")
                st.write(f"**Seed:** {item['seed']}")
                st.write(f"Content Type: {item['content_type']}")
                st.write(f"NSFW Content: {'Yes' if item['has_nsfw_concepts'] else 'No'}")
    
    # History display
    if st.session_state.history:
        st.header("Generation History")
        for i, item in enumerate(reversed(st.session_state.history[-8:])):  # Show last 8 items
            with st.expander(f"Generation {len(st.session_state.history)-i}: {item['prompt'][:50]}..."):
                col1, col2 = st.columns([2, 3])
                with col1:
                    # Display historical images at 50% of their original size
                    width = item['image'].width // 2
                    st.image(item['image'], caption="Generated Image", width=width)
                with col2:
                    st.write("**Prompt:**", item['prompt'])
                    st.write("**Seed:**", item.get('seed', 'Unknown'))
                    st.write("**Model:**", item.get('model', 'Unknown'))
                    if 'generation_time' in item:
                        st.write("**Generated at:**", item['generation_time'])
                    else:
                        st.write("**Generated at:** Not available")
                    st.write("**Safety Checker:**", "Enabled" if item.get('enable_safety_checker', True) else "Disabled")
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
else:
    st.warning("Please enter your FAL API Key in the sidebar to start generating images.")
