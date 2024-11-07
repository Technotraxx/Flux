import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO
import time
import uuid
from datetime import datetime
import base64

# Set page configuration
st.set_page_config(
    page_title="FLUX AI Image Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
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

# Sidebar: Generation Parameters
st.sidebar.title("Generation Parameters")

# Select Generation Mode in the sidebar
generation_mode = st.sidebar.radio(
    "Select Generation Mode:",
    ("Text-to-Image", "Image-to-Image"),
    horizontal=True
)

# API key input in sidebar
api_key = st.sidebar.text_input("Enter your FAL API Key:", type="password")

# Initialize session state for history, current generation, and uploaded image
if 'history' not in st.session_state:
    st.session_state.history = []

if 'current_generation' not in st.session_state:
    st.session_state.current_generation = []

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
    st.session_state.image_data_uri = None
    st.session_state.image_size_info = None

# Add this mapping at the top with other constants
            ULTRA_SIZE_MAP = {
                "square_hd": "1:1",
                "square": "1:1",
                "portrait_4_3": "3:4",
                "portrait_16_9": "9:16",
                "landscape_4_3": "4:3",
                "landscape_16_9": "16:9"
            }

# Import fal_client only if API key is provided
if api_key:
    set_api_key(api_key)
    try:
        import fal_client
    except ImportError:
        st.error("The 'fal_client' module is not installed. Please install it using `pip install fal-client`.")
        st.stop()

    # Function to generate image
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
        image_base64=None,
        strength=None,
        lora_path=None,
        lora_scale=None
    ):
        start_time = time.time()
        
        # Placeholder for status messages
        status_placeholder = st.empty()
        status_placeholder.info(f"Generating image using {model}...")

        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "image_size": image_size,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker
        }

        # Add safety_tolerance only for Text-to-Image models
        if generation_mode == "Text-to-Image" and model not in ["fal-ai/flux-general/image-to-image"]:
            payload["safety_tolerance"] = safety_tolerance

        # Add seed to payload if provided
        if seed:
            payload["seed"] = int(seed)
        
        # Add image_base64 and strength if provided (for Image-to-Image)
        if image_base64:
            payload["image_url"] = image_base64  # Using image_url field to send Base64 data URI
            if strength is not None:
                payload["strength"] = float(strength)
        
        # Add LoRA if provided
        if lora_path and lora_scale:
            payload["loras"] = [
                {
                    "path": lora_path,
                    "scale": float(lora_scale)
                }
            ]
        
        # Conditionally add inference_steps and guidance_scale
        # For Text-to-Image models, these are always sent
        if generation_mode == "Text-to-Image":
            payload["num_inference_steps"] = num_inference_steps
            payload["guidance_scale"] = guidance_scale
        else:
            # For Image-to-Image, exclude these if using a specific model version
            if model != "fal-ai/flux-pro/v1.1":
                payload["num_inference_steps"] = num_inference_steps
                payload["guidance_scale"] = guidance_scale

        # Debug: Show payload (optional)
        # st.write("Payload:", payload)

        # Submit the request
        handler = fal_client.submit(model, payload)
        
        # Wait for the result
        result = handler.get()
        
        # Calculate total time
        total_time = time.time() - start_time
        status_placeholder.success(f"Image generated successfully using {model}! (Total time: {total_time:.2f} seconds)")
        
        return result

    # Main Area: Prompt Input and Image Upload
    # Prompt input (used in both modes)
    prompt = st.text_area("Enter your prompt:", help="Describe the image you want to generate")

    # Conditional inputs based on generation mode
    if generation_mode == "Image-to-Image":
        # Arrange upload, image size, strength slider in left column and preview in right column
        left_col, right_col = st.columns([1, 1.5])  # Adjust column widths as needed

        with left_col:
            # Image upload
            uploaded_image = st.file_uploader("Upload an image for modification:", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                try:
                    image = Image.open(uploaded_image).convert("RGB")  # Ensure image is in RGB format
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_bytes = buffered.getvalue()
                    
                    # Encode image to Base64
                    image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    image_data_uri = f"data:image/jpeg;base64,{image_base64}"
                    
                    # Get image size
                    width, height = image.size
                    image_size_info = f"Width: {width}px, Height: {height}px"
                    
                    # Store in session state
                    st.session_state.uploaded_image = image
                    st.session_state.image_data_uri = image_data_uri
                    st.session_state.image_size_info = image_size_info
                except Exception as e:
                    st.error(f"Error processing the uploaded image: {e}")
                    st.session_state.uploaded_image = None
                    st.session_state.image_data_uri = None
                    st.session_state.image_size_info = None
            else:
                # If no new image is uploaded, use the existing one from session_state
                if st.session_state.uploaded_image:
                    image = st.session_state.uploaded_image
                    image_data_uri = st.session_state.image_data_uri
                    image_size_info = st.session_state.image_size_info
                else:
                    image = None
                    image_data_uri = None
                    image_size_info = None

            # Display Image Size Info
            if image_size_info:
                st.write(f"**Image Size:** {image_size_info}")

            # Strength slider for Image-to-Image
            strength = st.slider(
                "Strength:",
                min_value=0.00,
                max_value=1.00,
                value=0.50,
                step=0.05,
                help="Strength to use for image modification. 1.0 completely remakes the image while 0.0 preserves the original."
            )

        with right_col:
            # Display uploaded image with reduced size
            if image:
                st.image(image, caption="Uploaded Image Preview", width=300)  # Adjust width as needed

    else:
        image = None
        image_data_uri = None
        image_size_info = None
        strength = None

    # Sidebar: LoRA Configuration
    with st.sidebar.expander("LoRA Configuration", expanded=True):
        if generation_mode == "Image-to-Image":
            # LoRA Path input with default value
            lora_path_input = st.text_input(
                "Enter LoRA Path:",
                value="https://storage.googleapis.com/fal-flux-lora/",
                help="Provide the URL or file path to the LoRA weights."
            )

            # LoRA Scale input
            lora_scale_input = st.number_input(
                "Enter LoRA Scale:",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Specify the scale for the LoRA weights."
            )
        else:
            # If not Image-to-Image, LoRA configuration is not needed
            lora_path_input = None
            lora_scale_input = None

    # Sidebar: Advanced Settings
    with st.sidebar.expander("Advanced Settings", expanded=False):
        # Predefined enum values for image_size
        image_size_options = ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"]

        if generation_mode == "Text-to-Image":
            # List of public models, including the new model pro1.1
            model_options = [
                "fal-ai/flux-pro/v1.1",  # New Model Integrated
                "fal-ai/flux-pro/v1.1-ultra",
                "fal-ai/flux/dev",
                "fal-ai/flux-realism",
                "fal-ai/flux-pro"
            ]
            model = st.selectbox(
                "Choose AI Model:",
                model_options,
                help="Select the AI model for image generation"
            )

            # Image size selection restricted to enum values
            image_size = st.selectbox(
                "Image Size:",
                image_size_options,
                index=image_size_options.index("landscape_4_3"),
                help="Choose the size of the generated image."
            )
        else:
            # Fixed model for Image-to-Image
            model = "fal-ai/flux-general/image-to-image"
            st.markdown(f"**Model:** {model}")

            # Predefined image sizes as per user instruction
            predefined_sizes = {
                "square_hd": {"width": 1024, "height": 1024},
                "square": {"width": 512, "height": 512},
                "portrait_4_3": {"width": 768, "height": 1024},
                "portrait_16_9": {"width": 512, "height": 1024},
                "landscape_4_3": {"width": 1024, "height": 768},
                "landscape_16_9": {"width": 1024, "height": 512}
            }
           
            # Image size selection
            size_option = st.selectbox(
                "Image Size Option:",
                ["Use Uploaded Image Size", "Select Predefined Size"],
                help="Choose whether to use the uploaded image's size or select from predefined sizes."
            )
            
            if size_option == "Use Uploaded Image Size":
                if image:
                    image_size = {"width": image.width, "height": image.height}
                else:
                    st.warning("Please upload an image to use its size.")
                    image_size = {"width": 512, "height": 512}  # Default fallback
            else:
                # Use the predefined_sizes provided by the user
                image_size_option = st.selectbox(
                    "Select Image Size:",
                    list(predefined_sizes.keys()),
                    help="Choose the aspect ratio of the generated image."
                )
                image_size = predefined_sizes.get(image_size_option, {"width": 512, "height": 512})

        # Determine maximum number of images based on the selected model
        if generation_mode == "Text-to-Image":
            # Pro Models allow only 1 image; Dev Model allows up to 4 images
            if model in ["fal-ai/flux-pro/v1.1", "fal-ai/flux-pro", "fal-ai/flux-realism", "fal-ai/flux-pro/v1.1-ultra"]:
                max_num_images = 1
            elif model == "fal-ai/flux/dev":
                max_num_images = 4
            else:
                max_num_images = 1  # Default to 1 if model is unrecognized
        else:
            # Image-to-Image mode: Allow up to 4 images
            max_num_images = 4  # Assuming all Image-to-Image models support up to 4 images

        # Number of images input
        if generation_mode in ["Text-to-Image", "Image-to-Image"]:
            num_images = st.number_input(
                "Number of Images:",
                min_value=1,
                max_value=max_num_images,
                value=2 if generation_mode == "Image-to-Image" else 1,
                step=1,
                help=f"Number of images to generate in one go (up to {max_num_images} for selected model)."
            )
        else:
            # Fixed to 1 for Pro models in Image-to-Image
            st.write("**Number of Images:** 1 (fixed)")
            num_images = 1

        num_inference_steps = st.slider(
            "Inference Steps:",
            min_value=1,
            max_value=50,
            value=32,
            step=1,
            help="More steps generally result in better quality but take longer."
        )
        guidance_scale = st.slider(
            "Guidance Scale:",
            min_value=1.0,
            max_value=20.0,
            value=1.5,
            step=0.1,
            help="How closely the image should follow the prompt. Higher values stick closer to the prompt."
        )
        
        # Safety tolerance only for Text-to-Image
        if generation_mode == "Text-to-Image":
            safety_tolerance = st.selectbox(
                "Safety Tolerance:",
                ["1", "2", "3", "4", "5", "6"],
                index=5,
                help="6 is the most permissive, 1 is the most restrictive."
            )
        else:
            safety_tolerance = None  # Not applicable

        enable_safety_checker = st.checkbox(
            "Enable Safety Checker",
            value=True,
            help="If unchecked, the safety checker will be disabled."
        )

        # Seed input (common for both modes)
        seed_input = st.text_input(
            "Seed (optional):",
            help="Enter an integer for reproducible generation. Leave empty for random results."
        )

    # Generate button
    if st.button("Generate Image"):
        # Validation
        if not api_key:
            st.error("Please enter your API key.")
        elif not prompt:
            st.error("Please enter a prompt.")
        elif generation_mode == "Image-to-Image" and not st.session_state.uploaded_image and size_option == "Use Uploaded Image Size":
            st.error("Please upload an image for Image-to-Image generation.")
        elif generation_mode == "Image-to-Image" and (not lora_path_input or not lora_scale_input):
            st.error("Please provide both LoRA path and LoRA scale for Image-to-Image generation.")
        else:
            # In your generate button click handler, where you prepare the payload:
            try:
                # Convert seed to integer if provided, otherwise pass None
                seed_value = None
                if seed_input:
                    try:
                        seed_value = int(seed_input)
                    except ValueError:
                        st.error("Seed must be an integer.")
                        st.stop()
            
                # For Text-to-Image, handle the ultra model differently
                if generation_mode == "Text-to-Image" and model == "fal-ai/flux-pro/v1.1-ultra":
                    # Convert the selected image_size to the ultra model format
                    aspect_ratio = ULTRA_SIZE_MAP[image_size]
                    result = generate_image(
                        model=model,
                        prompt=prompt,
                        image_size={"aspect_ratio": aspect_ratio},
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images=num_images,
                        safety_tolerance=safety_tolerance,
                        enable_safety_checker=enable_safety_checker,
                        seed=seed_value
                    )
                else:
                    # Original code for other models
                    result = generate_image(
                        model=model,
                        prompt=prompt,
                        image_size=image_size,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images=num_images,
                        safety_tolerance=safety_tolerance if generation_mode == "Text-to-Image" else None,
                        enable_safety_checker=enable_safety_checker,
                        seed=seed_value,
                        image_base64=st.session_state.image_data_uri if generation_mode == "Image-to-Image" else None,
                        strength=strength if generation_mode == "Image-to-Image" else None,
                        lora_path=lora_path_input if generation_mode == "Image-to-Image" else None,
                        lora_scale=lora_scale_input if generation_mode == "Image-to-Image" else None
                    )
                            
                # Display seed information
                used_seed = result.get('seed')
                if used_seed is not None:
                    st.info(f"Seed used for generation: {used_seed}")
                else:
                    st.info("No seed information available from the API.")
                
                # Clear previous generation
                st.session_state.current_generation = []
                
                for idx, image_info in enumerate(result.get('images', [])):
                    image_url_generated = image_info.get('url')
                    if not image_url_generated:
                        st.error("Received an image without a URL from the API.")
                        continue

                    response = requests.get(image_url_generated)
                    if response.status_code != 200:
                        st.error(f"Failed to fetch image from URL: {image_url_generated}")
                        continue

                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    
                    # Get image size
                    width, height = img.size
                    image_size_generated = f"Width: {width}px, Height: {height}px"

                    # Generate filename
                    generation_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
                    filename = f"image_{used_seed}_{generation_time}_{unique_id}.jpg"
                    
                    # Validate content_type
                    content_type = image_info.get('content_type', "image/jpeg")
                    if not isinstance(content_type, str):
                        content_type = "image/jpeg"

                    # Store image and info in session state
                    st.session_state.current_generation.append({
                        'image': img,
                        'prompt': result.get('prompt', prompt),
                        'content_type': content_type,
                        'has_nsfw_concepts': result.get('has_nsfw_concepts', [False])[idx] if 'has_nsfw_concepts' in result else False,
                        'filename': filename,
                        'seed': used_seed,
                        'generation_time': generation_time,
                        'enable_safety_checker': enable_safety_checker,
                        'model': model,
                        'image_size': image_size_generated  # Added image size
                    })
                    
                    # Add to history
                    st.session_state.history.append(st.session_state.current_generation[-1])
                
            except fal_client.FalClientError as e:
                if e.status_code == 422 and "not public" in str(e):
                    st.error(f"The selected model '{model}' is not accessible. Please choose a different model or contact support.")
                else:
                    st.error(f"An unexpected error occurred: {e}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display current generation
    if st.session_state.current_generation:
        st.header("Current Generation")
        for idx, item in enumerate(st.session_state.current_generation):
            col1, col2, col3 = st.columns([6, 1, 3])
            
            with col1:
                # Download button
                img_byte_arr = BytesIO()
                item['image'].save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                st.download_button(
                    label=f"Download Image {idx+1}",
                    data=img_bytes,
                    file_name=item['filename'],
                    mime="image/jpeg"
                )
                
                # Display the image at 100% of its original size
                st.image(item['image'], caption=f"Generated Image {idx+1}", use_column_width=True)

            with col2:
                st.markdown("<p></p>", unsafe_allow_html=True)
                
            with col3:
                 # Display the prompt used
                st.subheader("**Prompt Used:**")
                st.code(item['prompt'])
                st.divider()
                 # Display additional info
                st.subheader(f"**Image {idx+1} Info:**")
                st.write(f"**Seed:** {item['seed']}")
                st.write(f"**Content Type:** {item['content_type']}")
                st.write(f"**Image Size:** {item.get('image_size', 'N/A')}")
                st.write(f"**NSFW Content:** {'Yes' if item['has_nsfw_concepts'] else 'No'}")

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
                    st.write("**Image Size:**", item.get('image_size', 'N/A'))  # Added image size
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
