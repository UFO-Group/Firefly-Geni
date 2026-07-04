from openai import OpenAI

# ====== 1. Configuration ======
BASE_URL = "https://URL"
API_KEY = "your-key"

# ====== 2. Model Constants ======
MODEL_PRO = "your-model"          # Logic / Complex Text
MODEL_IMAGE = "your-model"  # Image Processing (High Quality)
MODEL_FLASH = "your-model"      # Fast Vision / Speed

# ====== 3. Initialize Client (CRITICAL STEP) ======
try:
    #print(f"Connecting to LLM API at {BASE_URL}...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    #print("API Client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# ====== 4. Helper Function ======
def get_model_name(model_type="pro"):
    """
    Returns the model string based on input type.
    Options: 'pro', 'image', 'flash'
    """
    model_type = model_type.lower()
    selected_model = ""
    
    if "image" in model_type:
        selected_model = MODEL_IMAGE
    elif "flash" in model_type:
        selected_model = MODEL_FLASH
    else:
        # Default to Pro model
        selected_model = MODEL_PRO
        
    print(f"Model '{selected_model}' is currently initializing...")
    
    return selected_model