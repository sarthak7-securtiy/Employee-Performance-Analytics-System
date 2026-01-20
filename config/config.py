import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'performance_analytics_secret_key_2026'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'employee_performance_new.db')
    ONEDRIVE_API_KEY = os.environ.get('ONEDRIVE_API_KEY') or 'your_onedrive_api_key_here'
    AZURE_VISION_KEY = os.environ.get('AZURE_VISION_KEY') or 'your_azure_vision_key_here'
    AZURE_VISION_ENDPOINT = os.environ.get('AZURE_VISION_ENDPOINT') or 'your_azure_vision_endpoint_here'
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)