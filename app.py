# app.py

"""
Main application entry point - Simplified version
GAP Analysis System
"""

import streamlit as st
from datetime import datetime
import logging

# Configure page
st.set_page_config(
    page_title="GAP Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import authentication manager
from utils.auth import AuthManager

# Initialize authentication manager
auth_manager = AuthManager()

def show_login_page():
    """Display the login page"""
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # App title
        st.markdown("<h1 style='text-align: center;'>📊 GAP Analysis System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>Please login to continue</p>", unsafe_allow_html=True)
        
        # st.divider()
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            submit_button = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        # Handle login
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                with st.spinner("Authenticating..."):
                    success, user_info = auth_manager.authenticate(username, password)
                
                if success:
                    auth_manager.login(user_info)
                    st.success(f"Welcome, {user_info['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

def show_main_app():
    """Display the main application after login"""
    
    # Check authentication
    if not auth_manager.check_session():
        st.rerun()
    
    # Sidebar user info
    st.sidebar.markdown(f"### 👤 {auth_manager.get_user_display_name()}")
    st.sidebar.divider()
    
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
    # Main content
    st.title("Welcome to GAP Analysis System")
    
    st.info("""
    👈 **Select a page from the sidebar** to begin:
    - **Net GAP Analysis**: Supply vs Demand balance overview
    - More analysis tools coming soon...
    """)
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Today", datetime.now().strftime('%d %b %Y'))
    
    with col2:
        st.metric("User", auth_manager.get_user_display_name())
    
    with col3:
        st.metric("Status", "✅ Active")

def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Show appropriate page
    if st.session_state.authenticated:
        show_main_app()
    else:
        show_login_page()

if __name__ == "__main__":
    main()