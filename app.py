# app.py

"""
Main application entry point with login page
GAP Analysis System - iSCM Dashboard
"""

import streamlit as st
from datetime import datetime
import logging

# Configure page
st.set_page_config(
    page_title="iSCM GAP Analysis System",
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
        # App title and logo
        st.markdown("""
        <h1 style='text-align: center; color: #1E88E5;'>
            📊 iSCM GAP Analysis System
        </h1>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style='text-align: center; color: #666;'>
            Supply-Demand Intelligence Platform
        </p>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Login form
        with st.form("login_form"):
            st.subheader("🔐 Login")
            
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Contact admin if you forgot your username"
            )
            
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                help="Password is case-sensitive"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                submit_button = st.form_submit_button(
                    "Login",
                    type="primary",
                    use_container_width=True
                )
            
            with col_btn2:
                # Guest access (optional - remove if not needed)
                guest_button = st.form_submit_button(
                    "Guest Access",
                    use_container_width=True,
                    disabled=True,  # Enable if guest access is allowed
                    help="Guest access is currently disabled"
                )
        
        # Handle login submission
        if submit_button:
            if not username or not password:
                st.error("❌ Please enter both username and password")
            else:
                with st.spinner("Authenticating..."):
                    success, user_info = auth_manager.authenticate(username, password)
                
                if success:
                    # Set up session
                    auth_manager.login(user_info)
                    st.success(f"✅ Welcome back, {user_info['full_name']}!")
                    st.balloons()
                    
                    # Redirect to main page
                    st.rerun()
                else:
                    error_msg = user_info.get('error', 'Authentication failed')
                    st.error(f"❌ {error_msg}")
        
        # Footer information
        st.divider()
        
        # System information
        with st.expander("ℹ️ System Information", expanded=False):
            st.markdown("""
            ### Features:
            - 📊 **Net GAP Analysis**: Simple supply-demand balance overview
            - 📈 **Period GAP Analysis**: Time-based analysis with carry-forward
            - ⏰ **Timing GAP Analysis**: Supply-demand timing alignment
            - 📦 **PO Suggestions**: Automated purchase order recommendations
            - 🎯 **Allocation GAP**: Allocation-aware supply planning
            
            ### Quick Tips:
            - Use your company email username to login
            - Password is case-sensitive
            - Session expires after 8 hours of inactivity
            - Contact IT support for password reset
            
            ### Support:
            - 📧 Email: support@company.com
            - 📞 Phone: ext. 1234
            - 💬 Slack: #scm-support
            """)
        
        # Version and timestamp
        st.caption(f"""
        Version 1.0.0 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)

def show_main_app():
    """Display the main application after login"""
    
    # Check if user is still authenticated
    if not auth_manager.check_session():
        st.rerun()
    
    # Show user info in sidebar
    st.sidebar.markdown(f"### 👤 Welcome, {auth_manager.get_user_display_name()}")
    st.sidebar.markdown(f"**Role:** {st.session_state.get('user_role', 'User')}")
    st.sidebar.markdown(f"**Login time:** {st.session_state.get('login_time', datetime.now()).strftime('%H:%M')}")
    
    st.sidebar.divider()
    
    # Logout button
    if st.sidebar.button("🚪 Logout", type="secondary", use_container_width=True):
        auth_manager.logout()
        st.rerun()
    
    # Main content area
    st.title("🏠 GAP Analysis Dashboard")
    
    st.info("""
    👈 **Select a page from the sidebar** to begin your analysis:
    - **Net GAP**: Quick overview without time dimension
    - **Period GAP**: Analysis across time periods
    - **More coming soon...**
    """)
    
    # Quick stats (optional - can be removed or customized)
    st.subheader("📊 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Today's Date",
            value=datetime.now().strftime('%d %b %Y'),
            delta=datetime.now().strftime('%A')
        )
    
    with col2:
        st.metric(
            label="Analysis Period",
            value="Next 30 Days",
            delta="Default"
        )
    
    with col3:
        st.metric(
            label="Session Status",
            value="Active",
            delta="Secure"
        )
    
    with col4:
        time_remaining = 8 - (datetime.now() - st.session_state.get('login_time', datetime.now())).seconds // 3600
        st.metric(
            label="Session Time",
            value=f"{max(0, time_remaining)}h left",
            delta="Auto-renew on activity"
        )
    
    # Recent updates or announcements (optional)
    st.divider()
    
    with st.expander("📢 Recent Updates & Announcements", expanded=True):
        st.markdown("""
        ### 🆕 What's New (v1.0.0)
        - **Net GAP Analysis** page is now live!
        - Improved data loading performance with 5-minute cache
        - Enhanced tooltips for dates and quantities
        - Excel export with multiple sheets
        
        ### 🔔 Upcoming Features
        - Period GAP with carry-forward logic
        - Automated PO suggestion engine
        - Email notifications for critical shortages
        - Mobile-responsive design improvements
        
        ### 📅 Maintenance Schedule
        - Next scheduled maintenance: Sunday 2:00 AM - 4:00 AM
        """)
    
    # Help section
    with st.expander("❓ Need Help?", expanded=False):
        st.markdown("""
        ### Quick Start Guide:
        1. **Select a page** from the sidebar menu
        2. **Apply filters** to focus your analysis
        3. **Review KPIs** for quick insights
        4. **Explore visualizations** for patterns
        5. **Export data** for detailed analysis
        
        ### Common Questions:
        - **Q: How often is data updated?**
          - A: Data refreshes every 5 minutes from live database
        
        - **Q: Can I save my filter settings?**
          - A: Filters persist during your session
        
        - **Q: How do I export data?**
          - A: Click the Export button on any analysis page
        
        ### Training Resources:
        - [📖 User Manual](https://docs.company.com/gap-analysis)
        - [🎥 Video Tutorials](https://training.company.com/gap)
        - [💡 Best Practices Guide](https://wiki.company.com/gap-tips)
        """)

# Main application logic
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Check authentication and show appropriate page
    if st.session_state.authenticated:
        show_main_app()
    else:
        show_login_page()

if __name__ == "__main__":
    main()