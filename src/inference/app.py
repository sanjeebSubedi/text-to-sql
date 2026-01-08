import streamlit as st
import requests
import json

# Configuration
st.set_page_config(
    page_title="Text-to-SQL | Phi-3 Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# Custom CSS for Dark Mode Look
st.markdown("""
<style>
    /* 1. FORCE DARK SIDEBAR BACKGROUND */
    [data-testid="stSidebar"] {
        background-color: #262730; /* Standard Streamlit Dark Gray */
    }

    /* 2. ADJUST CARDS FOR DARK MODE */
    .stCard {
        background-color: #1f1f1f; /* Darker gray for cards */
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }

    /* 3. METRIC COLORS */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #818cf8; /* Light Indigo for contrast */
    }
    
    /* 4. BUTTON STYLING */
    .stButton button {
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    
    /* 5. REMOVE TOP PADDING */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Data & State Management

SAMPLE_SCHEMAS = {
    "Users Database": """Table: users
Columns: id (INTEGER, PK), name (TEXT), email (TEXT), age (INTEGER), created_at (DATETIME)

Table: orders
Columns: id (INTEGER, PK), user_id (INTEGER), product (TEXT), amount (REAL), order_date (DATE)""",
    
    "Music Database": """Table: artists
Columns: id (INTEGER, PK), name (TEXT), genre (TEXT), country (TEXT)

Table: albums
Columns: id (INTEGER, PK), title (TEXT), artist_id (INTEGER), release_year (INTEGER), sales (INTEGER)

Table: songs
Columns: id (INTEGER, PK), title (TEXT), album_id (INTEGER), duration_seconds (INTEGER)""",
    
    "School Database": """Table: students
Columns: id (INTEGER, PK), name (TEXT), grade (INTEGER), gpa (REAL), enrollment_date (DATE)

Table: courses
Columns: id (INTEGER, PK), name (TEXT), department (TEXT), credits (INTEGER)

Table: enrollments
Columns: student_id (INTEGER), course_id (INTEGER), semester (TEXT), grade (TEXT)""",
}

SAMPLE_QUESTIONS = {
    "Users Database": [
        "How many users are there?",
        "Find all users older than 30",
        "What is the total amount of all orders?",
        "List users who have placed orders",
    ],
    "Music Database": [
        "How many albums were released after 2010?",
        "Find all artists from the USA",
        "Average song duration in seconds?",
        "List albums with > 1 million sales",
    ],
    "School Database": [
        "How many students have a GPA above 3.5?",
        "List all courses in Math department",
        "Students enrolled in > 3 courses?",
        "What is the average GPA by grade?",
    ],
}

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = None
if "input_question" not in st.session_state:
    st.session_state.input_question = ""

def set_question(q):
    st.session_state.input_question = q

with st.sidebar:
    st.title("Text-to-SQL")
    st.caption("Powered by Fine-Tuned Phi-3-Mini")
    
    st.divider()
    
    # 1. Database Selection
    st.subheader("1. Select Database")
    selected_db = st.selectbox(
        "Choose a schema context:",
        list(SAMPLE_SCHEMAS.keys()) + ["Custom Schema"],
        index=0
    )
    
    # 2. Schema Viewer
    if selected_db == "Custom Schema":
        schema_context = st.text_area("Define your schema:", height=200, placeholder="Table: ...\nColumns: ...")
    else:
        schema_context = SAMPLE_SCHEMAS[selected_db]
        with st.expander("📄 View Schema Definition", expanded=True):
            st.code(schema_context, language="sql")
            
    st.divider()
    
    # 3. System Status
    st.subheader("System Status")
    col_s1, col_s2 = st.columns(2)
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=1).json()
        status_color = "green"
        status_text = "Online"
        device = health.get("device", "Unknown")
    except:
        status_color = "red"
        status_text = "Offline"
        device = "N/A"

    with col_s1:
        st.markdown(f":{status_color}[● API {status_text}]")
    with col_s2:
        st.markdown(f"**Device:** `{device}`")

# Main Content

# Header
st.markdown("### Ask your Database")
st.markdown("Convert natural language questions into valid SQL queries.")

# Quick Select Chips
if selected_db != "Custom Schema":
    st.caption(f"Try a sample question for **{selected_db}**:")
    
    cols = st.columns(4)
    questions = SAMPLE_QUESTIONS.get(selected_db, [])
    
    for i, q in enumerate(questions):
        with cols[i % 4]:
            if st.button(q, key=f"btn_{selected_db}_{i}", help=q, use_container_width=True):
                set_question(q)

# Main Input Area
with st.container():
    question_text = st.text_area(
        "Enter your question:",
        value=st.session_state.input_question,
        height=100,
        placeholder="e.g., Show me all users who signed up last week...",
        key="main_input"
    )
    
    col_act1, col_act2 = st.columns([1, 4])
    with col_act1:
        generate_btn = st.button("Generate SQL", type="primary", use_container_width=True)
    with col_act2:
        if not question_text:
            st.warning("⚠️ Please enter a question to proceed.")


# Logic & Results
if generate_btn and question_text and schema_context:
    with st.spinner("Model is thinking..."):
        try:
            payload = {"schema": schema_context, "question": question_text}
            response = requests.post(f"{API_URL}/generate_sql", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.generated_sql = data
            else:
                st.error(f"API Error: {response.text}")
                
        except Exception as e:
            st.error(f"Connection Failed: {str(e)}")

# Display Results
if st.session_state.generated_sql:
    result = st.session_state.generated_sql
    
    st.divider()
    
    t1, t2 = st.tabs(["Generated SQL", "Execution Details"])
    
    with t1:
        st.subheader("Result")
        st.code(result.get('sql', '-- No SQL generated'), language="sql")
        
        c1, c2, c3 = st.columns([1, 1, 8])
        with c1: st.button("👍 Good", key="good")
        with c2: st.button("👎 Bad", key="bad")
    
    with t2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Inference Time", f"{result.get('execution_time_ms', 0):.0f} ms")
        m2.metric("Model", result.get('model', 'Phi-3'))
        m3.metric("Tokens", "N/A")
        
        st.json(result)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.8em;'>"
    "Fine-tuned on Spider Dataset"
    "</div>", 
    unsafe_allow_html=True
)
