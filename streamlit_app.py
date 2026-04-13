"""
Streamlit Frontend for Dog Breed QA System

Interactive web interface for the Haystack RAG pipeline backend.
Supports Q&A queries, questionnaires, breed search, and evaluation metrics.
"""

import streamlit as st
import os
from datetime import datetime
from backend_api import QABackend, QAResponse

# Page configuration
st.set_page_config(
    page_title="🐕 Dog Breed QA System",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1em;
        }
        .response-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .retrieved-docs {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border: 1px solid #e0e0e0;
        }
        .status-good {
            color: #31a049;
            font-weight: bold;
        }
        .status-warning {
            color: #ff9800;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'backend' not in st.session_state:
    st.session_state.backend = None
    st.session_state.initialized = False
    st.session_state.init_status = "Not started"

if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Backend Settings")
    urls_dir = st.text_input("URLs Directory", value="data/urls", key="urls_dir")
    output_dir = st.text_input("Output Directory", value="data/qa_outputs", key="output_dir")
    
    use_openai = st.checkbox("🔑 Use OpenAI", value=bool(os.getenv("OPENAI_API_KEY")))
    use_hf = st.checkbox("🤗 Use HuggingFace", value=bool(os.getenv("HF_TOKEN")))
    
    url_limit = st.slider("Max URLs to load (0 = all)", 0, 500, 0, step=10)
    
    st.divider()
    st.subheader("System Status")
    
    # Initialize system
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Initialize System", use_container_width=True):
            with st.spinner("Initializing system..."):
                st.session_state.backend = QABackend(
                    urls_dir=urls_dir,
                    output_dir=output_dir,
                    use_openai=use_openai,
                    use_hf=use_hf
                )
                
                init_result = st.session_state.backend.initialize(limit=url_limit if url_limit > 0 else 0)
                
                if init_result["status"] == "success":
                    st.session_state.initialized = True
                    st.session_state.init_status = "✅ System Ready"
                    st.success(f"Indexed {init_result['urls_indexed']} URLs, {init_result['documents']} documents")
                else:
                    st.session_state.init_status = f"❌ {init_result['message']}"
                    st.error(init_result["message"])
    
    # Display status
    if st.session_state.initialized:
        st.markdown('<p class="status-good">✅ System Initialized</p>', unsafe_allow_html=True)
        status = st.session_state.backend.get_status()
        
        st.metric("Documents Indexed", status["documents_indexed"])
        
        # Evaluators status
        eval_status = st.session_state.backend.get_evaluators_status()
        if eval_status["evaluators_available"]:
            st.markdown('<p class="status-good">✅ Evaluators Available</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⚠️ Evaluators Not Available</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-warning">⚠️ Not Initialized</p>', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("Models")
    if st.session_state.backend and st.session_state.initialized:
        status = st.session_state.backend.get_status()
        st.write(f"**Embedding:** {status.get('models', {}).get('embedding', 'N/A').split('/')[-1]}")
        if status.get("models", {}).get("openai_enabled"):
            st.write("**Generator:** OpenAI GPT-4o-mini")
        elif status.get("models", {}).get("huggingface_enabled"):
            st.write("**Generator:** HuggingFace Mistral-7B")
        else:
            st.write("**Generator:** Context Only (no LLM)")


# Main Content
st.title("🐕 Dog Breed QA System")
st.markdown("*Powered by Haystack RAG Pipeline with Semantic Search*")

if not st.session_state.initialized:
    st.warning("⚠️ **System not initialized.** Please configure and click 'Initialize System' in the sidebar.")
    st.info("""
    **Getting Started:**
    1. Set your directories and API keys in the sidebar
    2. Click "Initialize System" to load documents
    3. Come back here to ask questions
    """)
else:
    # Create tabs for different modes
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🤔 Ask Question", "📋 Questionnaire", "🔍 Search Breed", "📊 Session History"]
    )
    
    # Tab 1: Ask a Question
    with tab1:
        st.header("Ask About Dog Breeds")
        st.write("Ask any question about dog breeds. The system will search relevant Wikipedia content and provide an answer.")
        
        question = st.text_area(
            "Your Question:",
            placeholder="e.g., Which dog breeds are best for apartment living?",
            height=100,
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            save_to_log = st.checkbox("Save to logs", value=True)
        with col2:
            submit = st.button("🔎 Get Answer", use_container_width=True)
        
        if submit and question:
            with st.spinner("Searching and generating answer..."):
                response = st.session_state.backend.answer_question(question, save_to_log=save_to_log)
            
            if response.is_confident:
                st.markdown("### Answer:")
                st.markdown(f'<div class="response-box">{response.answer}</div>', unsafe_allow_html=True)
                
                # Show retrieved documents
                if response.retrieved_docs:
                    with st.expander(f"📚 Retrieved Documents ({len(response.retrieved_docs)})"):
                        for i, doc in enumerate(response.retrieved_docs, 1):
                            st.markdown(f"**Document {i}:**")
                            st.write(doc["content"])
                            if doc["metadata"]:
                                st.caption(f"Metadata: {doc['metadata']}")
                
                # Add to history
                st.session_state.qa_history.append({
                    "type": "Question",
                    "query": question,
                    "answer": response.answer,
                    "timestamp": response.timestamp
                })
                
                if save_to_log:
                    st.success("✅ Saved to logs")
            else:
                st.error(response.answer)
    
    # Tab 2: Questionnaire
    with tab2:
        st.header("📋 Dog Breed Recommendation Questionnaire")
        st.write("Answer a few questions about your lifestyle to get personalized breed recommendations.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            living_space = st.selectbox(
                "Living Space",
                ["Apartment", "House with small yard", "House with large yard", "Farm/rural area"]
            )
            
            activity_level = st.selectbox(
                "Your Activity Level",
                ["Low (prefer quiet lifestyle)", "Moderate (some outdoor activities)", 
                 "High (active outdoors frequently)", "Very High (intense athletic activities)"]
            )
            
            experience = st.selectbox(
                "Dog Ownership Experience",
                ["First-time owner", "Moderate experience", "Experienced owner"]
            )
        
        with col2:
            family_with_kids = st.selectbox(
                "Family with Children",
                ["No children", "Young children (0-5)", "School-age children (6-12)", 
                 "Teenagers (13+)"]
            )
            
            grooming_commitment = st.selectbox(
                "Grooming Commitment",
                ["Minimal grooming", "Moderate grooming", "High grooming requirements OK"]
            )
            
            allergies = st.selectbox(
                "Allergies in Family",
                ["No allergies", "Someone has dog allergies", "Prefer hypoallergenic breed"]
            )
        
        if st.button("🎯 Get Recommendations", use_container_width=True):
            with st.spinner("Analyzing your preferences..."):
                questionnaire_data = {
                    "Living Space": living_space,
                    "Activity Level": activity_level,
                    "Experience": experience,
                    "Family": family_with_kids,
                    "Grooming": grooming_commitment,
                    "Allergies": allergies
                }
                
                response = st.session_state.backend.answer_questionnaire(questionnaire_data, save_to_log=True)
            
            if response.is_confident:
                st.markdown("### 🐕 Recommended Breeds:")
                st.markdown(f'<div class="response-box">{response.answer}</div>', unsafe_allow_html=True)
                
                st.session_state.qa_history.append({
                    "type": "Questionnaire",
                    "query": str(questionnaire_data),
                    "answer": response.answer,
                    "timestamp": response.timestamp
                })
                
                st.success("✅ Saved to logs")
            else:
                st.error(response.answer)
    
    # Tab 3: Search Breed
    with tab3:
        st.header("🔍 Search for Specific Breed")
        st.write("Get detailed information about a specific dog breed.")
        
        breed_name = st.text_input(
            "Breed Name:",
            placeholder="e.g., Golden Retriever, German Shepherd",
            key="breed_search"
        )
        
        if st.button("🔎 Search Breed", use_container_width=True):
            if breed_name:
                with st.spinner(f"Searching for {breed_name}..."):
                    response = st.session_state.backend.search_breed(breed_name, save_to_log=True)
                
                if response.is_confident:
                    st.markdown(f"### {breed_name} Information:")
                    st.markdown(f'<div class="response-box">{response.answer}</div>', unsafe_allow_html=True)
                    
                    st.session_state.qa_history.append({
                        "type": "Breed Search",
                        "query": breed_name,
                        "answer": response.answer,
                        "timestamp": response.timestamp
                    })
                    
                    st.success("✅ Saved to logs")
                else:
                    st.error(response.answer)
            else:
                st.warning("Please enter a breed name")
    
    # Tab 4: Session History
    with tab4:
        st.header("📊 Session History")
        
        if st.session_state.qa_history:
            st.info(f"Total interactions: {len(st.session_state.qa_history)}")
            
            for i, entry in enumerate(st.session_state.qa_history, 1):
                with st.expander(f"{i}. {entry['type']} - {entry['timestamp'][:10]}"):
                    st.write(f"**Type:** {entry['type']}")
                    st.write(f"**Query:** {entry['query']}")
                    st.write(f"**Answer:** {entry['answer'][:500]}...")
            
            # Option to export history
            history_json = __import__('json').dumps(st.session_state.qa_history, indent=2)
            st.download_button(
                label="📥 Download Session History (JSON)",
                data=history_json,
                file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("No interactions yet. Start asking questions!")


# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; margin-top: 20px;'>
        <p><strong>🐕 Dog Breed QA System</strong> | Powered by Haystack RAG + Streamlit</p>
        <p>Evaluators Available: ✅ Faithfulness, ✅ SAS, ✅ Retrieval MRR</p>
    </div>
""", unsafe_allow_html=True)
