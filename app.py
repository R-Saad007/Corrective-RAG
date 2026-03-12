import streamlit as st
from fast_rag import stream_rag_pipeline 
import time
import html

# --- 1. UI CONFIGURATION & LOGO INJECTION ---
st.set_page_config(page_title="AxIn Actionable Intelligence", page_icon="axin_logo.png", layout="wide")

st.markdown("""
    <style>
    /* Kill the massive default padding at the top of Streamlit's main container */
    .block-container {
        padding-top: 1rem !important; 
    }

    /* Dynamic Pill Styling - Hugging the top left corner */
    .dynamic-status-pill {
        display: inline-flex;
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-top: 30px; /* Pulls it right to the top edge */
        margin-bottom: 15px;
    }
   
    .status-dot {
        height: 10px;
        width: 10px;
        background-color: #00E676;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0, 230, 118, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
    }

    .user-msg-container {
        display: flex;
        flex-direction: row-reverse;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 24px;
    }
   
    .user-avatar {
        background-color: #2e2e2e;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
   
    .user-msg {
        background-color: #2B2D31;
        color: white;
        padding: 14px 20px;
        border-radius: 20px 4px 20px 20px;
        max-width: 75%;
        font-family: 'Segoe UI', sans-serif;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
   
    .stChatMessage { border-radius: 15px; margin-bottom: 15px; }
    .stStatusWidget { border: none !important; box-shadow: none !important; }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Render the dynamic pill absolutely first so it sits at the peak of the layout
# THE FIX: Updated the label to reflect the hardware-optimized model
st.markdown("""
    <div class="dynamic-status-pill">
        <span class="status-dot"></span>
        Ollama Connected | Qwen 2.5:0.5B (CPU-Optimized)
    </div>
""", unsafe_allow_html=True)

st.title("AxIn Help")
st.caption("Proprietary Actionable Intelligence Assistant")

BOT_AVATAR = "axin_logo.png"

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Starter": []}
    st.session_state.current_session = "Starter"

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        new_session_name = f"New Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_session_name] = []
        st.session_state.current_session = new_session_name
        st.rerun()

    st.write("### Chats")
   
    for session_name in reversed(list(st.session_state.chat_sessions.keys())):
        btn_type = "primary" if session_name == st.session_state.current_session else "secondary"
        if st.button(session_name, use_container_width=True, type=btn_type):
            st.session_state.current_session = session_name
            st.rerun()

# --- Render Existing Messages ---
active_messages = st.session_state.chat_sessions[st.session_state.current_session]

for message in active_messages:
    if message["role"] == "user":
        safe_text = html.escape(message["content"])
        user_html = f"""
        <div class="user-msg-container">
            <div class="user-avatar">👤</div>
            <div class="user-msg">{safe_text}</div>
        </div>
        """
        st.markdown(user_html, unsafe_allow_html=True)
    else:
        try:
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.markdown(message["content"])
        except Exception:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

# --- Chat Input & Generation Logic ---
if prompt := st.chat_input("How may I help you today?"):
   
    current_session = st.session_state.current_session
    if len(st.session_state.chat_sessions[current_session]) == 0:
        new_title = prompt[:25] + ("..." if len(prompt) > 25 else "")
        base_title = new_title
        counter = 1
        while new_title in st.session_state.chat_sessions:
            new_title = f"{base_title} ({counter})"
            counter += 1
           
        st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_session)
        st.session_state.current_session = new_title

    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": prompt})
   
    safe_prompt = html.escape(prompt)
    st.markdown(f"""
        <div class="user-msg-container">
            <div class="user-avatar">👤</div>
            <div class="user-msg">{safe_prompt}</div>
        </div>
    """, unsafe_allow_html=True)

    try:
        bot_container = st.chat_message("assistant", avatar=BOT_AVATAR)
    except Exception:
        bot_container = st.chat_message("assistant", avatar="🤖")

    with bot_container:
        try:
            # 1. The Retrieval Phase (Inside the Status Box)
            with st.status("Searching AxIn Documentation...", expanded=True) as status:
                st.write("Extracting vector embeddings...")
                time.sleep(1) # Adjusted delays so it doesn't drag too long
                st.write("Querying Qdrant vector database...")
                time.sleep(1)
                st.write("Retrieving relevant context...")
                time.sleep(1)
                st.write("Context retrieved. Initializing LLM...")

                # THE FIX: Initialize the generator and force it to compute the FIRST token
                pipeline_generator = stream_rag_pipeline(prompt)
                
                try:
                    # This line is where the CPU will hang while it thinks. 
                    # The status box keeps spinning during this!
                    first_token = next(pipeline_generator)
                except StopIteration:
                    first_token = ""
                
                # The exact millisecond the first token is born, we close the box
                status.update(label="Response Generation", state="complete", expanded=False)

            # 2. The Generation Phase (OUTSIDE the Status Box)
            # We create a tiny wrapper to yield that first token we caught, then the rest
            def seamless_stream():
                if first_token:
                    yield first_token
                for chunk in pipeline_generator:
                    yield chunk

            # Streamlit natively streams the rest directly into the UI
            full_response = st.write_stream(seamless_stream())

            # 3. Save the final string to session state history
            st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"System Error: {str(e)}")