import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
import torch
import numpy as np
import re


# ENVIRONMENT SETUP

if 'HF_TOKEN' not in os.environ:
    os.environ['HF_TOKEN'] = "your_huggingface_token_here"  # <---------put your own HF token

st.set_page_config(layout="wide", page_title="Serial Killers RAG")


# SESSION STATE INITIALIZATION (Models + Cache)


# LLM Model (Qwen)
if "my_llm_model" not in st.session_state:
    st.session_state["my_llm_model"] = "Qwen/Qwen2.5-72B-Instruct"

def update_llm_model():
    st.session_state["client"] = InferenceClient(
        st.session_state["my_llm_model"],
        token=os.getenv("HF_TOKEN")
    )

if "client" not in st.session_state:
    update_llm_model()

# Embeddings Model 
if "embeddings_model" not in st.session_state:
    st.session_state["embeddings_model"] = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2"
    )

# Cross-Encoder Reranker 
if "reranker" not in st.session_state:
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state["reranker"] = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=my_device
    )


# SYSTEM INSTRUCTIONS 

my_system_instructions = (
    "You are a factual assistant specializing in crime statistics."
    "Use the structured data provided (name, proven_victims, country, text) to answer accurately. "
    "When discussing victim counts, cite the exact numbers from the 'proven_victims' field. "
    "Keep answers concise (150 words max). Start with 'Hello!'"
)
first_message = "Hello, how can I help you today?"


# QDRANT CONNECTION 

QDRANT_URL = "..."   # <---- put your Qdrant URL here !!!
API_KEY = "..."  # <----- put your API key here !!!

qdrant = QdrantClient(url=QDRANT_URL, api_key=API_KEY)

# HELPER FUNCTIONS 


COUNTRIES = [
    'Afghanistan', 'Argentina', 'Australia', 'Austria', 'Belgium', 'Brazil',
    'Canada', 'Chile', 'China', 'Colombia', 'Czech Republic', 'Czechoslovakia',
    'Ecuador', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Hungary',
    'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Italy', 'Japan',
    'Kazakhstan', 'Latvia', 'Mexico', 'Morocco', 'Netherlands', 'Norway',
    'Pakistan', 'Peru', 'Poland', 'Puerto Rico', 'Romania', 'Russia', 'Rwanda',
    'South Africa', 'South Korea', 'Soviet Union', 'Spain', 'Swaziland',
    'Sweden', 'Switzerland', 'Tajikistan', 'Thailand', 'Tunisia', 'Ukraine',
    'United Kingdom', 'United States', 'Uzbekistan', 'Venezuela', 'West Germany',
    'Yugoslavia', 'Zambia'
]
COUNTRIES.sort(key=len, reverse=True)

# EXACT minimal_clean 
def minimal_clean(text):
    """Only fixes encoding/spacing. Keeps grammar/punctuation for the LLM."""
    text = str(text).replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# EXACT parse_numeric_filters
def parse_numeric_filters(query: str):
    """Detects strict numeric conditions: Greater Than, Less Than, Equal."""
    q = query.lower().replace(',', '')
    
    # 1. Determine Target Field
    target_field = "proven_victims"
    possible_keywords = ["possible", "suspected", "estimated", "alleged", "potential"]
    if any(word in q for word in possible_keywords):
        target_field = "possible_victims"
    
    # 2. Strict Patterns
    patterns = [
        # GREATER THAN (>)
        (r"more than (\d+)", "gt"),
        (r"over (\d+)", "gt"),
        (r"above (\d+)", "gt"),
        
        # LESS THAN (<)
        (r"less than (\d+)", "lt"),
        (r"under (\d+)", "lt"),
        (r"below (\d+)", "lt"),
        
        # EQUAL (=)
        (r"exactly (\d+)", "eq"),
        (r"(\d+)\s+victims", "eq"),
        (r"(\d+)\s+possible", "eq"),
        (r"with (\d+)$", "eq"),
    ]
    
    for pattern, operator in patterns:
        match = re.search(pattern, q)
        if match:
            value = int(match.group(1))
            return (target_field, operator, value)
    
    return None

# EXACT extract_country
def extract_country(query: str):
    q = query.lower()
    for c in COUNTRIES:
        if c.lower() in q:
            return c
    return None

# EXACT build_qdrant_filter
def build_qdrant_filter(query: str):
    """Returns a Qdrant Filter object based on detected numeric/country filters."""
    numeric = parse_numeric_filters(query)
    country = extract_country(query)
    
    must_clauses = []
    
    # Country filter
    if country:
        must_clauses.append(
            FieldCondition(
                key="country",
                match=MatchValue(value=country)
            )
        )
    
    # Numeric victims filter
    if numeric:
        field_name, op, value = numeric
        
        if op == "gt":
            rng = Range(gt=value)
        elif op == "lt":
            rng = Range(lt=value)
        else:
            rng = Range(gte=value, lte=value)
        
        must_clauses.append(
            FieldCondition(
                key=field_name,
                range=rng
            )
        )
    
    if not must_clauses:
        return None
    
    return Filter(must=must_clauses)

# RAG PIPELINE 

def handle_query(user_prompt):
    """
    EXACT implementation matching notebook's evaluation loop.
    Returns: augmented_prompt, display_data
    """
    # --- 1. PREPROCESS QUERY  ---
    clean_q = minimal_clean(user_prompt)
    query_embedding = st.session_state.embeddings_model.encode([clean_q])[0].tolist()
    
    # --- 2. HYBRID SEARCH ---
    q_filter = build_qdrant_filter(user_prompt)
    
    raw_results = qdrant.query_points(
        collection_name="serial_killers",
        query=query_embedding,
        limit=50,  # Fetch 50 candidates for deduplication
        with_payload=True,
        query_filter=q_filter
    )
    
    points = raw_results.points
    texts = [p.payload.get("text", "") for p in points]
    
    top_context_points = []
    
    # --- 3. RERANKING & DEDUPLICATION ---
    if texts:
        # Score pairs
        pairs = [(user_prompt, t) for t in texts]
        scores = st.session_state["reranker"].predict(pairs)
        
        # Sort by Score (High -> Low)
        ranked_points = sorted(list(zip(points, scores)), key=lambda x: x[1], reverse=True)
        
        # --- DEDUPLICATION LOGIC  ---
        seen_names = set()
        nof_keep_sentences = 10
        
        for point, score in ranked_points:
            name = point.payload.get("name", "Unknown")
            
            # Skip duplicates
            if name in seen_names:
                continue
            
            seen_names.add(name)
            top_context_points.append(point)
            
            # Stop when we have enough unique killers
            if len(top_context_points) >= nof_keep_sentences:
                break
    
    # --- 4. BUILD STRUCTURED CONTEXT ---
    context_parts = []
    display_data = []  # For UI debugging
    
    for i, point in enumerate(top_context_points):
        p = point.payload
        
        # Handle Country List
        c_val = p.get("country", "Unknown")
        c_str = ", ".join(c_val) if isinstance(c_val, list) else str(c_val)
        
        # For display in UI
        display_data.append({
            "name": p.get("name"),
            "proven_victims": p.get("proven_victims"),
            "country": c_str,
            "text": p.get("text", ""),
            "rerank_score": 1 / (1 + np.exp(-point.score))  # Sigmoid normalization
        })
        
        # Context Block 
        block = (
            f"[SERIAL KILLER {i+1}]\n"
            f"Name: {p.get('name', 'Unknown')}\n"
            f"Country: {c_str}\n"
            f"Proven victims: {p.get('proven_victims', 'Unknown')}\n"
            f"Details: {p.get('text', '')[:500]}...\n"
        )
        context_parts.append(block)
    
    context_str = "\n\n".join(context_parts)
    
    # --- 5. BUILD PROMPT  ---
    if context_str:
        augmented_prompt = (
            f"STRUCTURED CRIME DATA:\n\n"
            f"{context_str}\n\n"
            f"QUESTION: {user_prompt}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer based ONLY on the structured data above\n"
            f"2. When mentioning victim counts, use the exact 'Proven victims' numbers\n"
            f"3. Only mention the killer's name if the user's question specifically asks for name or names\n"
            f"4. Only mention the Country if the user's question specifically asks for location, origin, or limits by geography. Otherwise, do not mention the country\n"
            f"5. Only mention the victim counts (proven or possible) if the user's question specifically asks for numbers, statistics, ranking (e.g. 'top', 'worst'), or severity. Otherwise, do not mention the victim counts.\n"
            f"6. If the user asks for a summary, overview, or explanation of patterns: Provide a synthesized paragraph explaining the commonalities, behaviors, or trends found in the context. Do NOT just list names.\n"
            f"7. If the user asks for a definition: Define the term clearly based on the context provided.\n"
            f"8. Be concise and factual"
        )
    else:
        augmented_prompt = user_prompt  # Fallback
    
    return augmented_prompt, display_data


# UI LAYOUT

column_1, column_2 = st.columns([1, 2])


# LEFT COLUMN - RAG Setup

with column_1:
    st.markdown("## ⚙️ RAG Configuration")
    
    with st.expander("ℹ️ Disclaimer", expanded=False):
        st.markdown("""
**Educational Prototype**

This application was developed for the course **Search Engine and Web Mining**  
in the **Master of Science (MS) in Data Science** program at  
**American College of Greece | Deree**.

👩‍💻 **Team**: Aimilia, Nancy, Kyriaki  
👨‍🏫 **Instructor**: Dr. Lazaros Polymenakos

**Purpose**: Demonstrates Retrieval-Augmented Generation (RAG) with LLMs

⚠️ **Limitations**:
- Output may contain inaccuracies or hallucinations
- Not for commercial or real-world deployment
- No personal data collected

💡 Use for **academic demonstration only**.
""")
    
    st.markdown("### 🔪 Model Configuration")
    st.info(f"**LLM**: {st.session_state['my_llm_model']}")
    st.info(f"**Embeddings**: sentence-transformers/all-mpnet-base-v2")
    st.info(f"**Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    st.markdown("### 📊 Pipeline Parameters")
    st.write("- **Initial Retrieval**: 50 candidates")
    st.write("- **Reranking**: Cross-encoder scoring")
    st.write("- **Deduplication**: Top 10 unique results")
    st.write("- **Chunking**: 250 words, 40-word overlap")


# CHAT HISTORY INITIALIZATION

if "my_chat_messages" not in st.session_state:
    st.session_state["my_chat_messages"] = [
        {"role": "system", "content": my_system_instructions}
    ]

user_avatar = st.selectbox(
    "Choose your avatar:",
    ["🕵️‍♀️", "👤", "👮‍♀️", "🗂️", "🩸"],
    key="user_avatar"
)

# RIGHT COLUMN - Chat Interface

with column_2:
    st.markdown("## 💬 Chat Interface")
    messages_container = st.container(height=500)

    # Display first message
    messages_container.chat_message("ai", avatar="🔪").markdown(first_message)

    # Display chat history
    for message in st.session_state["my_chat_messages"]:
        if message["role"] != "system":
            avatar = user_avatar if message["role"] == "user" else "🔪"
            messages_container.chat_message(message["role"], avatar=avatar).markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your questions here..."):
        # === EXECUTE RAG PIPELINE ===
        augmented_prompt, display_data = handle_query(prompt)

        # Display user message
        messages_container.chat_message("user", avatar=user_avatar).markdown(prompt)

        # === GENERATE MODEL RESPONSE ===
        with messages_container.chat_message("ai", avatar="🔪"):
            response_placeholder = st.empty()

            if augmented_prompt:
                # Build messages list 
                messages = [
                    {"role": "system", "content": my_system_instructions},
                    {"role": "user", "content": augmented_prompt}
                ]

                # Generation 
                completion = st.session_state["client"].chat_completion(
                    messages=messages,
                    max_tokens=300, 
                    temperature=0.7, 
                    stream=False,
                )

                message = completion.choices[0].message["content"]
                response_placeholder.markdown(message)

            else:
                message = "I could not find any relevant information in my database."
                response_placeholder.markdown(message)

        #  SAVE TO CHAT HISTORY 
        st.session_state["my_chat_messages"].append({"role": "user", "content": prompt})
        st.session_state["my_chat_messages"].append({"role": "assistant", "content": message})

        # Trim history (keep last 10 messages + system)
        if len(st.session_state["my_chat_messages"]) > 10:
            st.session_state["my_chat_messages"] = (
                st.session_state["my_chat_messages"][:1] +  # Keep system
                st.session_state["my_chat_messages"][3:]    # Keep recent
            )

        
        # DEBUG PANEL (Below Chat)
        
        debug_col1, debug_col2 = st.columns([1, 1])

        with debug_col1:
            st.markdown(f"### 🔍 Retrieved Context ({len(display_data)} Unique)")
            for i, data in enumerate(display_data):
                with st.expander(f"#{i+1}: {data['name']} | Score: {data['rerank_score']:.3f}"):
                    if data.get('proven_victims'):
                        st.write(f"**Proven Victims**: {data['proven_victims']}")
                    if data.get('country'):
                        st.write(f"**Country**: {data['country']}")
                    st.write("**Text Preview**:")
                    st.text(data['text'][:300] + "..." if len(data['text']) > 300 else data['text'])

        with debug_col2:
            st.markdown("### 🧠 Query Analysis")
            st.write(f"**Total Chunks Retrieved**: {len(display_data)}")
            
            detected_country = extract_country(prompt)
            numeric_filter = parse_numeric_filters(prompt)
            
            if detected_country or numeric_filter:
                st.markdown("**🎯 Detected Filters:**")
                if detected_country:
                    st.write(f"- Country: `{detected_country}`")
                if numeric_filter:
                    field, op, val = numeric_filter
                    st.write(f"- {field}: `{op}` {val}")
            else:
                st.write("No filters detected (semantic search only)")


# SIDEBAR - Controls

with st.sidebar:
    st.markdown("## 🧹 Controls")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state["my_chat_messages"] = [
            {"role": "system", "content": my_system_instructions}
        ]
        st.rerun()
