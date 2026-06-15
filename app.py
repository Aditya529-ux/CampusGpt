import streamlit as st
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
# -------------------------------
# ChromaDB Setup
# -------------------------------
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="campusgpt"
)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="CampusGPT",
    page_icon="🎓",
)

st.title("🎓 CampusGPT")
st.caption(
    "AI-Powered College Assistant using RAG + Groq"
)
# -------------------------------
# Chat History Initialization
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# -------------------------------
# Load Embedding Model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_model()

# -------------------------------
# Load Gemini Model
# -------------------------------
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------
# PDF Upload
# -------------------------------
# -------------------------------
# PDF Upload + Sidebar Controls
# -------------------------------
with st.sidebar:

    st.title("🎓 CampusGPT")

    st.markdown("---")

    # PDF Upload
    uploaded_files = st.file_uploader(
        "📄 Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")

    # Clear Chat Button
    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    # Reset Knowledge Base Button
    if st.button("♻️ Reset Knowledge Base"):

        try:

            client.delete_collection(
                name="campusgpt"
            )

            collection = client.get_or_create_collection(
                name="campusgpt"
            )

            st.session_state.processed_files = []

            st.success(
                "✅ Knowledge base cleared!"
            )

        except:

            st.warning(
                "⚠️ Nothing to clear."
            )

    st.markdown("---")

    # PDF Statistics Section
    st.subheader("📊 PDF Statistics")

    if uploaded_files:

        st.write(
            f"📚 PDFs Uploaded: {len(uploaded_files)}"
        )

    else:

        st.caption(
            "No PDFs uploaded yet."
        )
# -------------------------------
# Process PDFs
# -------------------------------
if uploaded_files:
    # Get list of current file names
    current_files = [f.name for f in uploaded_files]
    
    # Check if new files were uploaded
    if current_files != st.session_state.processed_files:
        try:
            client.delete_collection(
                name="campusgpt"
            )
        except:
            pass

        collection = client.get_or_create_collection(
            name="campusgpt"
        )

        st.success(
            f"{len(uploaded_files)} PDF(s) uploaded successfully!"
        )

        for pdf in uploaded_files:

            st.subheader(f"📄 {pdf.name}")

            reader = PdfReader(pdf)

            text = ""

            # Extract text
            for page in reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted

            # Statistics
            st.write(
                f"📊 Extracted Characters: {len(text):,}"
            )

            st.write(
                f"📝 Approximate Words: {len(text.split()):,}"
            )

            # Skip empty PDFs
            if len(text.strip()) == 0:

                st.warning(
                    "⚠️ No text found in this PDF."
                )

                continue

            # -------------------------------
            # Chunking
            # -------------------------------
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_text(
                text
            )

            st.write(
                f"📦 Total Chunks Created: {len(chunks)}"
            )

            # -------------------------------
            # Generate Embeddings
            # -------------------------------
            embeddings = model.encode(
                chunks
            )

            # -------------------------------
            # Save to ChromaDB
            # -------------------------------
            for i, chunk in enumerate(chunks):

                chunk_id = (
                    f"{pdf.name}_{i}"
                )

                try:

                   collection.add(
                        documents=[chunk],
                        embeddings=[embeddings[i].tolist()],
                        ids=[chunk_id],
                        metadatas=[{
                            "source": pdf.name,
                            "chunk": i + 1
                        }]
                    )

                except Exception:
                    pass

            st.success(
                "✅ Stored in ChromaDB"
            )

            # -------------------------------
            # Preview First 3 Chunks
            # -------------------------------
            st.subheader(
                "📚 Chunk Preview"
            )

            for i, chunk in enumerate(
                chunks[:3]
            ):

                st.text_area(
                    f"Chunk {i+1}",
                    chunk,
                    height=150,
                    key=f"{pdf.name}_{i}"
                )
        
        # Update processed files list
        st.session_state.processed_files = current_files

# -------------------------------
# Ask Questions
# -------------------------------
if uploaded_files:

    st.divider()

    st.header("💬 Ask Questions")

    # Display previous chat history
    if st.session_state.messages:

        st.subheader("📝 Chat History")

        st.caption(
            f"🧠 Memory Active: "
            f"{min(len(st.session_state.messages), 6)} messages"
        )

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

    # Chat input ALWAYS visible
    query = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if query:

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        # Generate query embedding
        query_embedding = model.encode(query)

        # Get fresh collection reference
        collection = client.get_or_create_collection(
            name="campusgpt"
        )

        # Retrieve relevant chunks
        results = collection.query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=3,
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

        # Check if results found
        if not results["documents"][0]:
            context = "No documents found in the knowledge base."
        else:
            context = "\n\n".join(
                results["documents"][0]
            )

        # Gemini prompt
        # Build previous conversation context
        conversation_history = ""

        for message in st.session_state.messages[-6:]:
            conversation_history += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        prompt = f"""
You are CampusGPT, an AI assistant for students.

Use ONLY the information provided below to answer the question.

Use previous conversation history to understand follow-up questions.

If the answer is not present in the context, say:
"I could not find the answer in the uploaded documents."

Previous Conversation:
{conversation_history}

Document Context:
{context}

Current Question:
{query}

Answer:
"""

        # Generate response with spinner
        try:
            with st.spinner("🤖 Thinking..."):
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are CampusGPT, an AI assistant for students. "
                                "Answer ONLY using the provided context. "
                                "If the answer is not present, say "
                                "'I could not find the answer in the uploaded documents.'"
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2
                )

            answer = response.choices[0].message.content

        except Exception as e:
            st.error(
                f"Actual Groq Error:\n\n{str(e)}"
            )

            answer = (
                f"Error occurred:\n\n{str(e)}"
            )

        # Save assistant message (ALWAYS runs)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        if results["documents"][0]:
            st.success(
                "✅ Top Relevant Chunks Found!"
            )
            st.subheader("📚 Sources Used")

            for i in range(
                len(results["documents"][0])
            ):

                metadata = results["metadatas"][0][i]

                distance = results["distances"][0][i]

                st.info(
                    f"""
📄 PDF: {metadata['source']}

📦 Chunk: {metadata['chunk']}

📏 Distance Score: {distance:.4f}
"""
                )

            # Show retrieved chunks
            st.subheader(
                "📚 Most Relevant Chunks"
            )

            for i, chunk in enumerate(
                results["documents"][0]
            ):

                metadata = results["metadatas"][0][i]

                st.expander(
                    f"📄 {metadata['source']} | Chunk {metadata['chunk']}"
                ).write(chunk)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="CampusGPT",
    page_icon="🎓",
)

st.title("🎓 CampusGPT")
st.write("AI-Powered College Assistant")

uploaded_files = st.file_uploader(
    "Upload your study notes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")

    for pdf in uploaded_files:

        st.subheader(f"📄 {pdf.name}")

        reader = PdfReader(pdf)

        text = ""

        for page in reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

        # STEP 3: ADD THIS PART
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_text(text)
        # Debugging Information
        st.write(f"📊 Extracted Characters: {len(text):,}")

        st.write(f"📝 Approximate Words: {len(text.split()):,}")

        st.write(f"📦 Total Chunks Created: {len(chunks)}")

        st.write(f"### Total Chunks Created: {len(chunks)}")

        st.write("## Chunk Preview")

        for i, chunk in enumerate(chunks[:3]):
            st.text_area(
                f"Chunk {i+1}",
                chunk,
                height=200,
                key=f"chunk_{i}"
            )

        # Existing code
        st.write("### Extracted Text Preview")

        st.text_area(
            "Content",
            text[:3000],
            height=300
        )

