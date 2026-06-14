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