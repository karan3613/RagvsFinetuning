import os
import streamlit as st
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import load_dataset
import asyncio


# Ensure the current thread has an event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Configure page
st.set_page_config(
    page_title="Model Comparison Tool",
    page_icon="🤖",
    layout="wide"
)
load_dotenv()
# Mock dataset - Replace with your actual dataset
@st.cache_data
def loading_dataset():
    """Load your dataset here. Replace with actual data loading."""
    dataset = load_dataset("prsdm/Machine-Learning-QA-dataset", split= "train" , encoding="utf-8")
    return pd.DataFrame(dataset)


# Mock model implementations - Replace with your actual models
class RAGModel:
    """Replace this with your actual RAG model implementation"""

    def __init__(self):
        self.chain  = self.generate_chain()

    def generate_chain(self):
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        model = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_api_key
        )
        vector_embeddings = FAISS.load_local(
            "embeddings_db",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=vector_embeddings.as_retriever())
        return chain

    def generate_response(self, query: str) -> str:
        response = self.chain.invoke(query)
        return response['result']


class FineTunedModel:
    """Replace this with your actual fine-tuned model implementation"""

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("finetuned_model")
        self.tokenizer = AutoTokenizer.from_pretrained("finetuned_model")

    def generate_response(self, query: str) -> str:
        messages = [
            {"role": "user", "content": f"{query}"}
        ]
        # Tokenize the user input with the chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,  # Add padding to match sequence lengths
        ).to("cuda")

        attention_mask = inputs != self.tokenizer.pad_token_id

        outputs = self.model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=64,
            use_cache=True,  # Use cache for faster token generation
            temperature=0.6,  # Controls randomness in responses
            min_p=0.1,  # Set minimum probability threshold for token selection
        )
        # Decode the generated tokens
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant response
        if "assistant\n" in full_text:
            response = full_text.split("assistant\n", 1)[1].strip()
        else:
            response = full_text.strip()
        return response


# Initialize models (replace with your actual model initialization)
@st.cache_resource
def load_models():
    """Initialize your models here"""
    rag_model = RAGModel()
    finetuned_model = FineTunedModel()
    return rag_model, finetuned_model


# Helper function to get random dataset sample
def get_random_sample(dataset):
    """Get a random question-answer pair from dataset"""
    idx = np.random.randint(0, len(dataset))
    return dataset.iloc[idx]['Question'], dataset.iloc[idx]['Answer'], idx


# Main app
def main():
    st.title("Model Comparison App")
    st.markdown("Compare responses from RAG Model, Fine-tuned Model, and Original Dataset")

    # Load data and models
    dataset = loading_dataset()
    rag_model, finetuned_model = load_models()

    # Sidebar for dataset info
    with st.sidebar:
        st.header("📊 Dataset Info")
        st.write(f"Total samples: {len(dataset)}")
        st.write("Models loaded:")
        st.write("✅ RAG Model")
        st.write("✅ Dataset Reference")
        st.write("✅ Fine-tuned Model")


        if st.button("🎲 Load Random Question"):
            random_question, _, idx = get_random_sample(dataset)
            st.session_state.random_question = random_question
            st.session_state.random_idx = idx

    # Main input section
    st.header("📝 Input Section")

    # Show random question if available
    if 'random_question' in st.session_state:
        st.text(f"🎯 Random Question (Index: {st.session_state.random_idx}): {st.session_state.random_question}")
        if st.button("Use Random Question"):
            st.session_state.user_input = st.session_state.random_question

    # Text input
    user_input = st.text_area(
        "Enter your question:",
        value=st.session_state.get('user_input', ''),
        height=100,
        placeholder="Type your question here..."
    )

    # Generate button
    if st.button("🚀 Generate All Responses", type="primary", use_container_width=True):
        if user_input.strip():
            # Store the input
            st.session_state.user_input = user_input

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Generate responses
            with st.spinner("Generating responses..."):
                responses = {}

                # RAG Model Response
                status_text.text("Generating RAG response...")
                progress_bar.progress(25)
                responses['rag'] = rag_model.generate_response(user_input)

                # Fine-tuned Model Response
                status_text.text("Generating Fine-tuned model response...")
                progress_bar.progress(50)
                responses['finetuned'] = finetuned_model.generate_response(user_input)

                # Find closest match in dataset (mock implementation)
                status_text.text("Finding dataset reference...")
                progress_bar.progress(75)

                # Simple similarity search (replace with actual similarity search)
                best_match_idx = 0
                best_match_score = 0
                for idx, question in enumerate(dataset['Question']):
                    # Mock similarity calculation - replace with actual similarity
                    similarity = len(set(user_input.lower().split()) & set(question.lower().split()))
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_idx = idx

                responses['original'] = dataset.iloc[best_match_idx]['Answer']
                responses['original_question'] = dataset.iloc[best_match_idx]['Question']
                responses['dataset_idx'] = best_match_idx

                progress_bar.progress(100)
                status_text.text("Complete!")
                time.sleep(0.5)

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Store responses in session state
                st.session_state.responses = responses
                st.session_state.query = user_input
        else:
            st.error("Please enter a question first!")

    # Display results
    if 'responses' in st.session_state:
        st.header("📊 Comparison Results")
        st.markdown(f"**Query:** {st.session_state.query}")

        # Create three columns for comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔍 RAG Model")
            st.markdown("---")
            with st.container():
                st.write(st.session_state.responses['rag'])
                # Metrics for RAG (mock - replace with actual metrics)
                st.markdown("**Metrics:**")
                st.metric("Response Length", f"{len(st.session_state.responses['rag'])} chars")

        with col2:
            st.subheader("📚 Original Dataset")
            st.markdown("---")
            with st.container():
                if 'original_question' in st.session_state.responses:
                    st.markdown(f"**Matched Question (Index: {st.session_state.responses['dataset_idx']}):**")
                    st.info(st.session_state.responses['original_question'])

                st.markdown("**Answer:**")
                st.write(st.session_state.responses['original'])

                # Metrics for original
                st.markdown("**Metrics:**")
                st.metric("Response Length", f"{len(st.session_state.responses['original'])} chars")
                st.metric("Dataset Index", st.session_state.responses['dataset_idx'])

        with col3:
            st.subheader("🎯 Fine-tuned Model")
            st.markdown("---")
            with st.container():
                st.write(st.session_state.responses['finetuned'])

                # Metrics for fine-tuned (mock - replace with actual metrics)
                st.markdown("**Metrics:**")
                st.metric("Response Length", f"{len(st.session_state.responses['finetuned'])} chars")

        # Additional comparison section
        st.header("📈 Detailed Analysis")
        # Export functionality
        export_data = {
            'query': st.session_state.query,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'responses': st.session_state.responses
        }

        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"model_comparison_{int(time.time())}.json",
            mime="application/json"
        )

        # Display raw data
        with st.expander("View Raw Data"):
            st.json(export_data)


if __name__ == "__main__":
    main()