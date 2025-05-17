from flask import Flask, render_template, request, jsonify
import os
import requests
import json
from bs4 import BeautifulSoup
# Updated imports for langchain
from langchain_community.embeddings import SentenceTransformerEmbeddings
# Import FAISS instead of Chroma
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
# Import ChatOllama
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import logging

# Import torch to check for CUDA availability for embeddings
try:
    import torch
    print("Successfully imported torch.")
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch not installed. CUDA will not be available for embeddings.")
    TORCH_AVAILABLE = False
except Exception as e:
     print(f"Warning: Failed to import torch: {e}. CUDA will not be available for embeddings.")
     TORCH_AVAILABLE = False


app = Flask(__name__)

# --- Configuration ---
# Get API key from environment variable or use default for development
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "26fafe1d19be94686816682af49236ff5538ad76") # Replace with your actual key or ensure env var is set
# --- Ollama Configuration ---
# Specify the name of the Ollama model you want to use (e.g., mistral, llama2, phi3)
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "mistral")
# Optional: Specify the base URL for the Ollama server if it's not localhost:11434
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding model name
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")

# Determine the device for embeddings (CPU or CUDA)
EMBEDDING_DEVICE = 'cpu'
if TORCH_AVAILABLE and torch.cuda.is_available():
    EMBEDDING_DEVICE = 'cuda'
    print(f"CUDA is available. Embedding model will use device: {EMBEDDING_DEVICE}")
elif TORCH_AVAILABLE:
     print("torch installed, but CUDA is not available. Embedding model will use device: cpu")
else:
     print("torch not installed. Embedding model will use device: cpu")


# Setup callback manager for server-side logging (optional with Ollama API)
# Note: StreamingStdOutCallbackHandler prints to the console where the Flask app is running.
# You might want a different callback for production logging.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Set up basic logging for the Flask app
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

def search_with_serper(query, gl="us", hl="en", num_results=7):
    """
    Perform a search using the Serper API
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "gl": gl,
        "hl": hl,
        "num": num_results
    })

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    if not SERPER_API_KEY or SERPER_API_KEY == "YOUR_SERPER_API_KEY":
         app.logger.warning("SERPER_API_KEY is not set. Search will likely fail.")
         return None

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        app.logger.info(f"Serper API call successful for query: {query}")
        return response.json()

    except requests.exceptions.Timeout:
        app.logger.error(f"Serper API request timed out for query: {query}")
        return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error making Serper API request for query '{query}': {e}")
        return None
    except json.JSONDecodeError:
        app.logger.error(f"Error decoding Serper API JSON response for query: {query}")
        app.logger.error(f"Response content: {response.text[:500]}") # Log partial response
        return None

def get_urls_from_serper(query: str, num_results: int = 7) -> list[str]:
    """Uses Serper API to get search result URLs."""
    results = search_with_serper(query, gl="us", hl="en", num_results=num_results)
    if not results:
        app.logger.warning("Serper search failed or returned no results.")
        return []

    urls = []

    # Get URLs from organic results
    if 'organic' in results:
        urls.extend([res['link'] for res in results['organic'] if 'link' in res])

    # Get URL from answer box if available - often redundant but good practice
    if 'answerBox' in results and 'link' in results['answerBox']:
         # Check if the link is already in organic results to avoid immediate duplicates
         if results['answerBox']['link'] not in urls:
             urls.append(results['answerBox']['link'])

    # Get URL from related questions answer box if available
    if 'relatedQuestions' in results:
        for rq in results['relatedQuestions']:
            if 'answer' in rq and 'link' in rq['answer']:
                 if rq['answer']['link'] not in urls:
                     urls.append(rq['answer']['link'])

    # Get URL from knowledge graph if available
    if 'knowledgeGraph' in results and 'source' in results['knowledgeGraph'] and 'link' in results['knowledgeGraph']['source']:
         if results['knowledgeGraph']['source']['link'] not in urls:
             urls.append(results['knowledgeGraph']['source']['link'])

    # Remove duplicates while preserving order (more robust)
    urls = list(dict.fromkeys(urls))
    app.logger.info(f"Found {len(urls)} URLs from search: {urls}")

    # Optionally filter out known bad domains or file types
    # Added filtering for common forum/social media sites that might have poor text structure
    urls = [url for url in urls if not url.endswith(('.pdf', '.zip', '.jpg', '.png')) and 'youtube.com' not in url and 'reddit.com' not in url and 'twitter.com' not in url and 'facebook.com' not in url]
    app.logger.info(f"Filtered URLs: {urls}")

    return urls


def fetch_and_process_url(url: str) -> list[Document]:
    """Fetches content from a single URL, extracts text, and chunks it."""
    try:
        app.logger.info(f"Fetching content from {url}")
        # Add a reasonable timeout and user-agent
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Try lxml parser first, fall back to html.parser
        try:
            soup = BeautifulSoup(response.content, 'lxml')
        except Exception as e:
             app.logger.warning(f"lxml parser failed for {url}: {e}, falling back to html.parser")
             soup = BeautifulSoup(response.content, 'html.parser')

        # --- Improved Text Extraction ---
        # Remove script, style, meta, noscript, header, footer, nav, aside, form tags
        for script_or_style in soup(['script', 'style', 'meta', 'noscript', 'header', 'footer', 'nav', 'aside', 'form', 'svg']):
            script_or_style.decompose()

        # Get text from the body and normalize whitespace
        # Use get_text with a space separator to prevent words from different tags merging together
        text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''
        text = ' '.join(text.split()) # Further normalize whitespace

        # If text is still too short, might indicate an issue or very sparse page
        if not text.strip() or len(text) < 200: # Increased minimum length slightly
            app.logger.warning(f"Not enough substantial content after cleaning for {url}")
            return []

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Increased chunk size slightly
            chunk_overlap=200, # Maintain overlap
            separators=["\n\n", "\n", ". ", " ", ""] # Prioritize splitting by paragraphs, then lines, then sentences, etc.
        )

        # Split text into chunks
        chunks = text_splitter.split_text(text)
        app.logger.info(f"Created {len(chunks)} chunks from {url}")

        # Create Document objects
        docs = [Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]
        return docs

    except requests.exceptions.Timeout:
        app.logger.warning(f"Timeout fetching {url}")
        return []
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching {url}: {e}")
        return []
    except Exception as e:
        app.logger.exception(f"Unexpected error processing {url}") # Use exception logging for traceback
        return []


def fetch_and_process_urls(urls: list[str]) -> list[Document]:
    """Fetches content from multiple URLs, extracts text, and chunks it."""
    all_docs = []

    # Process a limited number of URLs to avoid excessive fetching
    # For web search, 5-7 good sources are usually sufficient
    urls_to_process = urls[:7] # Limit to first 7 URLs

    for url in urls_to_process:
        docs = fetch_and_process_url(url)
        all_docs.extend(docs)

    app.logger.info(f"Total documents created: {len(all_docs)}")
    return all_docs

def create_vector_store(docs: list[Document]):
    """Creates a FAISS vector store from documents using Sentence Transformers."""
    if not docs:
        app.logger.warning("No documents to create vector store from.")
        return None

    app.logger.info(f"Creating FAISS vector store with {len(docs)} documents using embedding device: {EMBEDDING_DEVICE}")
    try:
        # Initialize embeddings model, using the determined device (CUDA or CPU)
        # The model_kwargs dictionary is passed directly to the SentenceTransformer constructor
        embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': EMBEDDING_DEVICE} # Use determined device
        )

        # Create FAISS vector store from documents
        # FAISS.from_documents handles batching internally within the embedding call
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        app.logger.info("FAISS vector store created successfully.")
        return vectorstore

    except Exception as e:
        app.logger.exception(f"Error creating FAISS vector store")
        # FAISS.from_documents might fail due to OOM on GPU if docs are too many/large
        # or due to issues with the embedding model itself.
        # No automatic retry with smaller batches like Chroma,
        # as FAISS.from_documents doesn't expose batch size easily.
        return None

def setup_rag_chain(vectorstore):
    """Sets up the RAG chain using the vector store and Ollama model."""
    if vectorstore is None:
        app.logger.error("Cannot setup RAG chain without a valid vector store.")
        return None

    try:
        # Note: ChatOllama connects to the Ollama server.
        # GPU usage happens *within* the Ollama server process,
        # not controlled directly by this Python code,
        # provided Ollama is installed/configured for your GPU.
        app.logger.info(f"Loading Ollama model '{OLLAMA_MODEL_NAME}' from {OLLAMA_BASE_URL}")
        llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            # max_tokens=1000, # Use num_predict with Ollama
            num_predict=1000, # Max tokens to generate
            num_ctx=4096, # Context window size
            # callback_manager=callback_manager, # Use if you need server-side streaming logs
        )
        app.logger.info("Ollama model loaded successfully.")

    except Exception as e:
        app.logger.exception(f"Error loading Ollama model or connecting to server")
        app.logger.error(f"Please ensure Ollama server is running at {OLLAMA_BASE_URL}")
        app.logger.error(f"Also ensure the model '{OLLAMA_MODEL_NAME}' is pulled (run 'ollama pull {OLLAMA_MODEL_NAME}' in your terminal) AND that your Ollama installation is configured to use your GPU.")
        return None

    # Setup the retriever
    # FAISS vector store's as_retriever method works the same way
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks
    app.logger.info(f"FAISS retriever setup with k=5.")


    # Prompt template that instructs the model to cite sources
    # Ensure the prompt clearly asks for citations based *only* on provided context.
    template = """[INST] Use the following pieces of context to answer the user's question comprehensively and accurately.
If you don't know the answer based *only* on the context provided, state that you don't know.
Do not invent information or sources.
For each statement in your answer that is supported by the context, include a citation at the end of the sentence or paragraph, referencing the source URL like this: [URL].
Ensure you only cite the URLs present in the provided context documents. If a fact comes from multiple sources, list multiple citations [URL1][URL2].
If no relevant context is found for a specific point, do not provide a citation for that point.

Context:
{context}

User question: {question} [/INST]"""

    custom_rag_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Setup the RAG chain
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" is suitable for smaller contexts like fetched web pages
            retriever=retriever,
            return_source_documents=True, # Return source documents for citation
            chain_type_kwargs={"prompt": custom_rag_prompt}
        )
        app.logger.info("RAG chain setup complete.")
        return qa_chain

    except Exception as e:
        app.logger.exception(f"Error setting up RetrievalQA chain")
        return None


import time  # Add this import at the top with other imports

def answer_question_with_web_search(query: str):
    """Combines all steps to answer a question using web search RAG."""
    start_time = time.time()
    app.logger.info(f"\n--- Processing query: {query} ---")

    # 1. Get URLs from Serper
    serper_start = time.time()
    urls = get_urls_from_serper(query)
    serper_time = time.time() - serper_start

    if not urls:
        total_time = time.time() - start_time
        app.logger.info(f"Query failed in {total_time:.2f} seconds (Serper search: {serper_time:.2f}s)")
        return {"answer": "Sorry, I couldn't find enough relevant websites for your query.", 
                "sources": [],
                "timing": {"total": f"{total_time:.2f}s", "search": f"{serper_time:.2f}s"}}

    # 2. Fetch and process content
    fetch_start = time.time()
    documents = fetch_and_process_urls(urls)
    fetch_time = time.time() - fetch_start

    if not documents:
        total_time = time.time() - start_time
        app.logger.warning("Found URLs but failed to extract documents.")
        return {"answer": "Sorry, I found some websites but couldn't extract useful information from them.", 
                "sources": urls,
                "timing": {"total": f"{total_time:.2f}s", "search": f"{serper_time:.2f}s", "fetch": f"{fetch_time:.2f}s"}}

    # 3. Create Vector Store (FAISS)
    vectorstore_start = time.time()
    vectorstore = create_vector_store(documents)
    vectorstore_time = time.time() - vectorstore_start

    if vectorstore is None:
        total_time = time.time() - start_time
        app.logger.error("Failed to create FAISS vector store.")
        return {"answer": "Sorry, I couldn't process the web content to set up the knowledge base.", 
                "sources": urls,
                "timing": {"total": f"{total_time:.2f}s", "search": f"{serper_time:.2f}s", 
                          "fetch": f"{fetch_time:.2f}s", "vectorstore": f"{vectorstore_time:.2f}s"}}

    # 4. Setup RAG Chain
    chain_setup_start = time.time()
    qa_chain = setup_rag_chain(vectorstore)
    chain_setup_time = time.time() - chain_setup_start

    if qa_chain is None:
        total_time = time.time() - start_time
        return {"answer": "Failed to initialize the AI model. Please check the server logs.", 
                "sources": urls,
                "timing": {"total": f"{total_time:.2f}s", "search": f"{serper_time:.2f}s", 
                          "fetch": f"{fetch_time:.2f}s", "vectorstore": f"{vectorstore_time:.2f}s",
                          "chain_setup": f"{chain_setup_time:.2f}s"}}

    # 5. Run the RAG chain
    try:
        app.logger.info("Running RAG chain to generate answer...")
        inference_start = time.time()
        result = qa_chain.invoke({"query": query})
        inference_time = time.time() - inference_start

        answer = result.get("result", "No answer generated by the model.")
        source_documents = result.get("source_documents", [])
        sources = sorted(list(set([doc.metadata['source'] for doc in source_documents if 'source' in doc.metadata])))

        total_time = time.time() - start_time
        timing_info = {
            "total": f"{total_time:.2f}s",
            "search": f"{serper_time:.2f}s",
            "fetch": f"{fetch_time:.2f}s",
            "vectorstore": f"{vectorstore_time:.2f}s",
            "chain_setup": f"{chain_setup_time:.2f}s",
            "inference": f"{inference_time:.2f}s"
        }

        app.logger.info(f"Answer generated in {total_time:.2f} seconds:")
        app.logger.info(f"- Search: {serper_time:.2f}s")
        app.logger.info(f"- Fetch & Process: {fetch_time:.2f}s")
        app.logger.info(f"- Vector Store: {vectorstore_time:.2f}s")
        app.logger.info(f"- Chain Setup: {chain_setup_time:.2f}s")
        app.logger.info(f"- Inference: {inference_time:.2f}s")

        return {
            "answer": answer, 
            "sources": sources,
            "timing": timing_info
        }

    except Exception as e:
        total_time = time.time() - start_time
        app.logger.exception(f"Error during RAG chain execution")
        return {"answer": f"An error occurred while generating the answer. Please check the Ollama server and model '{OLLAMA_MODEL_NAME}'. Error: {str(e)}", 
                "sources": urls,
                "timing": {"total": f"{total_time:.2f}s", "error_at": "inference"}}
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Call the main function that orchestrates the web search RAG
    result = answer_question_with_web_search(question)
    return jsonify(result)

# Custom error handlers
@app.errorhandler(404)
def not_found(e):
    app.logger.warning(f"Not found: {request.path}")
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    # Log the actual error in the server logs for debugging
    app.logger.exception('An internal server error occurred')
    return jsonify({"error": "Internal server error. Please check server logs."}), 500

if __name__ == "__main__":
    print("--- Starting RAG Flask Application ---")
    print(f"Using Ollama Model: {OLLAMA_MODEL_NAME} at {OLLAMA_BASE_URL}")
    print(f"Using Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Embedding device set to: {EMBEDDING_DEVICE}") # Report the chosen device
    print("Vector Store: FAISS (in-memory per request)")


    if not SERPER_API_KEY or SERPER_API_KEY == "YOUR_SERPER_API_KEY":
         print("\n!!! WARNING: SERPER_API_KEY is not set or is using the placeholder value.")
         print("!!! Web search functionality will not work.")
         print("!!! Set the SERPER_API_KEY environment variable or replace the placeholder in the code.")

    print("\nAttempting to connect to Ollama server and load model config...")
    # Basic check: Try initializing the ChatOllama instance.
    # This doesn't guarantee GPU usage, that's an Ollama server detail.
    try:
        test_llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0 # Use temp=0 for a test call
        )
        # Optional: Make a tiny test call - more reliable but adds delay
        # test_llm.invoke("Hi", max_tokens=5)
        print(f"Ollama initialization for model '{OLLAMA_MODEL_NAME}' successful.")
        print("Model *might* be ready and potentially using GPU if configured in Ollama.")
        print("The first user query will perform a more thorough check via the RAG chain.")
    except Exception as e:
        print(f"\n!!! ERROR: Failed to initialize Ollama model '{OLLAMA_MODEL_NAME}'.")
        print(f"!!! Please ensure the Ollama server is running at {OLLAMA_BASE_URL}.")
        print(f"!!! And that the model '{OLLAMA_MODEL_NAME}' is downloaded (run 'ollama pull {OLLAMA_MODEL_NAME}').")
        print(f"!!! Also ensure your Ollama server is configured to use your GPU.")
        print(f"!!! Specific error: {e}")
        # Decide whether to exit or allow the app to start but fail on queries.
        # Allowing it to start might be useful for debugging other parts.
        # raise e # Uncomment this line to prevent server start if Ollama connection fails
    print("---------------------------------------")


    # Use environment variables for host/port if available
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    # Debug mode should be False in production
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    print(f"Starting Flask app on http://{host}:{port}/ (debug: {debug})")
    # use_reloader=False is important when models load resources (like CUDA contexts)
    app.run(host=host, port=port, debug=debug, use_reloader=False)