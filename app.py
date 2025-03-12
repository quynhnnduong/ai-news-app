import os
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
import spacy
import re

# Load spaCy model for NER and topic extraction
nlp = spacy.load("en_core_web_sm")
load_dotenv()

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = nx.Graph()
if 'topics' not in st.session_state:
    st.session_state.topics = Counter()
if 'entities' not in st.session_state:
    st.session_state.entities = Counter()

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_topics_and_entities(text):
    """Extract topics and named entities from text using spaCy"""
    doc = nlp(text)
    
    # Extract potential topics (nouns and noun phrases)
    topics = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
    
    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return topics, entities

def update_knowledge_graph(topics, entities, filename):
    """Update the knowledge graph with new topics and entities"""
    # Add the document node
    if filename not in st.session_state.knowledge_graph:
        st.session_state.knowledge_graph.add_node(filename, type='document')
    
    # Add topic nodes and connect to document
    for topic in topics:
        if topic not in st.session_state.knowledge_graph:
            st.session_state.knowledge_graph.add_node(topic, type='topic')
        st.session_state.knowledge_graph.add_edge(filename, topic)
        st.session_state.topics[topic] += 1
    
    # Add entity nodes and connect to document
    for entity, label in entities:
        entity_key = f"{entity} ({label})"
        if entity_key not in st.session_state.knowledge_graph:
            st.session_state.knowledge_graph.add_node(entity_key, type='entity', label=label)
        st.session_state.knowledge_graph.add_edge(filename, entity_key)
        st.session_state.entities[entity_key] += 1

def visualize_knowledge_graph():
    """Create a visualization of the knowledge graph"""
    G = st.session_state.knowledge_graph
    
    # Create position layout
    pos = nx.spring_layout(G)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Draw document nodes
    doc_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'document']
    nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color='red', node_size=500, alpha=0.8)
    
    # Draw topic nodes
    topic_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'topic']
    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_color='blue', node_size=300, alpha=0.6)
    
    # Draw entity nodes
    entity_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'entity']
    nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color='green', node_size=300, alpha=0.6)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    
    # Draw labels - only for nodes with high degree or documents
    labels = {n: n for n in G.nodes() if G.degree(n) > 1 or n in doc_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Knowledge Graph Visualization")
    plt.axis('off')
    return plt

def summarize_document(text, filename):
    """Generate a summary of the document using LLM"""
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    prompt = f"""
    Please provide a concise summary (maximum 3 paragraphs) of the following document:
    
    {text[:4000]}  # Using first 4000 chars for summary
    
    Include:
    1. Main topics and key insights
    2. Key entities mentioned (people, organizations, technologies)
    3. Overall importance of this document
    """
    
    summary = llm.predict(prompt)
    return summary

def create_conversational_chain(documents):
    """Create a conversational chain for the document collection"""
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Create a conversational chain
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True
    )
    
    return conversation

# Streamlit UI
st.title("AI Knowledge Manager Prototype")

# Sidebar for API key
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.subheader("Uploaded Documents")
    for file in st.session_state.processed_files:
        st.write(f"- {file}")
    
    st.subheader("Top Topics")
    for topic, count in st.session_state.topics.most_common(5):
        st.write(f"- {topic} ({count})")
    
    st.subheader("Top Entities")
    for entity, count in st.session_state.entities.most_common(5):
        st.write(f"- {entity} ({count})")

# Tabs for different functions
tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process", "Knowledge Graph", "Search & Chat", "Weekly Digest"])

# Tab 1: Upload & Process
with tab1:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.write(file_details)
        
        # Process the file
        if st.button("Process Document"):
            with st.spinner('Processing document...'):
                try:
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    
                    # Extract topics and entities
                    topics, entities = extract_topics_and_entities(text[:10000])  # Limit for performance
                    
                    # Generate summary
                    summary = summarize_document(text, uploaded_file.name)
                    
                    # Update knowledge graph
                    update_knowledge_graph(topics, entities, uploaded_file.name)
                    
                    # Add to processed files list
                    if uploaded_file.name not in st.session_state.processed_files:
                        st.session_state.processed_files.append(uploaded_file.name)
                    
                    # Load document for LangChain
                    with open(f"temp_{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(f"temp_{uploaded_file.name}")
                    pages = loader.load()
                    
                    # Create or update conversation chain
                    st.session_state.conversation = create_conversational_chain(pages)
                    
                    # Display summary
                    st.subheader("Document Summary")
                    st.write(summary)
                    
                    # Display extracted topics
                    st.subheader("Extracted Topics")
                    st.write(", ".join(list(set(topics))[:20]))
                    
                    # Display extracted entities
                    st.subheader("Named Entities")
                    st.write(", ".join([f"{entity} ({label})" for entity, label in entities[:20]]))
                    
                    # Success message
                    st.success("Document processed successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

# Tab 2: Knowledge Graph
with tab2:
    st.header("Knowledge Graph Visualization")
    
    if len(st.session_state.knowledge_graph.nodes) > 0:
        st.write(f"Total nodes: {len(st.session_state.knowledge_graph.nodes)}")
        st.write(f"Total connections: {len(st.session_state.knowledge_graph.edges)}")
        
        # Visualize the graph
        fig = visualize_knowledge_graph()
        st.pyplot(fig)
    else:
        st.info("Upload and process documents to build the knowledge graph.")

# Tab 3: Search & Chat
with tab3:
    st.header("Search & Chat with Your Knowledge")
    
    if not openai_api_key:
        st.warning("Please add your OpenAI API key in the sidebar to enable the chat functionality.")
    elif st.session_state.conversation is None:
        st.info("Please upload and process at least one document to enable chat.")
    else:
        # Chat interface
        query = st.text_input("Ask a question about your documents:")
        
        if query:
            with st.spinner("Thinking..."):
                # Get the response from the conversation chain
                result = st.session_state.conversation({"question": query, "chat_history": st.session_state.chat_history})
                
                # Extract the answer and source documents
                answer = result["answer"]
                source_docs = result["source_documents"]
                
                # Update chat history
                st.session_state.chat_history.append((query, answer))
                
                # Display the answer
                st.subheader("Answer")
                st.write(answer)
                
                # Display the source documents
                st.subheader("Sources")
                for i, doc in enumerate(source_docs):
                    st.write(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 'Unknown')}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.write(f"Question {i+1}: {q}")
                st.write(f"Answer {i+1}: {a}")
                st.write("---")

# Tab 4: Weekly Digest
with tab4:
    st.header("Weekly Knowledge Digest")
    
    if len(st.session_state.processed_files) > 0:
        if st.button("Generate Weekly Digest"):
            with st.spinner("Generating digest..."):
                try:
                    llm = ChatOpenAI(
                        temperature=0.3,
                        model="gpt-4o-mini",
                        openai_api_key=os.environ.get("OPENAI_API_KEY")
                    )
                    
                    # Create digest prompt
                    top_topics = ", ".join([topic for topic, _ in st.session_state.topics.most_common(10)])
                    top_entities = ", ".join([entity for entity, _ in st.session_state.entities.most_common(10)])
                    files = ", ".join(st.session_state.processed_files)
                    
                    prompt = f"""
                    Create a weekly knowledge digest based on the following information:
                    
                    Documents processed this week: {files}
                    
                    Top topics discovered: {top_topics}
                    
                    Key entities mentioned: {top_entities}
                    
                    Please include:
                    1. A summary of key insights from these documents
                    2. Connections between main topics
                    3. Suggested topics for further research
                    4. One practical action item based on this knowledge
                    
                    Format this as a professional weekly digest email.
                    """
                    
                    digest = llm.predict(prompt)
                    
                    # Display the digest
                    st.subheader("Your Weekly Knowledge Digest")
                    st.markdown(digest)
                    
                    # Option to export
                    if st.button("Export as Markdown"):
                        st.download_button(
                            label="Download Digest",
                            data=digest,
                            file_name="weekly_knowledge_digest.md",
                            mime="text/markdown"
                        )
                
                except Exception as e:
                    st.error(f"Error generating digest: {str(e)}")
    else:
        st.info("Upload and process documents to generate a weekly digest.")

# Footer
st.markdown("---")
st.markdown("AI Knowledge Manager Prototype - v0.1")