import nest_asyncio
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import urllib3
from urllib.parse import urljoin
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from langchain_community.llms import llamacpp
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser



local_llm = "llama3"

def get_all_links(url):
    """
    Fetch all links from the given URL that contain 'csulb' in the href attribute.
    """
    try:
        response = requests.get(url, timeout=10, verify=False)  # Added verify=False to bypass SSL cert errors
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    links = set()

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if 'csulb.edu' in href and '.pdf' not in href and 'mailto' not in href:
            full_url = urljoin(url, href)
            links.add(full_url)
    
    return links

def crawl(start_url, max_depth=5):
    """
    Crawl from the start URL and collect all valid URLs containing 'csulb' until a dead end is reached
    or the maximum depth is exceeded.
    """
    visited = set()
    to_visit = [(start_url, 0)]  # Each element is a tuple (URL, depth)

    # Setup requests session with retries
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    while to_visit:
        current_url, depth = to_visit.pop()
        if depth > max_depth:
            continue
        if current_url in visited:
            continue

        print(f"Crawling: {current_url} at depth {depth}")
        visited.add(current_url)

        links = get_all_links(current_url)
        for link in links:
            if link not in visited and (link, depth + 1) not in to_visit:
                to_visit.append((link, depth + 1))

    return visited

def make_db():
  urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Disable warnings for SSL cert errors

  start_url = "http://catalog.csulb.edu/index.php?catoid=10"  # this URL is for starting at the CSULB catalog
  crawled_urls = crawl(start_url, max_depth=5)  # depth indicates how many urls away from the root

  #Loading Documents
  loader = WebBaseLoader(list(crawled_urls), continue_on_failure=True)
  loader.requests_per_second = 2
  docs = loader.aload()

  #Split Documents
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  all_splits = text_splitter.split_documents(docs)

  #Storing Documents into ChromaDB
  embedding = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")

  vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma-1",
    embedding=embedding,
  )

  return vectorstore

def generate_response(context, query):
  # Create Retriever
  retriever = context.as_retriever()

  # Prompt 
  prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks 
    for students attending California State University, Long Beach.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say you don't know.
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"]
  )

  llm = ChatOllama(model=local_llm, temperature=0.8)

  # Chain
  rag_chain = prompt | llm | StrOutputParser()

  # Run
  docs = retriever.invoke(query)
  generation = rag_chain.invoke({"context": docs, "question": query})

  return generation



