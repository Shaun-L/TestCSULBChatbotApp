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
from langchain_community.vectorstores import FAISS
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START




local_llm = "llama3.1:8b-instruct-q6_K"
nest_asyncio.apply()

def get_all_links(url):
    """
    Fetch all links from the given URL that contain 'csulb' in the href attribute.
    """
    try:
        response = requests.get(url, timeout=20, verify=False)  # Increased timeout to 20 seconds
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

def get_course_links(url):
    """
    Fetch all course links from the given URL.
    """
    try:
        response = requests.get(url, timeout=20, verify=False)  # Increased timeout to 20 seconds
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    course_links = set()

    for tr_tag in soup.find_all('tr'):
        a_tag = tr_tag.find('a', href=True)
        if a_tag and 'preview_course_nopop.php' in a_tag['href']: #for course links
            full_url = urljoin(url, a_tag['href'])
            course_links.add(full_url)
        if a_tag and 'filter%5Bcpage%5D=' in a_tag['href']: #for pagination links
            for a_tag in tr_tag.find_all('a', href=True):
                full_url = urljoin(url, a_tag['href'])
                course_links.add(full_url)
        
    
    return course_links


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
        current_url, depth = to_visit.pop(0) #This makes it BFS, to make DFS, turn into 'pop()'
        if depth > max_depth:
            continue
        if current_url in visited:
            continue

        print(f"Crawling: {current_url} at depth {depth}")
        
        visited.add(current_url)
        
        links = get_all_links(current_url)
        course_links = get_course_links(current_url)

        for link in links:
            if link not in visited and (link, depth + 1) not in to_visit:
                to_visit.append((link, depth + 1))

        for course_link in course_links:
            if course_link not in visited and (course_link, depth + 1) not in to_visit:
                to_visit.append((course_link, depth + 1))

    return visited

def make_db():
  urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Disable warnings for SSL cert errors

  start_url = "http://catalog.csulb.edu/content.php?catoid=10&navoid=1156"  # Using HTTPS for starting at the CSULB catalog
  crawled_urls = crawl(start_url, max_depth=1)  # depth indicates how many urls away from the root

  print(f"Crawled URLs: {len(crawled_urls)}")
  #Loading Documents
  loader = WebBaseLoader(list(crawled_urls), continue_on_failure=True)
  loader.requests_per_second = 2
  docs = loader.aload()
  print("Finished loading")

  # This part was creating an error 

  #for document in docs:
  # document.page_content = document.page_content.replace("\n", "")
  # document.page_content = document.page_content.replace("\xa0", "")
  # document.page_content = document.page_content.replace("\t", "")


  #Split Documents
  print("1")
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
  print("2")
  all_splits = text_splitter.split_documents(docs)

  #Storing Documents into ChromaDB
  print("3")
  model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
  print("4")
  gpt4all_kwargs = {'allow_download': 'True'}
  print("5")
  embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
    )
  print("6")
  vectorstore = FAISS.from_documents(
    all_splits,
    embeddings
    )
  print("7")
  return vectorstore

def make_retrieval_grader():
    #Checks if query is located within the urls (FAISS)

    #LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need ot be a stringent test. The goal is to filter out erroneous retrievals.
    \n --- \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explaination. \n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"]
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    
    return retrieval_grader

def make_llm_generation():
    #Generating Output, based on relevance

    # Prompt 
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks for students attending California State University, Long Beach.\n
    Only provide information that lies within the retrieved documents.
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Retrieved Context:
    \n ------- \n
        {context} 
    \n ------- \n
    \nQuestion: {question}
    Answer: <|eot_id|><start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"]
    )

    llm = ChatOllama(model=local_llm, temperature=0.5)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

def make_hallucination_grader():
    ### Hallucination Grader

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is strictly grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader

def make_answer_grader():
    ### Answer Grader

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    return answer_grader

def make_question_grader():
    ### Question Grader

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether a question is appropriate and can be 
        answered using information on the California State University, Long Beach website and catalog.\n 
        Here is the question:
        \n ------- \n
        {question} 
        \n ------- \n
        Give a binary score 'yes' or 'no' to indicate whether the question is relevant and appropriate. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["question"],
    )

    question_grader = prompt | llm | JsonOutputParser()
    return question_grader

def make_question_rewriter():
    ### Question Re-writer

    # LLM
    llm = ChatOllama(model=local_llm, temperature=0)

    # Prompt
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval and remains relevant to a 'California State University, Long Beach' context. Look at the initial and formulate an improved question. \n
        Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter



###### RAG PIPELINE ########
def compile_model():
    
    ### State
    class GraphState(TypedDict):
        question : str
        generation : str
        documents : List[str]

    ### Nodes
    def retrieve(state):
        print("---RETRIEVE---")
        question = state['question']

        #Retrieval 
        documents = faissRetriever.invoke(question)
        return {"documents": documents, "question": question}

    def grade_documents(state):
        print("---CHECK DOCUMENTS RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        #Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            print(score)
            grade = score['score']

    #document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
                print(d.metadata['source'])
        return {"documents": filtered_docs, "question": question}

    def generate(state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    ### Conditional Edges
    def decide_to_generate(state):
        question = state["question"]

    # Need to create a question grader
        score = question_grader.invoke({"question": question})
        grade = score['score']
    
        if grade == "yes":
            print("---DECISION: QUESTION IS APPROPRIATE---")
            return "continue"
        else:
            print("---DECISION: QUESTION IS NOT APPROPRIATE---")
        return "do not continue"


    def check_hallucination(state):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        print(f"This is the generation:\n{generation}\n")

        # Need to create hallucination grader
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']
    
        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            # Need to create answer grader
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score['score']
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRY---")
            return "not supported" #make new generation with documents in generation

    def question_not_appropriate(state):
        print("---OUTPUTTING BAD QUESTION---")
        generation = "Please ask a question that is relevant to CSULB."
        return {"documents": [], "question": state['question'], "generation": generation}

    def fix_hallucination(state):
        print("---REWRITING QUESTION---")
        question = question_rewriter.invoke({"question": state['question']})
        print(F"\nRewritten Question: {question}\n")
        return {"question": question}

    def useless_context(state):
        print("---OUTPUTTING USELSS CONTEXT---")
        generation = "Sorry, I could not find documents relevant enough to answer your question, please consider checking the CSULB catalogs and websites."
        return {"documents": state['documents'], "question": state['question'], "generation": generation} 

    def build_graph():
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents) #grade deocuments
        workflow.add_node("generate", generate) 
        workflow.add_node("question_not_appropriate", question_not_appropriate)
        workflow.add_node("fix_hallucination", fix_hallucination)
        workflow.add_node("bad_context", useless_context)

        workflow.add_conditional_edges(
            START,
            decide_to_generate,
            {
                "continue": "retrieve",
                "do not continue": "question_not_appropriate", # CHANGE THIS LATER
            },
        )
        workflow.add_edge("question_not_appropriate", END)
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_conditional_edges(
            "generate",
            check_hallucination,
            {
                "useful": END, #successful
                "not useful": "bad_context",
                "not supported": "fix_hallucination", 

            },
        )
        workflow.add_edge("fix_hallucination", "retrieve")
        workflow.add_edge("bad_context", END)

        # Compile
        app = workflow.compile()

        return app
    
    vectorbase = make_db()
    print("VECTOR DB CREATED")
    faissRetriever = vectorbase.as_retriever()
    print("FAISS RETRIEVER CREATED")
    retrieval_grader = make_retrieval_grader()
    print("RETRIEVAL GRADER CREATED")
    rag_chain = make_llm_generation()
    print("RAG CHAIN CREATED")
    question_grader = make_question_grader()
    print("QUESTION GRADER CREATED")
    hallucination_grader = make_hallucination_grader()
    print("HALLUCINATION GRADER CREATED")
    answer_grader = make_answer_grader()
    print("ANSWER GRADER CREATED")
    question_rewriter = make_question_rewriter()
    print("QUESTION REWRITER CREATED (LAST ONE)")
    app = build_graph()
    print("WORKFLOW COMPILED")
    print("------MODEL FINISHED------")

    return app

   

def generate_response(app, query):
    # Run
    inputs = {"question": query} #getting rid of filler words can improve retriever accuracy
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            print(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)

    # Final generation
    return value["generation"]
