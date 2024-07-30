#The libraries and models to be downloaded are

!pip install qdrant-client
!pip install PyPDF2 --upgrade
!pip install langchain
!pip install pypdf
!pip install langchain_openai
!pip install langchain_community

import os
import shutil
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

#Set up environment and Initialize the language model

os.environ["OPENAI_API_KEY"] = "your_api_key"

# Initialize the language model with the specified model and temperature
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Paths to multiple PDF documents
num_pdfs=input("please enter the number of pdf files you want to simultaneously talk with \n")
num_pdfs=int(num_pdfs)
pdf_paths=[]
for i in range(num_pdfs):
  pdf_paths.append(input("please enter the pdf file path \n"))
#pdf_paths = [
#    "/content/Turkiye-Recovery-and-Reconstruction-Assessment.pdf",
#    "/content/978-1-349-12362-9_14.pdf",
#    "/content/DMAM_Report_2023_Kahramanmaras-Pazarcik_and_Elbistan_Earthquakes_Report_final_ENG.pdf"
#]

#Function to extract the document name from PDF metadata

def get_document_name(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            document_info = pdf_reader.metadata
            if document_info is not None:
                title = document_info.get('/Title')
                if title:
                    return title
            return os.path.basename(pdf_path)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return os.path.basename(pdf_path)
# Dictionary to hold document chunks and their corresponding documents
doc_chunks = {}

# Load and split all documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
for path in pdf_paths:
    try:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        docs = text_splitter.split_documents(pages)
        doc_chunks[path] = docs
        #print(f"Loaded and split {len(docs)} chunks from {path}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Function to combine selected document chunks
def get_selected_chunks(selected_paths):
    combined_chunks = []
    for path in selected_paths:
        if path in doc_chunks:
            combined_chunks.extend(doc_chunks[path])
        else:
            print(f"Warning: Path {path} not found in loaded documents.")
    return combined_chunks

# Function to create and initialize a Qdrant vector store
def initialize_qdrant(selected_chunks, storage_path=None):
    embeddings = OpenAIEmbeddings()
    qdrant = Qdrant.from_documents(
        selected_chunks,
        embeddings,
        collection_name="my_documents",
        path=storage_path,
        prefer_grpc=False  # Use in-memory storage for testing
    )
    return qdrant
    
#Clean up the storage folder

def cleanup_storage_folder(storage_path):
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
#Define the question to be answered

def ask_questions():
    questions = []
    while True:
        question = input("Please enter your question:\n")
        questions.append(question)
        more_questions = input("Do you have more questions on this topic? (yes/no):\n").strip().lower()
        if more_questions != 'yes':
            break
    return questions

# Main function to start the questioning process
def main():
    all_questions = ask_questions()
    print("\nYou have asked the following questions:")
    for idx, q in enumerate(all_questions, 1):
        print(f"{idx}. {q}")
if __name__ == "__main__":
    main()

#We will use this code where we provide the final version of the code below.

#If the user uploads multiple documents but only wants to get answers from one or a few of them, we use the following code to determine the path indices.

def display_pdf_options(pdf_paths):
    print("Available PDF documents:")
    for idx, path in enumerate(pdf_paths):
        print(f"{idx}. {path}")
    print(f"{len(pdf_paths)}. Select all documents")

def get_user_selection(pdf_paths):
    display_pdf_options(pdf_paths)
    selected_indices = input("Enter the indices of the documents you want to use (comma-separated):\n")
    selected_indices = selected_indices.split(",")
    selected_indices = [int(idx.strip()) for idx in selected_indices]

    if len(pdf_paths) in selected_indices:
        return list(range(len(pdf_paths)))  # User selected all documents
    else:
        return selected_indices
def main():
    selected_indices = get_user_selection(pdf_paths)
    selected_documents = [pdf_paths[i] for i in selected_indices]
    print("\nYou have selected the following documents:")
    for doc in selected_documents:
        print(doc)
if __name__ == "__main__":
    main()

# Get the selected document paths
selected_paths = [pdf_paths[i] for i in user_selected_indices]

# Perform a similarity search on the vector store for each document
for i, path in enumerate(selected_paths):
    document_name = get_document_name(path)
    chunks_for_doc = doc_chunks[path]
    # Use a unique storage folder for each document
    unique_storage_folder = f"./qdrant_storage_instance_{i}"

    # Clean up any previous instances in the storage folder
    cleanup_storage_folder(unique_storage_folder)

    qdrant_doc = initialize_qdrant(chunks_for_doc, storage_path=unique_storage_folder)
    found_docs = qdrant_doc.similarity_search_with_score(question)

    # Check if any documents were found
    if found_docs:
        # Sort the found documents by score in descending order
        found_docs.sort(key=lambda x: x[1], reverse=True)

        # Print the sorted responses
        for document, score in found_docs:
            context = document.page_content
            if context.strip():
                template = """According to {document_name}, the answer is:
                {context}

                Question: {question}

                Helpful Answer:"""

                prompt = PromptTemplate.from_template(template)
                result_prompt = prompt.invoke({"document_name": document_name, "context": context, "question": question})

                response = llm.invoke(result_prompt)

                print(f"Response for {document_name} with score {score}:\n{response.content}\n")
    else:
        print(f"No relevant documents found for {document_name}.")

def main():
    selected_indices = get_user_selection(pdf_paths)
    selected_paths = [pdf_paths[i] for i in selected_indices]

    question = input("Please enter your question:\n")

    best_response = None
    best_document_name = None
    best_score = None

    for i, path in enumerate(selected_paths):
        document_name = get_document_name(path)
        chunks_for_doc = doc_chunks[path]
        unique_storage_folder = f"./qdrant_storage_instance_{i}"

         # Clear storage folder
        cleanup_storage_folder(unique_storage_folder)

        qdrant_doc = initialize_qdrant(chunks_for_doc, storage_path=unique_storage_folder)
        found_docs = qdrant_doc.similarity_search_with_score(question)

        if found_docs:
            # Get the highest scoring document from the documents sorted by score
            found_docs.sort(key=lambda x: x[1], reverse=True)
            document, score = found_docs[0]

            if best_score is None or score > best_score:
                best_score = score
                best_response = document.page_content
                best_document_name = document_name

    if best_response:
        template = """According to {document_name}, the answer is:
        {context}

        Question: {question}

        Helpful Answer:"""

        prompt = PromptTemplate.from_template(template)
        result_prompt = prompt.invoke({"document_name": best_document_name, "context": best_response, "question": question})

        response = llm.invoke(result_prompt)

        print(f"Response from {best_document_name}:\n{response.content}\n")
    else:
        print("No relevant documents found.")

if __name__ == "__main__":
    main()
