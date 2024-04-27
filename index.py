import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader


question = "How long do you think it will take to recover from the earthquake?"


os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")



loader = PyPDFLoader("./test.pdf")
pages = loader.load_and_split()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings()


qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    path="./local_qdrant",
    collection_name="my_documents",
)


found_docs = qdrant.similarity_search_with_score(question)


document, score = found_docs[0]



template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""


prompt = PromptTemplate.from_template(template)
resultPrompt = prompt.invoke({"context":document.page_content, "question":question})

print(llm.invoke(resultPrompt).content)