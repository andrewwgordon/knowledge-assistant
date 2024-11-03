__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from os import environ
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

print('Starting...')

llm = ChatOpenAI(
    model_name='gpt-4o-mini', 
    temperature=0
)

vectorstore = Chroma(
    persist_directory='./db',
    embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)
retriever = vectorstore.as_retriever()

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print('Ready...')
while True:
    question = input('\n>')
    response = rag_chain.invoke(question)
    print(response)