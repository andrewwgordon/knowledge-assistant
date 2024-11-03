__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from os import environ
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
import pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import itertools, sys

load_dotenv()

llm_client = OpenAI(
    api_key=environ['OPENAI_API_KEY']
)

def get_hypothetical_questions(table):
    prompt = f"""
    Given the following table and its description and column names:
    Table Description and Content:
    {table}

    Provide a list of exactly three questions that the above table description and content could be used to answer
    """
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides hypothetical questions from a given table."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_table_description(table_content, document_context):
    prompt = f"""
    Given the following table and its context from the original document,
    provide a detailed description of the table and include the table column names. Then, include the table in markdown format.

    Original Document Context:
    {document_context}

    Table Content:
    {table_content}

    Please provide:
    1. A comprehensive description of the table and include the column names.
    2. The table in markdown format.
    """

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes tables and formats them in markdown."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def set_metadata(page):
    filtered_page = page.filter(lambda obj: obj["object_type"] == "char" and obj["size"] > 14)
    filtered_text = filtered_page.extract_text()
    texts = filtered_text.split('\n')
    for text in texts:
        if (text != 'CAUTION') and (text != 'NOTE') and (text != 'WARNING'):
            if text:
                print(text)

def process_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    print(f'Processing {len(pdf.pages)} pages:')
    transformed_document = []
    for page in pdf.pages:
        sys.stdout.write('.')
        sys.stdout.flush()            
        if page.page_number > 20:
            text = page.extract_text()
            transformed_document.append(
                Document(
                    page_content=text,
                    metadata={
                        'source': 'Diamond Aircraft Airplane Flight Manual DA 42 NG',
                        'page': page.page_number
                    }
                )
            )
            tables = page.extract_tables()
            for table in tables:
                first_cell = table[0][0]
                if first_cell:
                    if (not first_cell.startswith('Doc.')) and (not first_cell.startswith('Page')):
                        table_formated = dict()
                        for sub in table:
                            table_formated[tuple(sub[:2])] = tuple(sub[2:])
                        table_description = get_table_description(
                            table_formated,text
                        )
                        transformed_document.append(
                            Document(
                                page_content=table_description,
                                metadata={
                                    'source': 'Diamond Aircraft Airplane Flight Manual DA 42 NG',
                                    'page': page.page_number
                                }
                            )
                        )
    return transformed_document

def split_document(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(document)

def create_vectorstore(split_document):
    vectorstore = Chroma.from_documents(
        persist_directory='./db',
        documents=split_document, 
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    vectorstore.persist()

def pdf_to_vector():
    print('Starting..')
    pdf_path = r"./docs/70115e-r9-complete.pdf"
    print('Processing document...')
    document = process_pdf(pdf_path)
    print('Spliting processed document...')
    document_split = split_document(document)
    print('Creating vector store...')
    create_vectorstore(document_split)

pdf_to_vector()