from os import environ
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader

load_dotenv()

llm_client = OpenAI(api_key=environ['OPENAI_API_KEY'])

def get_hypothetical_questions(table):
    prompt = f"""
    Given the following table and its description and column names:
    Table Description and Content:
    {table}

    Provide a list of exactly three questions that the above table description and content could be used to answer
    """
    response = llm_client.chat.completions.create(
        model="gpt-4o",
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
        model="gpt-4o",
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
    all_text = []
    for page in pdf.pages:
        if page.page_number > 20:
            text = page.extract_text()
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
                        hypothetical_questions = get_hypothetical_questions(table_description)
                        print(hypothetical_questions)
       
# Path to your PDF file
pdf_path = r"./docs/70115e-r9-complete.pdf"
process_pdf(pdf_path)