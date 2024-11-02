from os import environ
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import pdfplumber
import pandas as pd
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox
from langchain_community.document_loaders import PDFPlumberLoader

load_dotenv()

def process_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    all_text = []

    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table)
            df.columns = df.iloc[0]
            markdown = df.drop(0).to_markdown(index=False)
            print(markdown)

# Path to your PDF file
pdf_path = r"./docs/70115e-r9-complete.pdf"
process_pdf(pdf_path)