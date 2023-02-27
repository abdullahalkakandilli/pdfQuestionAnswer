import re
from io import StringIO
import pandas as pd
import streamlit as st
import requests
import pdfminer
from transformers import AutoTokenizer, BertForQuestionAnswering
import torch
from functionforDownloadButtons import download_button
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

API_URL = "https://api-inference.huggingface.co/models/bert-large-uncased-whole-word-masking-finetuned-squad"
headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}




def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}

    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="images/icon.png", page_title="PDF Question Answering")


c2, c3 = st.columns([6, 1])

with c2:
    c31, c32 = st.columns([12, 2])
    with c31:
        st.caption("")
        st.title("Question to PDF File")
    with c32:
        st.image(
            "images/logo.png",
            width=200,
        )

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
uploaded_file = st.file_uploader(
    " ",
    type="pdf",
    key="1",
    help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",

)
result_ = ""
if uploaded_file is not None:
    output_string = StringIO()

    parser = PDFParser(uploaded_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)

    # st.write(merged_text)
    pdf_text_result_ = output_string.getvalue()

def get_values(question_input):
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": {
            "question": question_input,
            "context": pdf_text_result_
        },
    })

    return output


form = st.form(key="annotation")
result = ""
with form:
    question_input = st.text_input("Enter your query here")

    submitted = st.form_submit_button(label="Submit")

result_df = pd.DataFrame()
if submitted:

    result = get_values(question_input)


c4, c5 = st.columns([6, 1])
original_title = f'<p style="font-family:Courier; color:Green; font-size: 20px;">Your answer will appear here!</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.write(result)