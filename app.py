import streamlit as st
from PIL import Image
import base64
import spacy
import pandas as pd
import cv2 as cv
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import textwrap


file_prefix = './'

@st.cache_data(persist=True)
def load_data():
    data = pd.read_excel(file_prefix + 'Preprocessed_data.xlsx')
    return data


@st.cache_data(persist=True)
def load_model():
    model = spacy.load(file_prefix + "model-best")
    return model


@st.cache_data(persist=True)
def getImageAsBase64(file):
  with open(file, "rb") as f:
    data = f.read()
  return base64.b64encode(data).decode()


def pytes(image):
  result = pytesseract.image_to_string(image)
  result = ' '.join(result.split())
  result = result.replace('  ', ' ').replace('   ', ' ').replace('    ', ' ')
  return result


def wrapText(width, text, image, x, y, gap):
  wrapper = textwrap.TextWrapper(width=width)
  word_list = wrapper.wrap(text=text)
    
  for words in word_list:
    img_out = cv.putText(img = image, text = words, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)

    if words == word_list[-1]:
      y += gap
    else:
      y += 38
  return image, y


def callback():
  st.session_state.Button = True


st.set_page_config(page_title="Automated Claim Processing",page_icon="🔎",layout="wide",initial_sidebar_state="expanded")

with open(file_prefix + "style.css") as f:
  st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

img = getImageAsBase64(file_prefix + "background image.jpg")
st.markdown(f"""
  <style>
    [data-testid="stAppViewContainer"]{{
      background-image: url("data: image/png;base64,{img}");
      background-size: cover;
    }}
  </style>""",unsafe_allow_html=True)

_,col,_ = st.columns(3)
col.image(file_prefix + "logo.png", use_column_width='auto')

st.write("""<h1 style='text-align: center; font-size: 13mm; color: rgba(0,0,0,0.9);'>Automated Life Insurance Claim Process</h1>
<p style='text-align: center; font-size: 6mm; color: rgba(0,0,0,0.7)'><b>Created by :- M.H.M. HISHAM <font size="3">( BSc. (Hons) in Data Science )</font></br>
<font size="3">Faculty of Science</br>University of Peradeniya</font></b></p>
""", unsafe_allow_html=True)


nlp_ner = load_model()
df = load_data()

img = st.file_uploader("Drag and drop the CLAIM FORM : ",accept_multiple_files=False, type = ['png','jpg'])

if 'Button' not in st.session_state:
      st.session_state.Button = None

if st.button('Submit',on_click=callback) or st.session_state.Button:
  if img is not None:
    img = Image.open(img)
    img = np.array(img)
    col1, col2, col3 = st.columns(3)
    col2.image(img)


    input = pytes(img)
    doc = nlp_ner(input)

    Policy_no = None
    Request_date = None
    Doctor_name = None
    Hospital_name = None
    Amount = None
    Types = {}
    Types_in_claim_form = []


    policy_number_temp = []
    for entity in doc.ents:
      if entity.label_ == 'POLICY_NO':
        policy_number_temp.append(entity.text.replace('NGTO', 'NGT0').replace('L14', 'LI4'))
        Policy_no = '-'.join(policy_number_temp)
      elif entity.label_ == 'DATE':
        Request_date = entity.text
      elif entity.label_ == 'DOCTOR':
        Doctor_name = entity.text
      elif entity.label_ == 'HOSPITAL':
        Hospital_name = entity.text
      elif entity.label_ == 'AMOUNT':
        Amount = int(entity.text)
      else:
        if entity.label_ not in Types:
          Types[f'{entity.label_}'] = []
        if entity.label_ not in Types_in_claim_form:
          Types_in_claim_form.append(entity.label_)
        Types[f'{entity.label_}'].append(entity.text)


    temp_df = df.iloc[list(df['PolicyNo'] == Policy_no)]
    Types_in_DB = temp_df.groupby('TYPE').size().index.tolist()


    registered_types = []
    unregistered_types = []
    for typ in Types_in_claim_form:
      if typ in Types_in_DB:
        registered_types.append(typ)
      else:
        unregistered_types.append(typ)
        continue
    img_out = cv.imread(file_prefix+'claim details.png')
    img_out = cv.cvtColor(img_out, cv.COLOR_BGR2RGB)

    x = 130
    y = 650
    fontface = cv.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    clr = (0, 0, 0)
    thick = 2
    gap = 80

    col1, col2, col3 = st.columns(3)
    st.write(f"""<div class = "sheet1"><p class = "title"><b><u> &nbsp;CLAIM DETAILS&nbsp; </u></b></p>
    </br></br>
    <p class = "details"><b>(1)&ensp;&nbsp;POLICY NUMBER :- </b>{Policy_no}</br></br>
    <b>(2)&ensp;&nbsp;ACCIDENT DATE :- </b>{Request_date}</br></br>
    <b>(3)&ensp;&nbsp;DOCTOR NAME   :- </b>{Doctor_name}</br></br>
    <b>(4)&ensp;&nbsp;HOSPITAL NAME :- </b>{Hospital_name}</br></br>
    <b>(5)&ensp;&nbsp;REQUESTED AMOUNT :- </b>Rs.{Amount}</br></br>
    <b>(6)&ensp;&nbsp;CLAIM DETAILS :- </b></p></div>""",unsafe_allow_html=True)
    line_1 = f'(1) POLICY NUMBER :- {Policy_no}'
    img_out = cv.putText(img = img_out, text = line_1, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
    y += gap
    line_2 = f'(2) ACCIDENT DATE  :- {Request_date}'
    img_out = cv.putText(img = img_out, text = line_2, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
    y += gap
    line_3 = f'(3) DOCTOR NAME   :- {Doctor_name}'
    img_out = cv.putText(img = img_out, text = line_3, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
    y += gap
    line_4 = f'(4) HOSPITAL NAME  :- {Hospital_name}'
    img_out = cv.putText(img = img_out, text = line_4, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
    y += gap
    line_5 = f'(5) AMOUNT         :- Rs.{Amount}'
    img_out = cv.putText(img = img_out, text = line_5, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
    y += gap
    line_6 = '(6) CLAIM DETAILS   :- '
    img_out = cv.putText(img = img_out, text = line_6, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
    y += (gap-30)

    claiming_coverages = []
    for key, val in Types.items():
      st.write(f'''<div class = "sheet2"><b>{key} :- </b>{', '.join(val)}</div>''',unsafe_allow_html = True)
      temp = f"{key} :- {', '.join(val)}"
      claiming_coverages.append(temp)

    x = 250
    for tmp in claiming_coverages:
      img_out, y = wrapText(57, tmp, img_out, x, y, 55)
      if tmp == claiming_coverages[-1]:
        y += 25

    exp_date = temp_df.groupby('EXPIRYDATE').size().index[0]
    exp_date = pd.to_datetime(exp_date, dayfirst = True)
    Request_date = pd.to_datetime(Request_date, dayfirst = True)

    if Request_date > exp_date:
      st.write(f'''<div class = "sheet3"><b>(7)&ensp;&nbsp;CLAIM STATUS :- </b><span class = "red">REJECTED</span></br></br>
      <b>(8)&ensp;&nbsp;REJECTION DETAILS :- </b>All your life insurance coverages have <span class = "red">expired.</span></div>''',unsafe_allow_html=True)
      line_7 = '(7) CLAIM STATUS :- REJECTED'
      x = 130
      img_out = cv.putText(img = img_out, text = line_7, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
      y += gap
      line_8 = '(8) REJECTION DETAILS :- All your life insurance coverages have expired.'
      img_out, y = wrapText(69, line_8, img_out, x, y, gap)

    else:
      if len(registered_types) != 0:
        total_suminsured = 0
        for typ in registered_types:
          temp_amount = sum(temp_df.iloc[list(temp_df['TYPE'] == typ)]['SUMINSURED'].tolist())
          total_suminsured += temp_amount
        if len(unregistered_types) != 0:
          if total_suminsured <= Amount:
            st.write(f'''<div class = "sheet3"><b>(7)&ensp;&nbsp;CLAIM STATUS :- </b><span class = "green">ACCEPTED</span></br></br>
            <b>(8)&ensp;&nbsp;PAYMENT NOTE :- </b></br><div class = "temp">Dear valuable {Policy_no} policy holder, you are <span class = "red">not registered</span> for {", ".join(unregistered_types)} coverages. 
                        Therefore we <span class = "green">can only pay</span> for your {", ".join(registered_types)} coverages. We <span class = "red">cannot pay Rs.{Amount}.</span> We are <span class = "green">obligated to pay Rs.{int(total_suminsured)}</span> only.</div></br>
                        <b>(9)&ensp;&nbsp;ACCEPTED PAYING AMOUNT :- </b>Rs.{int(total_suminsured)}</div>''',unsafe_allow_html=True)
            line_7 = '(7) CLAIM STATUS :- ACCEPTED'
            x = 130
            img_out = cv.putText(img = img_out, text = line_7, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
            y += gap
            line_8 = f'(8) PAYMENT NOTE :- Dear valuable {Policy_no} policy holder, you are not registered for {", ".join(unregistered_types)} coverages. Therefore we can only payfor your {", ".join(registered_types)} coverages. We cannot pay Rs.{Amount}. We are obligated to pay Rs.{int(total_suminsured)} only.'
            img_out, y = wrapText(69, line_8, img_out, x, y, gap)
            line_9 = f'(9) ACCEPTED PAYING AMOUNT :- Rs.{int(total_suminsured)}'
            img_out, y = wrapText(65, line_9, img_out, x, y, gap)
          else:
            st.write(f'''<div class = "sheet3"><b>(7)&ensp;&nbsp;CLAIM STATUS :- </b><span class = "green">ACCEPTED</span></br></br>
            <b>(8)&ensp;&nbsp;PAYMENT NOTE :- </b></br><div class = "temp">Dear valuable {Policy_no} policy holder, you are <span class = "red">not registered</span> for {", ".join(unregistered_types)} coverages. Therefore we can only pay for your {", ".join(registered_types)} coverages. 
                        </span> We are <span class = "green">obligated to pay Rs.{int(Amount)}</span> only.</div></br>
                        <b>(9)&ensp;&nbsp;ACCEPTED PAYING AMOUNT :- </b>Rs.{int(Amount)}''',unsafe_allow_html=True)
            line_7 = '(7) CLAIM STATUS :- ACCEPTED'
            x = 130
            img_out = cv.putText(img = img_out, text = line_7, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
            y += gap
            line_8 = f'(8) PAYMENT NOTE :- Dear valuable {Policy_no} policy holder, you are not registered for {", ".join(unregistered_types)} coverages. Therefore we can only pay for your {", ".join(registered_types)} coverages. We are obligated to pay Rs.{int(Amount)} only.'
            img_out, y = wrapText(69, line_8, img_out, x, y, gap)
            line_9 = f'(9) ACCEPTED PAYING AMOUNT :- Rs.{int(Amount)}'
            img_out, y = wrapText(65, line_9, img_out, x, y, gap)
        else:
          if total_suminsured <= Amount:
            st.write(f'''<div class = "sheet3"><b>(7)&ensp;&nbsp;CLAIM STATUS :- </b><span class = "green">ACCEPTED</span></br></br>
            <b>(8)&ensp;&nbsp;PAYMENT NOTE :- </b></br><div class = "temp">Dear valuable {Policy_no} policy holder, you're sum insured coverage <span class = "red">limit is reached.</span> Therefor we <span class = "red">cannot pay Rs.{Amount}.</span> 
            We are <span class = "green">obligated to pay Rs.{int(total_suminsured)}</span> only.</div></br>
            <b>(9)&ensp;&nbsp;ACCEPTED PAYING AMOUNT :- </b>Rs.{int(total_suminsured)}''',unsafe_allow_html=True)
            line_7 = '(7) CLAIM STATUS :- ACCEPTED'
            x = 130
            img_out = cv.putText(img = img_out, text = line_7, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
            y += gap
            line_8 = f"(8) PAYMENT NOTE :- Dear valuable {Policy_no} policy holder, you're sum insured coverage limit is reached. Therefor we cannot pay Rs.{Amount}. We are obligated to pay Rs.{int(total_suminsured)} only."
            img_out, y = wrapText(69, line_8, img_out, x, y, gap)
            line_9 = f'(9) ACCEPTED PAYING AMOUNT :- Rs.{int(total_suminsured)}'
            img_out, y = wrapText(65, line_9, img_out, x, y, gap)
          else:
            st.write(f'''<div class = "sheet3"><b>(7)&ensp;&nbsp;CLAIM STATUS :- </b><span class = "green">ACCEPTED</span></br></br>
            <b>(8)&ensp;&nbsp;PAYMENT NOTE :- </b></br><div class = "temp">Dear valuable {Policy_no} policy holder, We are <span class = "green">obligated to pay Rs.{int(Amount)}.</span></div></br>
            <b>(9)&ensp;&nbsp;ACCEPTED PAYING AMOUNT :- </b>Rs.{Amount}''',unsafe_allow_html=True)
            line_7 = '(7) CLAIM STATUS :- ACCEPTED'
            x = 130
            img_out = cv.putText(img = img_out, text = line_7, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
            y += gap
            line_8 = f'(8) PAYMENT NOTE :- Dear valuable {Policy_no} policy holder, We are obligated to pay Rs.{int(Amount)}.'
            img_out, y = wrapText(69, line_8, img_out, x, y, gap)
            line_9 = f'(9) ACCEPTED PAYING AMOUNT :- Rs.{Amount}'
            img_out, y = wrapText(65, line_9, img_out, x, y, gap)
      else:
        st.write(f'''<div class = "sheet3"><b>(7)&ensp;&nbsp;CLAIM STATUS :- </b><span class = "red">REJECTED</span></br></br>
        <b>(8)&ensp;&nbsp;REJECTION DETAILS :- </b></br><div class = "temp">Dear valuable {Policy_no} policy holder, you are <span class = "red">not registered for all of the coverages</span> you have requested.</div></br>
        <b>(9)&ensp;&nbsp;UNREGISTERED COVERAGES :- </b></br><div class = "temp">{", ".join(unregistered_types)}</div></br>
        <b>(10) REGISTERED COVERAGES :- </b></br><div class = "temp">{', '.join(Types_in_DB)}</div>''',unsafe_allow_html=True)
        line_7 = '(7) CLAIM STATUS :- REJECTED'
        x = 130
        img_out = cv.putText(img = img_out, text = line_7, org = (x, y), fontFace = fontface, fontScale = fontscale, color = clr, thickness = thick)
        y += gap
        line_8 = f'(8) REJECTION DETAILS :- Dear valuable {Policy_no} policy holder, you are not registered for all of the coveragesyou have requested.'
        img_out, y = wrapText(69, line_8, img_out, x, y, gap)
        line_9 = f'(9) UNREGISTERED COVERAGES :- {", ".join(unregistered_types)}'
        img_out, y = wrapText(65, line_9, img_out, x, y, gap)
        line_10 = f'(10)REGISTERED COVERAGES :- {", ".join(Types_in_DB)}'
        img_out, _ = wrapText(65, line_10, img_out, x, y, gap)
    st.write(" ")
    img_out = cv.cvtColor(img_out, cv.COLOR_BGR2RGB)
    _, img_out = cv.imencode(".jpg", img_out)
    img_out = img_out.tobytes()
    btn = st.download_button(label="Download invoice", data=img_out, file_name="Claim invoice.png", mime="image/png",)
