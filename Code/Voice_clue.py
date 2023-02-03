import os
import sys
import cohere
from PIL import Image
from streamlit.web import cli as stcli
from streamlit import runtime
import streamlit as st
from gtts import gTTS
from dotenv import load_dotenv
load_dotenv()
def describe(keyword):
  co = cohere.Client(os.environ.get("COHERE_CLIENT"))
  response = co.generate(
    model='command-xlarge-20221108',
    prompt=f"Write a paragraph about {keyword} ",
    max_tokens=200,
    temperature=0.5,
    k=0,
    p=0.75,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop_sequences=["--"],
    return_likelihoods='NONE')
  
  print('Prediction: {}'.format(response.generations[0].text))
  return response.generations[0].text


def main():
  img=Image.open("/ICON.jpg")
  st.image(img)
  st.title("Voice CLUE")
  form = st.form(key="user_settings")
  with form:
      kw = st.text_input("English Keyword",placeholder="Enter the word to describe",key = "en_keyword")
      req_tld=st.selectbox("Choose an accent",["com.au","co.uk","us","ca","co.in","ie","co.za"])
      st.write("* Australian English - com.au\n* British English - co.uk\n* American English - us\n* Canadian English - ca\n* Indian English - co.in\n* Irish English - ie\n* South African English - co.za")
      submit = form.form_submit_button("Generate the description")
      if submit:
        cohere_txts = describe(kw)
        myobj = gTTS(text=cohere_txts,lang='en', slow=False, tld=req_tld)
        myobj.save("output.mp3")
        os.system("output.mp3")
        st.markdown(f"Generated Description : ")
        st.write(myobj.text)  
if __name__ == "__main__":
  if runtime.exists():
    main()
  else:
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())
