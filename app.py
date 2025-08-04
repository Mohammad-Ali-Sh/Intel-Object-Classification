import streamlit as st
from tensorflow.keras import models
from PIL import Image
import cv2
import numpy as np

model = models.load_model('intel_image_classification_model.h5')

st.title('Intel Image Classification')
st.set_page_config(page_title='Image Classification', page_icon='')

st.subheader('This model detect the possibly of existence:')
items_with_icons = {
    "building": "🏢", "forest": "🌲",
    "glacier": "❄️", "mountain": "⛰️",
    "sea": "🌊", "street": "🛣️"
}
_, items_col = st.columns([1, 10])
for item, icon in items_with_icons.items():
    items_col.markdown(f'**{icon} - {item.capitalize()}**')


image = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
if image is not None:
    img_pil = Image.open(image).convert('RGB')

    if img_pil.size[0] < 300 or img_pil.size[1] < 300:
        img_show = img_pil.resize((300, 300))
        st.write(f'Your image actual size is: ({(img_pil.size[0], img_pil.size[1])} ; Converted to (300, 300))')
        _, center_col, _ = st.columns([1, 2, 1])
        center_col.image(img_show, caption='Uploaded Imaged')

    else:
        st.image(image, caption='Uploaded Image') 



    img_arr = np.array(img_pil) 
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    img_r = cv2.resize(img_bgr, dsize=(150, 150))
    img_sc = img_r / 255.
    img_inp = np.expand_dims(img_sc, axis=0)

    threshold = st.number_input('Prediction threshold showing:', 1, 99, 20) / 100
    predict_prob = list(model.predict(img_inp)[0])
    classes = list(items_with_icons.keys())
    shown_number = sum([pred>threshold for pred in predict_prob])    
    cols = [1, 1]
    for i in range(shown_number):
        cols.insert(1, 3)
    cols_index = list(range(len(cols)))

    col_tab = st.columns(cols)
    cols_iterator = iter(cols_index)
    next(cols_iterator)

    
    for index, prob in enumerate(predict_prob):
        if prob > threshold:
            col_tab[next(cols_iterator)].metric(f'**{classes[index].capitalize()}**', f'{round(float(prob), 6)*100} %')
