from tensorflow.keras import models # pyright: ignore[reportMissingImports]
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import cv2  
import io


st.set_page_config(
        page_title='Image Classification',
        page_icon='🖼️',
        layout='wide'
)


model = models.load_model('intel-object-classification.keras')



with st.container():

    st.markdown("<h1 style='color:white;text-align:center'> 🖼️ Intel Image Classification</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='color:#D3D3D3;text-align:center;'>Detect probability of existence for these 6 Categories:</h3>",
        unsafe_allow_html=True
    )
    items_with_icons = {
        "buildings": "🏢", "forest": "🌲",
        "glacier": "❄️", "mountain": "⛰️",
        "sea": "🌊", "street": "🛣️"
    }
    labels = list(items_with_icons.keys())
    # st.write(labels)

    icons_col = st.columns(len(list(items_with_icons.keys()))+2)
    for index, (label, icon) in enumerate(items_with_icons.items()):
        icons_col[index+1].markdown(f"<div style='text-align:center; font-size:25px;'>{icon}<br>{label.capitalize()}</div>", unsafe_allow_html=True)

st.divider()


st.markdown('<br>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider('Prediction threshold showing:', 1, 99, 20) / 100
st.sidebar.markdown("<h4 style='color:white;font-size:20px;'>Developed by:<br><b style='font-size:16px;color:yellow'>Mohammad Ali Shaghaghian</b></h4>", unsafe_allow_html=True)


st.subheader('📤 Upload Image')
image = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
with st.container():
    if image is not None:
        try:
            img_bytes = image.getvalue()
            img_pil=Image.open(io.BytesIO(img_bytes))
            img_pil = ImageOps.exif_transpose(img_pil).convert('RGB')

        except UnidentifiedImageError:
            st.error("The upload file isn't a valid image or is corrupted")
            st.stop()
        
        _, center_col, _ = st.columns([1, 3, 1])  # create side spacing
        with center_col:  # all content inside here will be centered in wide layout
            width , height= img_pil.width, img_pil.height

            if width < 300 or height < 300: # check if the width or height of image is less than 300
                img_show = img_pil.resize((300, 300))
                _, inner_col, _ = st.columns([1, 2, 1])
                inner_col.caption(f'Image original size: {img_pil.size} ; Resized to (300, 300)')
                inner_col.image(img_show, caption='Uploaded Imaged')

            else:
                st.image(image, caption='Uploaded Image') 


            img_array = np.array(img_pil) 
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_resize = cv2.resize(img_bgr, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            img_scale = img_resize.astype(np.float32) / 255.
            img_expand= np.expand_dims(img_scale, axis=0)


            predict_prob = model.predict(img_expand)[0]
            results = [(labels[index], round(float(predict_prob[index]), 6)) for index in range(len(labels))]

            shown_number = sum([pred[1]>threshold for pred in results])    
            cols = [1, 1]
            for i in range(shown_number):
                cols.insert(1, 3)
            cols_index = list(range(len(cols)))

            col_tab = st.columns(cols)
            cols_iterator = iter(cols_index)
            next(cols_iterator)


            for (label, prob) in results:
                if prob > threshold:
                    col_tab[next(cols_iterator)].metric(f'**{label.capitalize()}**', f'{round(prob * 100, 4)} %')
        
st.markdown('<br>', unsafe_allow_html=True)
