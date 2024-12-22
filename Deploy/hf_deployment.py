import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import streamlit as st
from PIL import Image

image_size = (150, 150)

def load_model():
    model = tf.keras.models.load_model('./Deploy/Model/model.keras')
    return model

def preprocess_image(img, size):
  image = load_img(img, target_size=size)
  img = img_to_array(image)
  img = img / 255.0
  img = np.expand_dims(img, axis=0)

  return img

def predict_image(model, img, size, label):
  preprocess_img = preprocess_image(img, size)
  predictions = model.predict(preprocess_img)
  predicted_class_idx = np.argmax(predictions)
  predicted_class_label = label[predicted_class_idx]
  confidence = np.max(predictions[0])
    
  return predicted_class_label, confidence

def main():
    # Set judul halaman
    st.title('Klasifikasi Sampah')
    
    # Tambahkan deskripsi
    st.write('Upload foto sampah untuk diklasifikasikan')
    
    label_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Pilih gambar...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        try:
            # Tampilkan gambar yang diupload
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang diupload', width=300)
            
            # Load model
            with st.spinner('Loading model...'):
                model = load_model()
            
            # Lakukan prediksi
            with st.spinner('Melakukan prediksi...'):
                predicted_image, confidence = predict_image(model, uploaded_file, image_size, label_names)
            
            # Tampilkan hasil
            st.success('Prediksi berhasil!')
            st.write(f'Ini adalah: {predicted_image}')
            st.write(f'Tingkat keyakinan: {confidence:.2%}')
            
        except Exception as e:
            st.error(f'Error: {str(e)}')
            
if __name__ == '__main__':
    main()