import streamlit as st
from PIL import Image
import os
import torch
from torch import tensor

import matplotlib.pyplot as plt
from detecto.core import DataLoader, Model
import matplotlib.pyplot as plt
from detecto.utils import read_image

from detecto import core, utils, visualize

calorie_dict={"dosa":"calories: 160 , carbs: 29g , protein: 3.9g , fat: 3.7g (per piece)",
              "idly":"calories: 35 , carbs: 8g , protein: 2.0g , fat: 0.1g , fibre: 0.3g (per piece)",
              "samosa":"calories: 260 , carbs: 24g , protein: 3.5g , fat: 17g , fibre: 2.1g (per piece)",
              "puri":"calories: 101 , carbs: 12g , protein: 2g , fat: 5g , fibre: 1.8g (per piece)",
              "curd_rice":"calories: 370 , carbs: 36g , protein: 8.5g , fat: 18.7g , fibre: 1.8g (per serving)",
              "vada":"calories: 97 , carbs: 8.9g , protein: 2g , fat: 5.2g , fibre: 1.8g (per piece)"}




from detecto import core, utils, visualize

def filter_prediction(labels,boxes,scores,filter_value):
       predicted = zip(labels, boxes.tolist(), scores.tolist())
       filtered_prediction = []

       for x in predicted:
              print(x[2])
              if (x[2] >= filter_value/100):
                     filtered_prediction.append(x)

       labels, boxes, scores = zip(*filtered_prediction)
       boxes = torch.tensor((boxes))
       scores = torch.tensor(scores)

       return labels,boxes,scores

def detecto_m(pic,filter_value):
       image = utils.read_image(pic)
       your_labels = ['dosa', 'idly', 'samosa', 'puri', 'curd_rice', 'vada']
       model =core.Model.load((directory+r"/models/model_weights_30_epoch.pth"),your_labels)
      # model = core.Model()
       labels, boxes, scores = model.predict_top(image)
       #st.markdown(labels)
       #st.markdown(boxes)
       #st.markdown(scores)
       labels, boxes, scores=filter_prediction(labels,boxes,scores,filter_value)

       result = visualize.show_labeled_image(image, boxes, labels)
       return result,labels

st.markdown("# Nutrition Detector")
st.markdown("#### A nutrition detector for indian foods")
st.markdown("Made by Mohamed Akif , Hiruthik Vigasin")
filter_value=st.slider("Accuracy Filter",0,100,60)
# getting the image from the user
uploaded_file = st.file_uploader("Choose an pic", type="jpg")
directory = os.getcwd()
if uploaded_file is not None:
       st.image(uploaded_file, caption='Your Image.',         use_column_width=True)

       with open(os.path.join("temp", uploaded_file.name), "wb") as f:
              f.write(uploaded_file.getbuffer())
       st.success("Saved File")
       st.success("Running.... please wait")

       for filename in os.scandir("temp"):
              if filename.is_file():
                     pass
                     #st.markdown(filename.path)

       result,final_labels = detecto_m(filename.path,filter_value)


       os.remove(filename.path)# passing image to our function

       st.set_option('deprecation.showPyplotGlobalUse', False)
       result = plt.plot() # plotting the result
       st.pyplot(result)
       for x in final_labels:
              st.markdown(x.capitalize()+" : "+calorie_dict[x])
       #st.balloons()
