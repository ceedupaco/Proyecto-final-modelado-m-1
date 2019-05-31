import sys
import keras
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'


with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        cnn = load_model(modelo)


#cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict_classes(x)
  print(array)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Disco")
  elif answer == 1:
    print("pred: earlymerger")
  elif answer == 2:
    print("pred: latemerger")
  return answer

filename=str(sys.argv[1])

if __name__=='__main__':
  predict(filename)  


