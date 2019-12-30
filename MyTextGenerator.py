# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 20:27:11 2019

@author: sylwi
"""
########PART 1 - Training Daten für RNN vorbereiten 
import numpy as np
# kann man beliebigen Text File hinzufügen...
with open ("text.txt", "r") as file:
    contents = file.read() 
    
contents = "\n".join(contents.split("\n"))

print(contents)
len(contents)
len(set(contents))

unique_chars = set(contents)

int_to_char ={}
char_to_int ={}

#Buchstaben in Zahlen umwandeln 
for i, j in enumerate(unique_chars):
    
    int_to_char[i]= j
    char_to_int[j]= i
    
print(int_to_char)    
print(char_to_int)

len(unique_chars)
#60 - also wir haben 60 Categories 

#hier definieren wir was wird benötigt um die Vorhersage zu machen (X), und was wird als Vorhersage ausgegeben (Y)
length = 40
X=[]
Y=[]
#wie gehen den ganzen text durch und defieniren X und auch Y 
#wir nehmmen in dem Fall immer 40 Buchstaben um die Vorhersage zu machen 
for i in range (0, len(contents)-40): 
     line = contents[i:i+length]
     #jede Buchstabe (l) vom line wird ins int umgewandelt
     X.append([char_to_int[l] for l in line])
     letter = contents[i+length]
     Y.append([char_to_int[l] for l in letter])
    
   
print(X[:4])
print(Y[:4])    

#one-hot encoding 
from keras.utils import to_categorical
#to_categorical übergeben wir 2 Argumenten (samples, die Anzahl von Kategorien, also die Länge von dem Set mit Buchstaben

X = to_categorical(X, num_classes =len(unique_chars))
Y = to_categorical(Y, num_classes = len(unique_chars))

X.shape 

################PART 2 - RNN Training 
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, input_shape=(10, 60)))
model.add(Dense(60, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy") 
model.fit(X, Y, batch_size=32, epochs=3)

#ModelCheckpoint to save weights from Recurrent Neural Networks
from tensorflow.keras.callbacks import ModelCheckpoint
#ModelCheckpoint? to get the argument: {epoch:02d}
save_model=ModelCheckpoint("weights.{epoch:02d}.hdf5")
model.fit(X, y, batch_size=32, epochs=10, callbacks=[save_model])

############################PART 3 Training Model wieder herstellen
#mapping für Testing-Daten
import pickle

with open ("char_to_int.pickle", "rb") as file:
    char_to_int = pickle.load(file)
    
with open ("int_to_char.pickle", "rb") as file:
    int_to_char=pickle.load(file)

with open("text.txt", "r") as file:
    contents = file.read()    
    # wenn denn Text zu groß ist, können wir hier den Bereich in Text definieren 
# zum Beispiel: 
# contents = "\n".join(contents.split("\n")[0:1000])

#das Model wieder herstellen
from keras.models import load_models
model.load_model("weights.hdf5")
#model.summary()
 
############################PART 4 Daten für testing vorbereiten
#mein Model braucht 40 Ziffern, um die nächste Zahl/Buchstabe vorherzusagen 
line = contents[:40]
# jetzt muss ich das in Ziffern umwandeln mit List Comprehensions 
transfomed_line = [char_to_int[l] for l in line]
# one-hot encoding 
from keras.utils import to_categorical 
transfomed_line = to_categorical(transformed_line, num_classes = len(unique_chars))


###############################PART 5 Prediction

prediction = model.predict(transformed_line.reshape(1,40,60))

#wir reduzieren eine dimension, eine KlammerPaar weg 
prediction[0]

# jetzt werwenden wir argmax, das gibt uns zurück an welche Position die höchste Zahl steht
import numpy as np 
#Das generiert nur die nächste Buchstabe
np. argmax(prediction[0])
# hier habe ich auch den Parameter end geändert, damit da kein Zeilenumbruch ist
print(line, end ="")

#die Zahl die diese Function zurückgibt, kann man in Buchstabe umwandeln mit dem Dictionary int_to_char
# Beispiel: 19 
# int_to_char[19]
#oder int_to_char[np.argmax(prediction[0])]

################################################################Text Generieren

line_transformed = to_categorical(line_transformed, num_classes= len(char_to_int))
for i in range(0,100):
    #als ertse hole ich aus dem Model meine Prediction 
    prediction = model.predict(line_transformed.reshape(1,40,60))
    # Das Model kann auf mehrere Beispiele angewandt werden,ich will aber nur ein Bespiel schätzen, deswegen brauche ich die verschachtelte Array Struktur nicht, und hole nur die erste Prediction 
    prediction = prediction[0]
    
    # jetzt hole ich den wahrscheinlichsten Wert daraus
    beste_position =np.random.choice(60,1, p=prediction)[0]
    # optional: als char alles ausgeben
    # print macht immer Zeilen umbruch weil der ein Parameter hat end="\n", um das zu elimieniern,habe ich den Parameter geändert auf nichts
    print(int_to_char[beste_position], end="")
    
    #jetzt will ich das line_transformed ändern, den ersten Element überspringen
    # und den Text der vorhergesagt war,am Ende von den line_transformed kleben
    #erste Zeile überspirngen, und alle Spalten hinfügen
    line_transformed[1:,:]
    #jetzt will ich den Zahl von beste_position one_hot entcoding, und ich ins Liste packen damit den Shape da passt..Alternative kann man auch reshape function verwenden..
    new_text = to_categorical([beste_position], num_classses= len(char_to_int))
    line_transformed= np.append( line_transformed, new_text, axis=0)
   
    
