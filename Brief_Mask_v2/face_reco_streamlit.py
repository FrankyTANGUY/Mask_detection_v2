from asyncio.windows_events import NULL
from dataclasses import dataclass
from genericpath import exists
from inspect import Traceback
from logging import warning
import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
from datetime import datetime
import pandas as pd
import xlsxwriter
import streamlit as st
from webcam import webcam
import keras


face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	count=0
	color = (255, 255, 255)
	liste_jour=[]
	liste_heure=[]
	liste_personne=[]
	liste_mask=[]
	data=[]
	tableau=[]
	model = keras.models.load_model('./Mask.h5')
	fileName = "vision.csv"
	if (os.path.exists(fileName)==False): 
		tableau = pd.DataFrame(tableau, columns=["Personne", "Date", "Heure", "Statut"])
		tableau.to_csv("vision.csv", index=False)
		report = pd.read_csv('vision.csv')
		tableau = list(report.values)
	report = pd.read_csv('vision.csv')
	tableau = list(report.values)
	l=-1
	for (x, y, w, h) in faces:	
		l=l+1
		img_trimmed = img[y:y + h, x:x + w]
		b, g, r = cv2.split(img_trimmed)
		img_trimmed = cv2.merge([r, g, b])
		image_a_tester=cv2.resize(img_trimmed, (224,224))
		image_test= np.expand_dims(image_a_tester,axis=0)
		predictions = model.predict(image_test)
		label_predit=np.argmax(predictions)
		if label_predit==0 :
			message = "No Mask"
		else :
			message = "Mask"
		count+=1
		cv2.rectangle(img, (x, y), (x+w, y+h), color)
		cv2.rectangle(img, (x, y - 20), (x + w, y), color,-1)
		cv2.putText(img,str(count)+" "+message, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
		date_now = datetime.now().strftime('%d/%m/%Y')
		heure_now = datetime.now().strftime('%H:%M:%S')
		tableau.append([f"Personne {l+1}", date_now, heure_now, message])		
	tableau = pd.DataFrame(tableau, columns=["Personne", "Date", "Heure", "Statut"])
	print(tableau)
	tableau.to_csv("vision.csv", index=False)
	return img,faces,count, tableau


def main():
	"""Face Detection App"""
	st.title("Appli de détection faciale")
	st.text("Construite avec Streamlit et OpenCV")

	activities = ["Détection","Live","À propos"]
	choice = st.sidebar.selectbox("Selection de l'activité",activities)

	if choice == 'Détection':
		st.subheader("Face Detection")
		image_file = st.file_uploader("Charger une image",type=['jpg','png','jpeg'])
		if image_file is None:
			st.text("En attente d'une image")
		if image_file is not None:
			our_image = Image.open(image_file)
			st.text("Image Originale")
			# st.write(type(our_image))
			st.image(our_image)
			
		enhance_type = st.sidebar.radio("Type de l'image",["Original","Echelle de gris"])
		if enhance_type == 'Echelle de gris':
			new_img = np.array(our_image.convert('RGB'))
			img = cv2.cvtColor(new_img,1)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# st.write(new_img)
			st.image(gray)
           
		elif enhance_type == 'Original':
			st.image(our_image,width=300)
		else:
			st.image(our_image,width=300)

		if st.button("Process"):
			result_img,result_faces,nb_visages,tableau = detect_faces(our_image)
			st.image(result_img)
			st.text("Nombre de visage détectés :")
			st.text(str(nb_visages))
			st.table(tableau)

	elif choice == 'À propos':
		st.subheader("À propos de l'appli")
		st.markdown("Construite avec Streamlit par Franky TANGUY")
		st.text("Franky TANGUY")
		st.success(":)")

	elif choice == 'Live':
		cam = webcam()
		st.write("Got an image from the webcam:")
		result_img,result_faces,nb_visages = detect_faces(cam)
		st.image(result_img)
		st.text("Nombre de visage détectés :")
		st.text(str(nb_visages))

if __name__ == '__main__':
		main()