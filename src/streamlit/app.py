import sys
sys.path.insert(0, './../..')

from src.streamlit import preproc

import streamlit as st
import os
import random


#creating the picture list
folder = "../../data/raw/selected_streamlit"
files = os.listdir(folder)
image_files = [f for f in files if f.endswith(('.tif'))]




#creating the menu for the app
st.title("Classification de documents")
st.sidebar.title("Menu")
pages = ["Home", "Datasets", "Méthodologie", "OCR", "Modélisation visuelle", "Modélisation textuelle", "Conclusion"]
page = st.sidebar.radio("Aller vers", pages)

#links
st.sidebar.markdown("""
<style>
.blue-box {
  background-color: #f0f0ff;  
  border: 1px solid blue;
  padding: 10px;
  margin: 5px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="blue-box">
Projet data science mai 2024<br><br>

Participants:<br>

[Chaima Haddoudi](https://www.linkedin.com/in/chaimahaddoudi/)<br>
[Julien Thielleux](https://www.linkedin.com/in/julien-thielleux-6a639648/)<br>
[Quentin Pizenberg](https://www.linkedin.com/in/quentin-pizenberg-a5a11970/)<br>
[Sofiane Louiba](https://www.linkedin.com/in/sofiane-louiba/)
</div>
""", unsafe_allow_html=True)




#Home page
if page == pages[0] : 
    st.write("### Home")
    st.write("Ce projet est issus de la formation data science de DataScientest. Il a pour but de classifier des documents scannés en 16 classes différentes.")
    st.write("Cette démarche vise à simplifier et à accélérer le processus de gestion des documents, en permettant aux utilisateurs de classer automatiquement une grande quantité de documents en un temps record.")
    st.write("Ce streamlit présente les différentes étapes du projet de l'exploration du dataset jusqu'à la modélisation")
    




#Datasets page
if page == pages[1] : 
    st.write("### Datasets")

    st.write("Le dataset utilisé est le RVL_CDIP qui contient 400 000 images de documents scannés en 16 classes différentes.")
    st.write("Les images sont caractérisées par une faible résolution (~100dpi), du bruit et des artefacts de scan.")

    #showing the exemples
    st.write("Voici les classes de documents, cliquer dessus affichera un exemple de cette classe:")
    documents_classes = ["lettre", "formulaire", "email", "manuscrit", "publicité", "rapport scientifique", "publication scientifique", "specification", "dossier", "article de presse", "budget", "facture", "presentation", "questionnaire", "cv", "note de service"]
    picture_dict = {
    "lettre": "0000049717.tif",
    "formulaire": "2505412210.tif",
    "email": "3100800995.tif",
    "manuscrit": "501225219+-5222.tif",
    "publicité": "0000125675.tif",
    "rapport scientifique": "2505601286_1329.tif",
    "publication scientifique": "00399159_9164.tif",
    "specification": "0000054785.tif",
    "dossier": "0000414367.tif",
    "article de presse": "0000240413.tif",
    "budget": "0000167320.tif",
    "facture": "0000037010.tif",
    "presentation": "0000001531.tif",
    "questionnaire": "0000093787.tif",
    "cv": "2501369858.tif",
    "note de service": "0000072858.tif"
}
    
    num_columns = 4
    cols = st.columns(num_columns)

    for i, doc in enumerate(documents_classes):
        if cols[i % num_columns].checkbox(doc):
            cols[i % num_columns].image("../../data/raw/selected_streamlit/" + picture_dict[doc])

    #showing a random picture
    st.markdown("****Afficher une image aléatoire:****  ")
    if st.button("Afficher"):
        file = random.choice(image_files)
        file_path = os.path.join(folder, file)

        #class number to class name dictionnary
        class_name_dict = {
        0: "lettre",
        1: "formulaire",
        2: "email",
        3: "manuscrit",
        4: "publicité",
        5: "rapport scientifique",
        6: "publication scientifique",
        7: "spécification",
        8: "dossier",
        9: "article de presse",
        10: "budget",
        11: "facture",
        12: "présentation",
        13: "questionnaire",
        14: "CV",
        15: "note de service"
        }

        txt_file = os.path.splitext(file)[0] + '.txt'
        txt_file_path = os.path.join(folder, txt_file)
        with open(txt_file_path, 'r') as f:
            class_num = int(f.read().strip())
        document_class = class_name_dict[class_num]

        st.markdown(f"Classe du document: ****{document_class}****")
        st.image(file_path)







#Méthodologie page
if page == pages[2] : 
    st.write("### Méthodologie")
    st.write("#### Approche")
    st.write("Notre projet sera traité selon deux axes principaux.")
    st.write("Le premier axe consiste à extraire des features visuelles des images pour les classifier.")
    st.write("Le deuxième axe consiste à extraire du texte des images pour les classifier.")
    st.write("Finalement, nous fusionnerons les deux modèles pour obtenir un modèle plus performant.")
    st.write("#### Méthode visuelles")
    st.write("Pour l'approche visuelles, nous utiliserons un modèle pré-entrainé sur ImageNet pour faire du transfert learning avec notre dataset.")
    st.write("Nous ajouterons une couche de classification à la fin du modele pour prédire les classes de documents.")
    st.write("#### Méthode textuelles")
    st.write("Pour l'approche textuelles, nous utiliserons un modèle de reconnaissance optique de caractères (OCR) pour extraire du texte des images.")
    st.write("Nous utiliserons ensuite ce texte avec un modele de classification de texte.")
    st.write("#### Modèle final")
    st.write("Pour le modèle final, nous fusionnerons les prédictions des modèles visuels et textuels pour obtenir de meilleurs performances.")

    #priting the architecture of the model
    if st.checkbox("Architecture du modèle"):
        st.image("../../data/visualization/full_models.png")




#OCR page
if page == pages[3] : 
    st.write("### OCR")
    st.write("#### Principe")
    st.write("L'OCR (Optical Character Recognition) est une technologie qui permet de convertir des documents scannés en texte.")
    st.write("#### Visualisation")
    st.write("Plusieurs facteurs peuvent influencer la qualité de l'OCR, quatre d'entre eux ont été analysés:")
    #dimensions de l'image
    if st.checkbox("Dimension de l'image"):
        st.write("Toutes les images ont une hauteur de 1000 pixels, mais la largeur varie.")
        st.image("../../data/visualization/kde_largeur.png")
    #rotation de l'image
    if st.checkbox("Rotation de l'image"):
        st.write("La majorité des images sont orientées horizontalement, quelques'un sont retournés de 90°.")
        st.image("../../data/visualization/kde_orientation.png")
        #TODO: ajouter un bouton pour voir une image retournée ?
    #resolution de l'image
    if st.checkbox("Résolution de l'image"):
        st.write("Toutes les images ont une resolution de 72 dpi, ce qui est peu pour de l'OCR.")
        st.image("../../data/raw/selected_streamlit/00399159_9164.tif")
        st.markdown("""
        <style>
        img {
            cursor: pointer;
            transition: all .2s ease-in-out;
        }
        img:hover {
            transform: scale(3);
        }
        </style>
        """, unsafe_allow_html=True)
    #taille de la police
    if st.checkbox("Taille de la police de charactere"):
        st.write("50% des images ont une taille de police entre 5 et 8 pixels.")
        st.image("../../data/visualization/boxplot_police.png")
    st.write("La qualité du texte est très en dessous des characteristiques optimales pour l'OCR. Cela impactera la qualité de l'OCR.")
    #documentation tesseract
    if st.checkbox("Documentation Tesseract"):
        st.markdown("""
                <div style="background-color: #FFDAB9; padding: 10px; border-radius: 10px;">
                <p style="font-size: 14px;">There is a minimum text size for reasonable accuracy. You have to consider resolution as well as point size. Accuracy drops off below 10pt x 300dpi, rapidly below 8pt x 300dpi. A quick check is to count the pixels of the x-height of your characters. (X-height is the height of the lower case x.) At 10pt x 300dpi x-heights are typically about 20 pixels, although this can vary dramatically from font to font. Below an x-height of 10 pixels, you have very little chance of accurate results, and below about 8 pixels, most of the text will be "noise removed".</p>
                </div>
                """, unsafe_allow_html=True)

    #preprocessing
    st.write("#### Preprocessing")
    st.write("Pour améliorer la qualité de l'OCR, nous avons appliqué plusieurs techniques de preprocessing:")
    st.write("1. changement de dimension")
    st.write("Lors du preprocessing les images sont toutes redimensionnées en 1000x1000 pixels.")
    st.write("2. redressement de l'image")
    st.write("Les images sont redressées si elles sont orientées de 90°.\nPlus tard on remarquera que le changement de rotation a plutot un impact negatif sur la qualité de l'OCR. Cette modification ne sera pas présente pour la suite.")
    #choix du modele
    st.write("#### Choix du modèle d'OCR et metriques")
    st.write("Trois modèles d'OCR ont été testés: PyTesseract Easy OCR et Keras OCR.")
    st.write("Le score de similarité et l'indice de Jaccard ont été utilisés pour comparer les textes extraits aux textes de référence")
    if st.checkbox("Définition du score de similarité"):
        st.markdown("""
                <div style="background-color: #FFDAB9; padding: 10px; border-radius: 10px;">
                <p style="font-size: 14px;">Le score de similarité est une mesure de similarité cosinus entre deux vecteurs representant les textes à comparer.</p>
                <p style="font-size: 14px;">Chaque token d'un texte est transformé en vecteur selon la methode d'embedding de la librairie spacy.</p>
                <p style="font-size: 14px;">Le vecteur d'un texte est la moyenne des vecteurs de ses tokens.</p>
                </div>
                """, unsafe_allow_html=True)
    if st.checkbox("Définition de l'indice de Jaccard"):
        st.markdown("""
                <div style="background-color: #FFDAB9; padding: 10px; border-radius: 10px;">
                <p style="font-size: 14px;">L'indice de Jaccard est un ratio entre l'intersection et l'union de deux ensembles.</p>
                <p style="font-size: 14px;">Dans notre cas, les ensembles sont les mots des textes à comparer.</p>
                <p style="font-size: 14px;">L'indice de Jaccard est le rapport entre le nombre de mots communs entre les deux textes et le nombre total de mots des deux textes.</p>
                </div>
                """, unsafe_allow_html=True)
    st.write("#### Résultats")
    if st.checkbox("Score de similarité"):
        st.image("../../data/visualization/similarity_score.png")
    if st.checkbox("Indice de Jaccard"):
        st.image("../../data/visualization/jaccard_index.png")
    st.write("La librairie d'OCR retenue est pytesseract, car elle a donné en moyenne les meilleurs résultats. ")

    #exemple de texte extrait
    st.write("#### Exemple de texte extrait")
    if st.button("Afficher un exemple"):
        #showing a random picture
        file = random.choice(image_files)
        file_path = "../../data/raw/selected_streamlit/" + file 
        st.image(file_path)
        #preprocessing the image
        resized_image = preproc.resize_image(file_path, (1000, 1000))
        #extracting the text
        extracted_text = preproc.ocr_pytesseract(resized_image)
        st.write(extracted_text)









#Modélisation visuelle page
if page == pages[4] : 
    st.write("### Modélisation visuelle")




#Modélisation textuelle page
if page == pages[5] : 
    st.write("### Modélisation textuelle")




#Conclusion page
if page == pages[6] : 
    st.write("### Conclusion")


