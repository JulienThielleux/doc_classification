import preproc
import models

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import streamlit as st
import torch
import torch.nn as nn


#creation de la liste d'image
folder = "data/raw/selected_streamlit"
files = os.listdir(folder)
image_files = [f for f in files if f.endswith(('.tif'))]




#menu
st.title("Classification de documents")
st.sidebar.title("Menu")
pages = ["Home", "Datasets", "Méthodologie", "OCR", "Modélisation textuelle ", "Modélisation visuelle", "Voting", "Conclusion"]
page = st.sidebar.radio("Aller vers", pages)

#liens
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

with st.sidebar.expander("Classes des documents", expanded=False):
    st.markdown("""
                0. letter
                1. form
                2. email
                3. handwritten
                4. advertisement
                5. scientific report
                6. scientific publication
                7. specification
                8. file folder
                9. news article
                10. budget
                11. invoice
                12. presentation
                13. questionnaire
                14. resume
                15. memo
                """)

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
    st.header("Home")
    st.write("Ce projet est issus de la formation data science de DataScientest. Il a pour but de classifier des documents scannés en 16 classes différentes.")
    st.write("Cette démarche vise à simplifier et à accélérer le processus de gestion des documents, en permettant aux utilisateurs de classer automatiquement une grande quantité de documents en un temps record.")
    st.write("Ce streamlit présente les différentes étapes du projet de l'exploration du dataset jusqu'à la modélisation")
    




#Datasets page
if page == pages[1] : 
    st.header("Datasets")

    st.write("Le dataset utilisé est le RVL_CDIP qui contient 400 000 images de documents scannés en 16 classes différentes.")
    st.write("Les images sont caractérisées par une faible résolution (<100dpi), du bruit et des artefacts de scan.")

    tab1, tab2 = st.tabs(["Typologie", "Caractéristiques"])

    with tab1:

        #montrer un exemple de chaque classe
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
                cols[i % num_columns].image("data/raw/selected_streamlit/" + picture_dict[doc])

    with tab2:
        st.write("Les documents scannés ont plusieurs caractéristiques qui pourraient complexifier la sélection du modèle.")
        st.write("Les voici :")
        with st.expander("Luminance"):
            st.write("Les images ne contiennent pas la même proportion de pixels clairs ou foncés")
            st.image("reports/figures/rvlcdip_dark_pixel_distribution.png")
            st.write("La majorité des images ont entre 1 et 20% de pixels foncés ce qui correspond à du texte ou des images.")
            st.write("Regardons quelques exemples d'images extremement claires ou foncées :")
            st.image("reports/figures/rvlcdip_dark_pixel_sample.png")
        with st.expander("Taille des caractères"):
            st.write("La taille des lettres peut influencer les modèles sélectionnés.")
            st.write("Les images ne contiennent pas la même taille de caractère.")
            st.image("reports/figures/rvlcdip_letter_size_distribution_raw.png")
            st.write("La majorité des caractères ont entre 2 et 20 pixels. en enlevant les quelques abérations :")
            st.image("reports/figures/rvlcdip_letter_size_clean_distribution.png")
            st.write("Regardons quelques exemples d'images abérantes :")
            st.image("reports/figures/rvlcdip_max_letter_size.png")
            st.write("C'est le modèle de detection de caractère qui voit des bordures comme des caractères, ce qui ne devrait pas influencer le CNN. ")
        with st.expander("Orientation des images"):
            st.write("L'orientation des images peuvent influencer la detection de caractères de l'OCR mais aussi empecher la classification visuelle.")
            st.image("reports/figures/rvlcdip_dark_pixel_distribution.png")
            st.write("En réorientant les images, on devrait facilité la detection de la classe par le modèle.")
            st.write("Cependant, la detection d'orientation n'est pas parfaite car certains documents possedent des écrits dans différentes directions ou encore de l'écriture manuscrite.")
            st.image("reports/figures/rvlcdip_rotation_sample.png")


    #montrer une image aléatoire
    st.write("##### Image aléatoire:")
    if st.button("Afficher"):
        file = random.choice(image_files)
        file_path = os.path.join(folder, file)

        txt_file = os.path.splitext(file)[0] + '.txt'
        txt_file_path = os.path.join(folder, txt_file)
        with open(txt_file_path, 'r') as f:
            class_num = int(f.read().strip())
        document_class = preproc.to_class_name(class_num)

        st.write("Classe du document: ")
        st.markdown(f'<span style="color:blue;">{document_class}</span>', unsafe_allow_html=True)
        st.image(file_path)







#Méthodologie page
if page == pages[2] : 

    st.header("Méthodologie")
    tab1, tab2, tab3 = st.tabs(["Approche", "Architecture du modèle","Dataset"])

    with tab1:

        st.write("Ce projet sera traité selon deux axes principaux.")
        st.write("- Le premier axe consiste à extraire des features visuelles des images pour les classifier.")
        st.write("- Le deuxième axe consiste à extraire du texte des images pour les classifier.")
        st.write("Finalement, les deux modèles seront fusionnés pour obtenir un modèle plus performant.")
        st.write("##### Méthode visuelle")
        st.write("Pour l'approche visuelle, un modèle pré-entrainé sur ImageNet sera utilisé pour faire du transfert learning avec notre dataset.")
        st.write("On ajoutera une couche de classification à la fin du modèle pour prédire les classes de documents.")
        st.write("##### Méthode textuelle")
        st.write("Pour l'approche textuelle, un modèle de reconnaissance optique de caractères (OCR) sera utilisé pour extraire du texte des images.")
        st.write("Ce texte servira ensuite de base à un modele de classification de texte.")
        st.write("##### Modèle final")
        st.write("Finalement les probabilités renvoyées par les modèles visuels et textuels seront moyennées pour obtenir une prédiction globale.")
    
    with tab2:
        st.image("reports/figures/full_models.png")

    with tab3:
        st.write("- Le dataset fourni contient près de 400.000 images.")
        st.write("Afin de procéder à la selection du modèle et à son entrainement dans le temps imparti tout en améliorant son efficacité, nous avons sélectionné 20.000 images équiréparties sur l'ensemble des 16 classes.")
        # Titre de l'application
        st.write('##### Histogramme des 16 classes')

        # Générer des données pour l'histogramme
        # 16 classes contenant chacune 1250 éléments
        classes = 16
        elements_per_class = 20000/16
        data = np.repeat(np.arange(classes), elements_per_class)

        # Créer l'histogramme
        fig, ax = plt.subplots()
        ax.hist(data, bins=classes, edgecolor='black')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Nombre d\'éléments')
        ax.set_title('Histogramme de 16 classes contenant chacune 1250 éléments')

        # Afficher l'histogramme dans Streamlit
        st.pyplot(fig)
        st.write("- Nous possédons aussi un dataset de 520 images afin de procéder à la sélection de l'OCR.")




#OCR page
if page == pages[3] :

    st.header("OCR")
    st.write("L'OCR (Optical Character Recognition) est une technologie qui permet de convertir des documents scannés en texte. Il s'agit d'une étape de préprocessing préalable à la classification textuelle.")

    tab1, tab2, tab3, tab4 = st.tabs(["Visualisation", "Preprocessing", "Choix du modèle", "Tests"])

    with tab1:

        st.write("Plusieurs facteurs peuvent influencer la qualité de l'OCR, quatre d'entre eux ont été analysés:")
        #dimensions de l'image
        if st.checkbox("Dimension de l'image"):
            st.write("Toutes les images ont une hauteur de 1000 pixels, mais la largeur varie.")
            st.image("reports/figures/kde_largeur.png")
        #rotation de l'image
        if st.checkbox("Rotation de l'image"):
            st.write("La majorité des images sont orientées horizontalement, quelques'un sont retournés de 90°.")
            st.image("reports/figures/kde_orientation.png")
        #resolution de l'image
        if st.checkbox("Résolution de l'image"):
            st.write("Toutes les images ont une resolution de 72 dpi, ce qui est peu pour de l'OCR.")
            st.image("data/raw/selected_streamlit/00399159_9164.tif")
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
        if st.checkbox("Taille de la police de caractere"):
            st.write("50% des images ont une taille de police entre 5 et 8 pixels.")
            st.image("reports/figures/boxplot_police.png")
        st.write("La qualité du texte est très en dessous des caracteristiques optimales pour l'OCR. Cela impactera la qualité de l'OCR.")
        #documentation tesseract
        if st.checkbox("Documentation Tesseract"):
            st.markdown("""
                    <div style="background-color: #FFDAB9; padding: 10px; border-radius: 10px;">
                    <p style="font-size: 14px;">There is a minimum text size for reasonable accuracy. You have to consider resolution as well as point size. Accuracy drops off below 10pt x 300dpi, rapidly below 8pt x 300dpi. A quick check is to count the pixels of the x-height of your characters. (X-height is the height of the lower case x.) At 10pt x 300dpi x-heights are typically about 20 pixels, although this can vary dramatically from font to font. Below an x-height of 10 pixels, you have very little chance of accurate results, and below about 8 pixels, most of the text will be "noise removed".</p>
                    </div>
                    """, unsafe_allow_html=True)

    #preprocessing
    with tab2:
        st.write("Plusieurs techniques de preprocessing ont été appliquées pour améliorer la qualité de l'OCR:")
        st.write("1. changement de dimension")
        st.write("Lors du preprocessing les images sont toutes redimensionnées en 1000x1000 pixels.")
        st.write("2. redressement de l'image")
        st.write("Les images sont redressées si elles sont orientées de 90°.\nPlus tard on remarquera que le changement de rotation a plutot un impact negatif sur la qualité de l'OCR. Cette modification ne sera pas présente pour la suite.")
    
    #choix du modele
    with tab3:
        st.write("Trois modèles d'OCR ont été testés: PyTesseract, Easy OCR et Keras OCR.")
        st.write("##### Métriques")
        st.write("Le score de similarité et l'indice de Jaccard ont été utilisés pour comparer les textes extraits aux textes de référence")

        #definition des metriques
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
            
        st.write("##### Résultats")
        if st.checkbox("Score de similarité"):
            st.image("reports/figures/similarity_score.png")
        if st.checkbox("Indice de Jaccard"):
            st.image("reports/figures/jaccard_index.png")
        st.write("La librairie d'OCR retenue est pytesseract, car elle a donné en moyenne les meilleurs résultats. ")
            
    #resultats
    with tab4:
        #exemple de texte extrait
        st.write("##### Exemple de texte extrait")
        if st.button("Afficher un exemple"):
            file = random.choice(image_files)
            file_path = "data/raw/selected_streamlit/" + file 
            st.image(file_path)
            
            #preproc + ocr
            resized_image = preproc.resize_image(file_path, (1000, 1000))
            extracted_text = preproc.ocr_pytesseract(resized_image)
            st.write("Texte extrait:")
            st.markdown(f'<p style="color:blue;">{extracted_text}</p>', unsafe_allow_html=True)

        #importer un fichier et extraire le texte
        st.write("##### Tester avec un fichier personnel")
        uploaded_file = st.file_uploader("Importer un fichier", type=["tif"])
        if uploaded_file is not None:
            st.image(uploaded_file)

            #preproc + ocr
            resized_image = preproc.resize_image(uploaded_file, (1000, 1000))
            extracted_text = preproc.ocr_pytesseract(resized_image)
            st.write("Texte extrait:")
            st.markdown(f'<p style="color:blue;">{extracted_text}</p>', unsafe_allow_html=True)




#Modélisation textuelle page
if page == pages[4] :
    st.header("Modélisation textuelle") 
    st.write("Cette étape a pour but de classifier les textes extraits des images suite à l'OCR. 20000 images ont été utilisées pour l'entrainement.")

    tab1, tab2, tab3 = st.tabs(["Preprocessing", "Modèles", "Tests"])

    with tab1:
        st.write("1. OCR")
        st.write("2. Nettoyage")
        st.write("(Suppression des caractères spéciaux, suppression des chiffres, suppression des mots de moins de 3 lettres, suppression des stops words, suppression des mots apparaissant moins de 3 fois dans tous les textes)")
        st.write("3. Lemmatization")

        #exemple de texte nettoyé
        if st.button("Exemple au hasard"):
            file = random.choice(image_files)
            file_path = "data/raw/selected_streamlit/" + file         
            resized_image = preproc.resize_image(file_path, (1000, 1000))
            extracted_text = preproc.ocr_pytesseract(resized_image)
            st.write("###### Texte extrait:")
            st.markdown(f'<p style="color:blue;">{extracted_text}</p>', unsafe_allow_html=True)
            st.write("###### Texte après preprocessing:")
            cleaned_text = preproc.clean_text(extracted_text)
            st.markdown(f'<p style="color:blue;">{cleaned_text}</p>', unsafe_allow_html=True)
            st.write("On constate que le texte extrait étant souvent de mauvaise qualité, le texte après preprocessing est également très pollué.")

    #modèles
    with tab2:
        st.write("Avant la modélisation, les textes ont été vectorisés selon une methode TF-IDF.")
        if st.checkbox("Définition de TF-IDF"):
            st.markdown("""
                    <div style="background-color: #FFDAB9; padding: 10px; border-radius: 10px;">
                    <p style="font-size: 14px;">La vectorisation TF-IDF (Term Frequency-Inverse Document Frequency) est une technique utilisée pour évaluer l’importance d’un mot dans un document par rapport à un corpus de documents.</p>
                    <p style="font-size: 14px;">Elle combine deux mesures : la fréquence du terme (TF), qui compte le nombre de fois qu’un mot apparaît dans un document, et la fréquence inverse du document (IDF), qui mesure la rareté du mot dans l’ensemble des documents.</p>
                    <p style="font-size: 14px;">En multipliant ces deux valeurs, TF-IDF permet de pondérer les mots de manière à mettre en avant ceux qui sont significatifs pour un document donné tout en réduisant l’importance des mots courants.</p>
                    </div>
                    """, unsafe_allow_html=True)
        st.write("Plusieurs modèles de classification de texte ont donnés de bons résultats:")
        #matrice de confusion des modeles
        if st.checkbox("SVC"):
            st.image("reports/figures/cm_SVC.png")
            st.write("Le modèle SVC a une accuracy de 0.70.")
        if st.checkbox("Random Forest"):
            st.image("reports/figures/cm_rf.png")
            st.write("Le modèle Random Forest a une accuracy de 0.67.")
        if st.checkbox("Regression logistique"):
            st.image("reports/figures/cm_lr.png")
            st.write("Le modèle Regression logistique a une accuracy de 0.68.")

    #choix du modele
    with tab3:
        model = st.selectbox("Choisir un modèle", ["SVC", "Random Forest", "Regression logistique"])
        if st.button("Tester"):
            #afficher une image
            file = random.choice(image_files)
            file_path = "data/raw/selected_streamlit/" + file 
            st.image(file_path)

            #extraire la classe
            txt_file = os.path.splitext(file)[0] + '.txt'
            txt_file_path = os.path.join(folder, txt_file)
            with open(txt_file_path, 'r') as f:
                class_num = int(f.read().strip())
            document_class = preproc.to_class_name(class_num)

            #extraire le texte
            resized_image = preproc.resize_image(file_path, (1000, 1000))
            extracted_text = preproc.ocr_pytesseract(resized_image)
            cleaned_text = preproc.clean_text(extracted_text)

            #faire la prédiction
            if model == "SVC":
                prediction = models.SVC_model(cleaned_text)
            elif model == "Random Forest":
                prediction = models.RF_model(cleaned_text)
            elif model == "Regression logistique":
                prediction = models.LR_model(cleaned_text)
            st.write("Classe du document: ")
            st.markdown(f'<span style="color:blue;">{document_class}</span>', unsafe_allow_html=True)
            st.write("Categorie prédite:: ")
            st.markdown(f'<span style="color:blue;">{preproc.to_class_name(prediction[0])}</span>', unsafe_allow_html=True)

        #importer un fichier et faire une prediction
        st.write("#### Tester avec un fichier personnel.")
        uploaded_file = st.file_uploader("Importer un fichier", type=["tif"])
        if uploaded_file is not None:
            st.image(uploaded_file)

            #preproc + ocr + clean + prediction
            resized_image = preproc.resize_image(uploaded_file, (1000, 1000))
            extracted_text = preproc.ocr_pytesseract(resized_image)
            cleaned_text = preproc.clean_text(extracted_text)
            prediction = models.LR_model(cleaned_text)
            st.write("Categorie prédite:: ")
            st.markdown(f'<span style="color:blue;">{preproc.to_class_name(prediction[0])}</span>', unsafe_allow_html=True)



#Modélisation visuelle page
if page == pages[5] : 
    st.write("### Modélisation visuelle")
    st.write("""
             Notre objectif principal était de former un réseau de neurones convolutifs (CNN) capable de reconnaître et de classifier certaines caractéristiques spécifiques présentes dans des documents.""")
    st.write("""
             Pour ce faire, nous avons entraîné les CNN sur un jeu de données constitué de 20 000 documents scannés et étiquetés. """)
    
    
    with st.container():
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["CNN simple", "CNN complexe", "EfficientNet B0", "EfficientNet B0 enhanced","EfficientNet B1"])

        with tab1:
            st.write("### Modèle simple de CNN")
            st.write("Notre exploration d'un modèle à débuté par un modèle simple de CNN")
            with st.expander("Architecture du modèle simple"):
                st.write('''
                        Le modèle simple est constitué de 3 coeurs de convolutions chacun associés à un MaxPooling.
                        Deux couches linéaires permettent de réduire les dimensions pour retrouver nos 16 classes.
                        
                ''')
                st.image("reports/figures/CNN_simple_architecture.png")
            st.write("""Ce modèle simple constitué de 3 couches neuronales à permis d'obtenir une precision de 56%.
                    Cependant, certaines classes ne sont pas du tout detectées. Ces classes sont souvent des classes hybrides proche d'autres classes.""")
            with st.expander("Matrice de confusion"):
                st.write('''
                        Le modèle est tres performant sur certains type de document du fait de leur structure tres normée et particulière :
                         - 2 - email
                         - 4 - advertisement
                         - 9 - news article
                         - 14 - resume

                         Le format codifié de ces documents les rend facilement detectable par rapport à d'autres documents. 
                         Cependant le modèle n'arrive pas à extraire d'autres structures aussi finement comme le serait des dossier ou des tableaux de données.
                        
                ''')
                st.image("reports/figures/cm_cnn_simple.png")

            # Pros & Cons
            st.write("#### Avantages et Inconvénients")

            # Créer deux colonnes
            col1, col2 = st.columns(2)
            # Contenu de la colonne des avantages
            with col1:
                st.write("##### Avantages")
                st.markdown(
                        """
                        - ✅ **Simplicité du modèle** : Modèle simple et compréhensible qui extrait les détails les plus importants.
                        - ✅ **Rapidité d'entrainement** : L'entrainement et l'execution est rapide.
                        """
                    )

            # Contenu de la colonne des inconvénients
            with col2:
                st.write("##### Inconvénients")
                st.markdown(
                        """
                        - ❌ **Detection** : Faible sur les documents n'ayant pas un format spécifique.
                        - ❌ **Disparité** : Trés précis sur certains documents uniquement.
                        """
                    )


        with tab2:
            st.write("### Modèle complexe de CNN")
            st.write("Nous avons complexifier le premier modèle pour augmenter le niveau de détails et ainsi mieux discriminer les documents.")
            with st.expander("Architecture du modèle complexe"):
                st.write('''
                         Le modèle complexe est constitué de 3 coeurs de convolutions double chacun associé à un MaxPooling.                        
                ''')
                st.image("reports/figures/Sequentiel_complexe.png")

                st.write('''
                         Deux couches linéaires permettent de réduire les dimensions pour retrouver nos 16 classes.
                         Le modèle final est :
                        
                ''')
                st.image("reports/figures/CNN_complexe.png")
            st.write("""Ce modèle complexe a permis d'obtenir une precision de 33%. Cela est du au faible nombre d'epoques parcourues qui n'a pas permis de réduir suffisamment le coût [15 époques parcourues en 1085 min].
                     Un temps plus long aurait permis une meilleure précision du modèle.""")
            with st.expander("Matrice de confusion"):
                st.write('''
                        Le faible entrainement de ce modèle ne permet pas d'extraire suffisamment d'informations pour distinguer des classes proches comme les rapports scientifiques, les questionnaires ou les formulaires.
                         Ceopendant très vite, ce modèle distingue les structures des emails, des dossiers, des publicités.
                        
                ''')
                st.image("reports/figures/cm_cnn_complex.png")
                st.write("Un plus grand nombre d'itération permettrais d'améliorer la précision mais au détriment d'un temps nassez long. De plus, la complexité du modèle ne semble pas encore detecter des structures et motifs plus petits pour distinguer certains documents.")
            st.write("La complexification et l'entrainement du modèle augmente considérablement le temps d'exploration. Nous explorons les modèles pré-entrainés.")

            # Pros & Cons
            st.write("#### Avantages et Inconvénients")

            # Créer deux colonnes
            col1, col2 = st.columns(2)
            # Contenu de la colonne des avantages
            with col1:
                st.write("##### Avantages")
                st.markdown(
                        """
                        - ✅ **Complexité augmentée** : Detecte des détails plus fins.
                        """
                    )

            # Contenu de la colonne des inconvénients
            with col2:
                st.write("##### Inconvénients")
                st.markdown(
                        """
                        - ❌ **Disparité** : Le modèle renforce sa précision la où le modèle simple est déja précis.
                        - ❌ **Temps d'execution** : Un temps long d'entrainement qui n'a pas permis de réduire la fonction de coût.
                        """
                    )




        with tab3:
            st.write("### Modèle pré-entrainé EfficientNet B0")
            st.write("Nous testons le transfert de connaissance en se basant sur le modèle le plus performant")
            with st.expander("Selection d'un modèle pré-entrainé"):
                st.image("reports/figures/efficient_net_vs_other_models.png")
                st.write('''
                         Le modèle EfficientNet possede la précision la plus forte pour une complexité maitrisée et donc un temps de calcul restreint par rapport à d'autres modeles.                        
                ''')
            with st.expander("Architecture EfficientNet B0"):
                st.write('''
                         EfficientNet possede plusieurs noyaux de convolutions afin de faire ressortir les détails et les motifs différentiant.
                         ''')
                st.image("reports/figures/Architecture-of-EfficientNet-B0.png")

                st.write('''
                         Son architecture assez légère est rapide en calcul, cependant elle necessite des photos au format 224x224 et donc une réduction de taille par rapport à nos images.                        
                ''')
            st.write("""Le transfert de connaissance a permis d'obtenir une precision de 77,7%. 
                     Le temps d'entrainement reste raisonnable (680min) malgré la complexité du réseau.""")
            with st.expander("Matrice de confusion"):
                st.write('''
                        Le modèle est tres fort sur les elements tres distincts mais sous performe sur les rapports scientifiques qui contiennent plusieurs structures similaires à d'autres documents.
                        
                ''')
                st.image("reports/figures/cm_efnetB0.png")
            st.write("Nous explorons des améliorations du modèles pré-entrainés afin d'accroitre sa précision au global.")

            # Pros & Cons
            st.write("#### Avantages et Inconvénients")

            # Créer deux colonnes
            col1, col2 = st.columns(2)
            # Contenu de la colonne des avantages
            with col1:
                st.write("##### Avantages")
                st.markdown(
                        """
                        - ✅ **Entraînement** : Modèle entraîné sur un tres large dataset d'image.
                        - ✅ **Complexité** : La grande profondeur du réseau permet la detection de plus de motifs.
                        """
                    )

            # Contenu de la colonne des inconvénients
            with col2:
                st.write("##### Inconvénients")
                st.markdown(
                        """
                        - ❌ **Généraliste** : Le modèle a été entrainé sur ImageNet et non une base de données de documents.
                        - ❌ **Résolution** : Nécessite des images de faible résolution réduisant le niveau de détail .
                        """
                    )



        with tab4:
            st.write("### Modèle pré-entrainé EfficientNet B0 amélioré")
            col1, col2 = st.columns(2)

            
            with col1:
                st.write("#### Amélioration de la partie Neuronale")
                st.write("On ajoute une couche neuronales supplémentaire afin de faire ressortir de nouveaux détails et renforcer le modèle")
                st.image("reports/figures/EffNetB0_NAddOn.png")
                st.write("Elle accroît la complexité d'EfficientNet B0 tout en conservant le pré-entrainement des couches précédentes")
                
            with col2:
                st.write("#### Amélioration de la partie Linéaire")
                st.write("On ajoute une couche linéaire qui élimine les informations faibles afin de renforcer les signaux dominants")
                st.image("reports/figures/EffNetB0_LAddOn.png")
                st.write("Elle remplace la partie linéaire d'EfficientNet B0")
                
                
            col3, col4 = st.columns(2)
            
            with col3:
                st.write(""" Neuronal Add-On : 77,9% de précision 
                     Le temps d'entrainement est plus important (1080min) sans atteindre un niveau de coût suffisant.""")
                with st.expander("Matrice de confusion"):
                    st.write('''
                            Le modèle se renforce la où il est déja fort ce qui augmente la précision sans augmenter le F1-score.
                            
                    ''')
                    st.image("reports/figures/cm_efnetB0_NAO.png")
                st.markdown("---")    
                # Pros & Cons
                st.write("#### Avantages et Inconvénients")
                st.write("##### Avantages")
                st.markdown(
                        """
                        - ✅ **Complexité** : Accroît la complexité du modèle performant poru detecter plus déléments.
                        - ✅ **Transfert de connaissance** : Conserve l'apprentissage D'EfficientNet et l'augmente.
                        """
                    )
                st.write("##### Inconvénients")
                st.markdown(
                        """
                        - ❌ **Temps** : temps d'entrainement doublé pour peu de résultats.
                        - ❌ **Précision** : Le modèle renforce les inégalités de detection.
                        """
                    )

            with col4:
                st.write(""" Linear Add-On : 78,2% de précision 
                     La précision augmente mais le coût est moins minimisé et necessite plus d'epoques.""")
                with st.expander("Matrice de confusion"):
                    st.write('''
                            Le modèle augmente sa précision sur les éléments les moins biens detectés.
                            
                    ''')
                    st.image("reports/figures/cm_efnetB0_LAO.png")
                st.markdown("---")    
                # Pros & Cons
                st.write("#### Avantages et Inconvénients")
                st.write("##### Avantages")
                st.markdown(
                        """
                        - ✅ **Simplicité** : Permet de réduire le bruit et de renforcer les prédiction.
                        - ✅ **Efficacité** : Augmente la précision du modèle au global.
                        """
                    )
                st.write("##### Inconvénients")
                st.markdown(
                        """
                        - ❌ **Limite de B0** : La couche linéaire ne peut pallier les faiblesses intrinseques de B0 comme la résolution d'entrée.
                        """
                    )



            st.write("La modification de la partie linéaire semble etre la plus éfficiente.")
            st.write("On souhaite tout de même explorer les modèles plus complexe d'EfficientNet pour voir si l'on améliore la précision.")

        with tab5:
            st.write("### Modèle pré-entrainé EfficientNet B1")
            st.write("Nous testons le transfert de connaissance avec un modèle plus complexe, EfficientNetB1")
            with st.expander("Architecture EfficientNet B1"):
                st.write('''
                         EfficientNet B1 possède des noyaux supplémentaires par rapport au modèle B0.
                         ''')
                
                # Initialiser une variable de session pour suivre l'état du bouton
                if 'show_image1' not in st.session_state:
                    st.session_state.show_image1 = True

                # Définir les boutons
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Modèle B0"):
                        st.session_state.show_image1 = True
                with col2:
                    if st.button("Modèle B1"):
                        st.session_state.show_image1 = False

                # Afficher l'image en fonction de l'état du bouton
                if st.session_state.show_image1:
                    st.image('reports/figures/EffnetB0.png', caption='Architecture EfficientNet B0')
                else:
                    st.image('reports/figures/EffnetB1.png', caption='Architecture EfficientNet B1')

                st.write('''
                         Son architecture permet d'accoître la précision en detectant plus de détails sans accroître fortement le temps d'entrainement.
                         Elle necessite des photos au format 240x240 avec plus de détails que le modèle B0.                        
                ''')
            st.write("""Le transfert de connaissance a permis d'obtenir une precision de 81,9%. 
                     Le temps d'entrainement reste raisonnable (880min) malgré la complexité du réseau.""")
            with st.expander("Matrice de confusion"):
                st.write('''
                        Le modèle augmente sa précision au global mais conserve quelques lacunes.
                        
                ''')
                st.image("reports/figures/cm_efnet_B1.png")
            st.write("Nous décidons de partir sur ce modèle qui s'est montré le plus performant dans le temps imparti.")

            # Pros & Cons
            st.write("#### Avantages et Inconvénients")

            # Créer deux colonnes
            col1, col2 = st.columns(2)
            # Contenu de la colonne des avantages
            with col1:
                st.write("##### Avantages")
                st.markdown(
                        """
                        - ✅ **Complexité du modèle** : Detecte plus de détails que le modèle B0.
                        - ✅ **Résolution** : Des documents avec une meilleure résolution vs B0.
                        """
                    )

            # Contenu de la colonne des inconvénients
            with col2:
                st.write("##### Inconvénients")
                st.markdown(
                        """
                        - ❌ **Temps d'execution** : x1,5 temps d'entrainement par rapport à B0.
                        - ❌ **Précision** : Les documents sans format spécifique ont une disparité de detection.
                        """
                    )

    st.markdown("---")    
    with st.expander("Modèle sélectionné"):
        st.write("Après avoir réalisé plusieurs tests comparatifs d'algorithmes, nous avons opté pour l'utilisation de **EfficientNetB1**. ")
        st.write("""
                Grâce à cette méthode,nous avons réussi à obtenir un score de précision satisfaisant de 81,9 %.""")
    st.write("### Grad CAM")
    st.write("""
                Pour mieux comprendre les décisions prises par notre modèle, nous avons utilisé la technique Grad-CAM. 
                Cette mèthode nous permet de visualiser les zones des documents qui ont le plus contribué à la décision finale du modèle. En mettant en évidence ces régions d'intérêt, 
                Grad-CAM nous offre une meilleure interprétation des performances du réseau 
                et nous aide à identifier les caractéristiques visuelles clés sur lesquelles se base notre modèle pour la classification.""")
    
    st.write('### Test de classification et Grad CAM')

    # Number of classes
    NUM_CLASSES = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the EfficientNetB1 model
    model = models.CNN_model()
    model.eval()

    # Function to load image for display
    @st.cache_data
    def load_image(image_file):
        img = Image.open(image_file)
        return img

    # Path to the directory containing .tif images and text files
    image_files = [f for f in os.listdir(folder) if f.endswith('.tif')]

    # Get the list of target layers dynamically from the model
    target_layers = preproc.get_layers(model)

    # Initialize session state for the selected image
    if "selected_image_file" not in st.session_state:
        st.session_state.selected_image_file = None

    # CSS to align the button to the left and make it smaller
    st.markdown("""
        <style>
        .stButton button {
            width: 200px;
            display: block;
            margin-left: 0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Create a two-column layout
    col1, col2 = st.columns([1, 4])

    with col1:
        # Button to display a random image
        if st.button("Exemple au hasard"):
            st.session_state.selected_image_file = random.choice(image_files)

    if st.session_state.selected_image_file:
        selected_image_file = st.session_state.selected_image_file
        
        # Create a two-column layout for the image and the layer selection
        img_col, layer_col = st.columns([4, 5])

        with layer_col:
            selected_layer = st.selectbox("Sélectionnez une couche cible", target_layers)

        # Display the selected image and the Grad-CAM image side by side
        image_path = os.path.join(folder, selected_image_file)
        img = load_image(image_path)
        img_resized = img.resize((400, 500))  # Resize the random image

        # Load and preprocess the image
        image_tensor = preproc.preprocess_image(image_path)

        # Apply Grad-CAM
        gradients, activations, pred_class = preproc.grad_cam(model, image_tensor, selected_layer)

        # Generate and superimpose heatmap
        heatmap = preproc.generate_heatmap(gradients, activations)
        superimposed_img = preproc.superimpose_heatmap(image_tensor, heatmap)
        
        # Get the actual label from the corresponding text file
        txt_file_name = os.path.splitext(selected_image_file)[0] + '.txt'
        txt_file_path = os.path.join(folder, txt_file_name)

        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as file:
                actual_label_num = int(file.read().strip())
                actual_label = preproc.to_class_name(actual_label_num)

                # Display the results below the images
                st.write(f"**Classe prédite:** {preproc.to_class_name(pred_class)}")
                st.write(f"**Classe réelle:** {actual_label}")
        else:
            st.write("No matching label found for the selected image.")

        # Resize Grad-CAM image
        superimposed_img_resized = Image.fromarray(superimposed_img.astype(np.uint8)).resize((400, 500))

        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.image(img_resized, caption=f"Selected Image: {os.path.basename(image_path)}", width=400)

        with img_col2:
            st.image(superimposed_img_resized, caption=f"Grad-CAM\nLayer: {selected_layer}", width=400)

        # Add space between the columns
        st.markdown("""
            <style>
            .stColumn:first-child {
                margin-right: 20px;
            }
            </style>
            """, unsafe_allow_html=True)



#Voting page
if page == pages[6] :
    st.header("Voting")

    tab1, tab2, tab3 = st.tabs(["Principe", "Résultats", "Tests"])

    with tab1:
        st.write("Les deux approches ont été combinées pour améliorer encore les performances.")
        st.write("La méthode retenue a été un voting dit faible, c'est à dire que les probabilités renvoyées par les deux modeles sont moyennées pour obtenir une prédiction.")
        st.write("De plus des poids ont été attribués à chaque classe pour donner plus d'importance aux modèle le plus performant dans la prédiction de certaines classes.")

        #montrer l'architecture du voting
        if st.checkbox("Visualisation du Voting"):
            st.image("reports/figures/voting_models.png")

    #resultats
    with tab2:
        st.image("reports/figures/cm_voting.png")
        st.write("Pour rappel le meilleur modèle CNN avait une accuracy de 0.82. Le meilleur modèle textuel avait une accuracy de 0.70.")
        st.write("Le modèle de voting a une accuracy de 0.8475, ce qui est une amélioration significative.")

    #exemple avec les modèles optimaux
    with tab3:
        st.write("##### Tester avec un fichier aléatoire.")
        if st.button("Tester"):
            #afficher une image
            file = random.choice(image_files)
            file_path = "data/raw/selected_streamlit/" + file 
            st.image(file_path)

            #extraire la classe
            txt_file = os.path.splitext(file)[0] + '.txt'
            txt_file_path = os.path.join(folder, txt_file)
            with open(txt_file_path, 'r') as f:
                class_num = int(f.read().strip())
            document_class = preproc.to_class_name(class_num)

            #NLP
            #extraire le texte
            resized_image = preproc.resize_image(file_path, (1000, 1000))
            extracted_text = preproc.ocr_pytesseract(resized_image)
            cleaned_text = preproc.clean_text(extracted_text)

            #extraire les probabilités
            nlp_proba = models.LR_model_proba(cleaned_text)

            #CNN
            image_tensor = preproc.preprocess_image(file_path)

            #extraire les probabilités
            model = models.CNN_model()
            cnn_proba = models.CNN_model_proba(model, image_tensor)
            cnn_proba = cnn_proba.cpu().numpy()

            #voting
            voting_pred = models.voting(nlp_proba, cnn_proba)

            st.write("Classe du document: ")
            st.markdown(f'<span style="color:blue;">{document_class}</span>', unsafe_allow_html=True)
            st.write("Categorie prédite:: ")
            st.markdown(f'<span style="color:blue;">{preproc.to_class_name(voting_pred)}</span>', unsafe_allow_html=True)


        #importer un fichier et faire une prediction
        st.write("##### Tester avec un fichier personnel.")
        uploaded_file = st.file_uploader("Importer un fichier", type=["tif"])
        if uploaded_file is not None:
            st.image(uploaded_file)

            #preproc + ocr + clean + prediction
            resized_image = preproc.resize_image(uploaded_file, (1000, 1000))
            extracted_text = preproc.ocr_pytesseract(resized_image)
            cleaned_text = preproc.clean_text(extracted_text)
            nlp_proba = models.LR_model_proba(cleaned_text)

            #preproc + cnn + prediction
            image_tensor = preproc.preprocess_image(uploaded_file)
            model = models.CNN_model()
            cnn_proba = models.CNN_model_proba(model, image_tensor)
            cnn_proba = cnn_proba.cpu().numpy()

            voting_pred = models.voting(nlp_proba, cnn_proba)

            st.write("Categorie prédite:: ")
            st.markdown(f'<span style="color:blue;">{preproc.to_class_name(voting_pred)}</span>', unsafe_allow_html=True)

#Conclusion page
if page == pages[7] : 
    st.header("Conclusion")
    tab1, tab2, tab3 = st.tabs(["Résultats", "Comparaison", "Amélioration"])

    with tab1:
        st.subheader("Résultats")
        st.write("En combinant des modèles de classification visuelle et de classification textuelle, et en utilisant un modèle de vote pour combiner ces deux approches, nous avons réussi à obtenir une accuracy proche de 85%.")
        st.write("Le CNN a permis l’identification des caractéristiques visuelles uniques des documents (formatage, presence de colonnes, tableaux, ...etc), tandis que la NLP a été efficace pour comprendre le contenu du texte.")
        st.write("Le modèle de vote a ensuite servi de mécanisme d'arbitrage, pondérant les probabilités prédites par chaque modèle en fonction de leur pertinence pour chaque classe.")  
        st.write("Cette approche hybride nous a permis de tirer parti des forces de chaque modèle, maximisant ainsi notre capacité à classer correctement les documents.")
    
    with tab2:
        st.subheader("Etat de l'art")
        st.write("En comparant nos résultats avec ceux d'universitaires, notre système se comporte mieux que des première itération de CNN non préentrainée.")
        st.write("Cependant il n'atteint pas l'état de l'art qui est EfficientNet B4 entrainés sur des plus grands dataset.")
        
        st.write("")
        st.write("")
        #"Rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        data = {    
        "Model": [
            "DocXClassifier-L",
            "DocBert [DOCBERT]",
            "Eff-GNN + Word2Vec [word2vec]",
            "Multimodal Side-Tuning (MobileNetV2)",
            "Multimodal Side-Tuning (ResNet50)",
            "Proposed work",
            "DocBERT [DOCBERT]",
            "BERT [BERT]",
            "Eff-GNN + Word2Vec [word2vec] + Image Embedding",
            "Eff-GNN+ Word2Vec [word2vec]",
            "VGG"
        ],
        "Accuracy": [
            95.57,
            91.95,
            91,
            90.50,
            90.30,
            84.8,
            82.3,
            79,
            77.5,
            73.5,
            7.08
        ],
        "Paper": [
            "DocXClassifier: High Performance Explainable Deep Network for Document Image Classification",
            "Efficient Document Image Classification Using Region-Based Graph Neural Network",
            "Efficient Document Image Classification Using Region-Based Graph Neural Network",
            "Multimodal Side-Tuning for Document Classification",
            "Multimodal Side-Tuning for Document Classification",
            "Document Classification with EfficientNet and OCR+NLP voting combine",
            "Efficient Document Image Classification Using Region-Based Graph Neural Network",
            "Efficient Document Image Classification Using Region-Based Graph Neural Network",
            "Efficient Document Image Classification Using Region-Based Graph Neural Network",
            "Efficient Document Image Classification Using Region-Based Graph Neural Network",
            "Efficient Document Image Classification Using Region-Based Graph Neural Network"
        ],
        "Year": [2022, 2021, 2021, 2021, 2021, 2024, 2021, 2021, 2021, 2021, 2021]
        }




        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['Year'] = df['Year'].astype(int)  # Ensure 'Year' has no decimal places
        df.update(df.drop(columns='Year').applymap(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x))
        def highlight_row(row):
            if row['Year'] == 2024:
                return ['background-color: yellow'] * len(row)
            else:
                return [''] * len(row)
        
        df.set_index('Model', inplace=True)
        df = df.style.apply(highlight_row, axis=1)
        #styled_df = df.style.set_properties(**{'text-align': 'center'})

        # Display the styled DataFrame with st.table
        st.table(df)
        st.write("Nos limites techniques et temporelles n'ont pas permis d'atteindre ces resultats de plus de 95% de precision dans la prediction de classes.")


    with tab3:
        st.subheader("Axe d'amélioration")

        st.write("##### CNN")
        # Resolution EfficientNet
        data = {
            "Base model": ["EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7"],
            "Resolution": [224, 240, 260, 300, 380, 456, 528, 600]
        }

        # Data
        df = pd.DataFrame(data)

        # Affichage
        
          
    
    
    
        """
        Notre modèle avec une precision de 85% est satisfaisant pour une première approche du probleme de classification.
        C'est une base robuste pour iterer dessus.

        Les limites materiels et temporelles sont tres vite apparus comme facteurs limitant d'itérations sur le CNN. 
        La parallelisation de l'entrainement aurait permis d'iterer sur le CNN multi-couche ou bien tester des modèles EfficientNet plus complexes B2 à B7.
        """
        st.image("reports/figures/EfficientNet_comparison.png")
        st.write("La parallelisation permet de diviser jusqu'à 4 fois le temps d'entrainement.")
        st.table(df) 
        st.write("Ainsi un modèle EfficientNet plus complexe permet d'accroître la résolution de l'image en entrée et donc le niveau de détail.")
        
        st.markdown("""---""")
        st.write("##### NLP")
        
        """
        La difficulté principale du côté de l'approche NLP a été le preprocessing des documents en vue d'en obtenir le texte (OCR). 
        La qualité des documents rendait difficile la lecture par la libairie d'ocr. La librairie utilisée produisait très souvent des mots inexactes. 
        
        """
        st.image("reports/figures/OCR_error.png")
        """
        
        Une possibilité serait de transformer les mots inconnus par leur plus proche voisin lexical.
        """
        st.markdown("""---""")
        st.write("##### Dataset")
        """
        Enfin, notre dataset d'entrainement ne couvrait que 5% du dataset fournis.
        Un plus grand dataset d'entrainement permettrait d'ameliorer la precision, cependant le rendement est décroissant par rapport à la taille du dataset.
        Plus il est grand, plus le temps d'entrainement augmente et la precision prend une allure logarithmique."""


