import cv2
import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error  # Importation explicite de urllib.error

# Constantes
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# Variables globales pour le FPS
prev_frame_time = 0
fps = 0

@st.cache_resource
def load_cascade_classifier():
    """Charge le classificateur, télécharge si absent et gère les erreurs."""
    cascade_path = Path(CASCADE_FILENAME)

    if not cascade_path.exists():
        st.info("Téléchargement du fichier de classification...")
        try:
            urllib.request.urlretrieve(CASCADE_URL, cascade_path)
            st.success("Fichier de classification téléchargé avec succès.")
        except urllib.error.URLError as e:
            st.error(f"Erreur de téléchargement (URL invalide ? Problème de réseau ?): {e}")
            st.stop()  # Arrêt de l'application en cas d'erreur critique
        except Exception as e:
            st.error(f"Erreur lors du téléchargement: {e}")
            st.stop()

    try:
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if face_cascade.empty():
            raise ValueError(f"Erreur de chargement du classificateur depuis : {cascade_path}")
        return face_cascade
    except Exception as e:
        st.error(f"Erreur lors de la création du classificateur : {e}")
        st.stop()

def calculate_fps():
    global prev_frame_time, fps
    curr_frame_time = time.time()
    try:
        fps = 1 / (curr_frame_time - prev_frame_time)
    except ZeroDivisionError:
        fps = 0
    prev_frame_time = curr_frame_time
    return int(fps)

def safe_camera_release(cap):
    if cap is not None and cap.isOpened():
        cap.release()
        time.sleep(0.1)  # Réduction du temps d'attente

def initialize_camera():
    st.info("Initialisation de la caméra...")
    for i in range(5):  # Essayer plusieurs index de caméra
        safe_camera_release(cv2.VideoCapture(i))
    time.sleep(0.5)

    try:
        # Essayer de capturer la caméra avec l'index 0 (et utiliser CAP_DSHOW pour Windows si nécessaire)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)

        if not cap.isOpened():
            raise RuntimeError("Échec de l'ouverture de la caméra")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Essayer plusieurs fois de récupérer une image valide
        for _ in range(3):  # Réduction du nombre de tentatives
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                st.success("Caméra initialisée avec succès")
                return cap
            time.sleep(0.5)  # Réduction du temps d'attente avant de réessayer

        raise RuntimeError("La caméra ne produit pas d'images valides")

    except Exception as e:
        if 'cap' in locals():
            safe_camera_release(cap)
        st.error(f"Erreur d'initialisation de la caméra: {e}")
        st.stop()  # Arrêt de l'application en cas d'erreur critique

def detect_faces_frame(frame, face_cascade, scaleFactor, minNeighbors, rectangle_color):
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Frame invalide")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
            cv2.putText(frame, "Visage", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)

        cv2.putText(frame, f'Visages: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, rectangle_color, 2)
        current_fps = calculate_fps()
        cv2.putText(frame, f'FPS: {current_fps}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, rectangle_color, 2)

        return frame, len(faces)

    except Exception as e:
        st.error(f"Erreur lors de la détection: {e}")
        return frame, 0

def main():
    st.set_page_config(page_title="Détection de Visages", page_icon="", layout="wide")
    st.title("Détection de Visages en Temps Réel")

    try:
        face_cascade = load_cascade_classifier()

        with st.sidebar:
            st.header("Paramètres de détection")
            rectangle_color = st.color_picker("Couleur du rectangle", "#00FF00")
            minNeighbors = st.slider("Minimum de voisins", 1, 10, 5)
            scaleFactor = st.slider("Facteur d'échelle", 1.01, 1.5, 1.1, 0.01)

        color_rgb = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        cap = initialize_camera()

        col1, col2 = st.columns([3, 1])
        with col1:
            frame_placeholder = st.empty()
        with col2:
            st.markdown("### Statistiques")
            stats_placeholder = st.empty()

        run = st.button("Démarrer/Arrêter la détection")

        while run:
            try:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    st.warning("Erreur de lecture de la caméra. Tentative de réinitialisation...")
                    safe_camera_release(cap)
                    time.sleep(1)
                    cap = initialize_camera()
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame, face_count = detect_faces_frame(frame, face_cascade, scaleFactor, minNeighbors, color_rgb)

                frame_placeholder.image(processed_frame, channels="RGB", use_container_width=True)

                stats_placeholder.markdown(f"""
                    - FPS: {fps:.1f}
                    - Échelle: {scaleFactor:.2f}
                    - Voisins: {minNeighbors}
                    - Visages détectés: {face_count}
                """)
                time.sleep(0.01)

            except Exception as e:
                st.error(f"Erreur dans la boucle principale: {e}")
                time.sleep(0.1)

    except Exception as e:
        st.error(f"Erreur générale: {e}")
    finally:
        if 'cap' in locals():
            safe_camera_release(cap)
        st.info("Ressources libérées")

if __name__ == "__main__":
    main()
