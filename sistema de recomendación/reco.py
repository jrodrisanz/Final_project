import streamlit as st
import pandas as pd
from PIL import Image
import pylab as plt
import webbrowser
import base64
import io
import re

st.title('FlickPick')

st.write('¡Welcome to FlickPick! tu recomendador de películas personalizado, pero personalizado de verdad. Este test te ayudará a descubrir películas y series que se ajusten a tus preferencias y gustos. Responde las siguientes preguntas y al final obtendrás recomendaciones. ¡Comencemos!')

st.sidebar.header('FlickPick Navigator')
st.sidebar.subheader('Streamlit Recom')
st.sidebar.info('Aquí puedes poner una barra de navegación o zonas para cargar archivos')


# Preguntas
pregunta1 = st.radio('¿Prefieres los clásicos o las nuevas?', ['Clásicas', 'Contemporáneas', 'Ambas'])

pregunta2 = st.radio('¿Cuál es tu género favorito?', ['Acción', 'Drama', 'Comedia', 'Documental', 'Horror', 'Romance', 'TODOS'])

pregunta3 = st.radio('¿Qué tipo de trama te resulta más interesante?', ['Misterio', 'Aventura','Fantasía', '¡Cualquiera!'])

pregunta4 = st.text_input('Escribe el nombre del actor o la actriz que deba aparecer en tu lista (o no escribas ninguno)')

pregunta5 = st.radio('¿Prefieres películas basadas en hechos reales o ficción?', ['Hechos reales', 'Ficción', '¡Cualquiera!'])

#pregunta6 = st.selectbox('¿Qué duración prefieres?', ['Cortas (menos de 90 minutos)', 'Estándar (90-120 minutos)', 'Largas (más de 120 minutos)'])

duracion_minima = st.slider('Duración mínima (minutos)', 0, 300, 0)
duracion_maxima = st.slider('Duración máxima (minutos)', 0, 300, 300)

pregunta7 = st.radio('¿Te gustan los finales felices?', ['Mucho', 'No', '¯\_(ツ)_/¯'])

pregunta8 = st.text_input('¿Tienes alguna preferencia sobre el lugar o la época en la que se desarrolla la película?')

pregunta9 = st.selectbox('¿Qué plataforma de streaming utilizas con más frecuencia?', ['Netflix', 'Amazon Prime', 'HBO'])

# Recolectar las respuestas
respuestas = {
    'pregunta1': pregunta1,
    'pregunta2': pregunta2,
    'pregunta3': pregunta3,
    'pregunta4': pregunta4,
    'pregunta5': pregunta5,
    'pregunta6': (duracion_minima, duracion_maxima),
    'pregunta7': pregunta7,
    'pregunta8': pregunta8,
    'pregunta9': pregunta9,
}

# Mostrar las respuestas
st.write('Respuestas seleccionadas:')
st.write(respuestas)



#filtrar recomendaciones

def generar_recomendaciones(respuestas):
    titles = pd.read_csv('../data/clean/titles.csv', encoding='utf-8', encoding_errors='ignore')
    comments = pd.read_csv('../data/clean/com_group.csv', encoding='utf-8', encoding_errors='ignore')

    # 1. Filtro por antiguedad
    if respuestas['pregunta1'] == 'Clásicas':
        df_filtrado = titles[titles['release_year'] <= 1990]
    elif respuestas['pregunta1'] == 'Contemporáneas':
        df_filtrado = titles[titles['release_year'] > 1990]
    elif respuestas['pregunta1'] == 'Ambas':
        df_filtrado = titles

    # 2. Filtro por género favorito
    if respuestas['pregunta2'] == 'Acción':
        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('action', case=False)]
    elif respuestas['pregunta2'] == 'Drama':
        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('drama', case=False)]
    elif respuestas['pregunta2'] == 'Comedia':
        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('comedy', case=False)]
    elif respuestas['pregunta2'] == 'Documental':
        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('documentation', case=False)]
    elif respuestas['pregunta2'] == 'Horror':
        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('horror', case=False) | df_filtrado['genres'].str.contains('thriller', case=False)]
    elif respuestas['pregunta2'] == 'Romance':
        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('romance', case=False)]
    elif respuestas['pregunta2'] == 'TODOS':
        pass

# 3. Filtro por tipo de trama favorita
    sinonimos_misterio = ['mystery', 'enigma', 'puzzle', 'riddle', 'conundrum', 'secret', 'intrigue', 'clue', 'suspense']
    sinonimos_aventura = ['thrill', 'adventure', 'expedition', 'journey', 'quest', 'explor', 'trek', 'voyage', 'clue', 'safari', 'exploration']
    sinonimos_fantasia = ['fantasy', 'imagination', 'imaginary', 'enchant', 'magic', 'tail', 'fairy', 'myth', 'wonder', 'dream', 'illusion', 'super', 'superhero']

    def buscar_sinonimos(critica, sinonimos):
        for sinonimo in sinonimos:
            if sinonimo in critica:
                return True
        return False

    if respuestas['pregunta3'] == 'Misterio':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_misterio))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta3'] == 'Aventura':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_aventura))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta3'] == 'Fantasía':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_fantasia))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta3'] == '¡Cualquiera!':
        pass



    # 4. Filtro por actor o actriz favorito
    if respuestas['pregunta4'] != '':
        actor_favorito = respuestas['pregunta4']
        df_filtrado = df_filtrado[df_filtrado['actors'].str.contains(actor_favorito, case=False)]
    else:
        pass


    # 6. Filtro para determinar ficción
    palabras_clave_ficcion = ['fiction', 'imaginary', 'fantasy', 'fictional', 'otherworld', 'extraterrestrial']
    
    if respuestas['pregunta5'] == "Ficción":
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, palabras_clave_ficcion))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta5'] == "Hechos reales":
        comentarios_filtrados = comments[~comments['review'].apply(lambda x: buscar_sinonimos(x, palabras_clave_ficcion))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta5'] == '¡Cualquiera!':
        pass


    # 6. Filtro por duración de la película
    df_filtrado = df_filtrado[(df_filtrado['duration'] >= duracion_minima) & (df_filtrado['duration'] <= duracion_maxima)]


    # 7. Filtro por final feliz
    sinonimos_final_feliz = ['happy ending', 'positive outcome', 'pleasant conclusion', 'satisfying resolution', 'joyful finale', 'contented ending', 'delightful outcome', 'cheerful conclusion', 'pleasant ending', 'uplifting finale']
    
    if respuestas['pregunta7'] == 'Mucho':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_final_feliz))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta7'] == 'No':
        comentarios_filtrados = comments[~comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_final_feliz))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta7'] == '¯\_(ツ)_/¯':
        pass



    # Resto de filtros...
    # df_filtrado = df_filtrado[...]

    recomendaciones = df_filtrado['title'].tolist()

    return recomendaciones




