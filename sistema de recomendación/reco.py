import streamlit as st
import pandas as pd
from PIL import Image
import pylab as plt
import webbrowser
import base64
import io
import re



image_path = "../images/Sweet_Popcorn__1_-removebg-preview (1).png"

st.image(image_path, use_column_width=False)


st.write('### Bienvenidos a FlickPick🖖 Tu recomendador de películas personalizado, MUY PERSONALIZADO. Nuestro objetivo es ayudarte a descubrir películas y series que se adapten perfectamente a tí. ¡Empecemos a explorar juntos!')

st.sidebar.header('FlickPick Navigator')
st.sidebar.subheader('Streamlit Recom')
st.sidebar.info('Aquí puedes poner una barra de navegación o zonas para cargar archivos')


# Preguntas
pregunta1 = st.radio('¿Prefieres los clásicos o las producciones contemporáneas?', ['Clásicas', 'Contemporáneas', 'Ambas'])

#pregunta2 = st.radio('¿Cuál es tu género favorito?', ['Acción', 'Drama', 'Comedia', 'Documental', 'Horror', 'Romance', '¯\_(ツ)_/¯'])

pregunta3 = st.radio('¿Qué tipo de trama te resulta más interesante?', ['Misterio', 'Aventura','Fantasía', '¡Cualquiera!'])

pregunta4 = st.text_input('Escribe el nombre del actor o la actriz que deba aparecer en tu lista (o no escribas ninguno)')

pregunta5 = st.radio('¿Prefieres películas basadas en hechos reales o ficción?', ['Hechos reales', 'Ficción', '¡Cualquiera!'])

duracion_minima, duracion_maxima = st.slider('¿Cuánto debería durar?', 0, 300, (0, 300))

pregunta7 = st.radio('¿Te gustan los finales felices?', ['¿A quién no le va a gustar?', 'No', '¯\_(ツ)_/¯'])

pregunta8 = st.text_input('¿Tienes algún tema concreto, época o lugar favorito?')

#pregunta9 = st.selectbox('¿Qué plataforma de streaming utilizas con más frecuencia?', ['Netflix', 'Amazon Prime', 'HBO'])

# Recolectar las respuestas
respuestas = {
    'pregunta1': pregunta1,
    #'pregunta2': pregunta2,
    'pregunta3': pregunta3,
    'pregunta4': pregunta4,
    'pregunta5': pregunta5,
    'pregunta6': (duracion_minima, duracion_maxima),
    'pregunta7': pregunta7,
    'pregunta8': pregunta8,
    #'pregunta9': pregunta9,
}

# Mostrar las respuestas
#st.write('Respuestas seleccionadas:')
#st.write(respuestas)

def buscar_sinonimos(critica, sinonimos):
    for sinonimo in sinonimos:
        if sinonimo in critica:
            return True
    return False



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
#    if respuestas['pregunta2'] == 'Acción':
#       df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('action', case=False)]
#  elif respuestas['pregunta2'] == 'Drama':
#        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('drama', case=False)]
#    elif respuestas['pregunta2'] == 'Comedia':
#        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('comedy', case=False)]
#    elif respuestas['pregunta2'] == 'Documental':
#        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('documentation', case=False)]
#    elif respuestas['pregunta2'] == 'Horror':
#        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('horror', case=False) | df_filtrado['genres'].str.contains('thriller', case=False)]
#    elif respuestas['pregunta2'] == 'Romance':
#        df_filtrado = df_filtrado[df_filtrado['genres'].str.contains('romance', case=False)]
#    elif respuestas['pregunta2'] == '¯\_(ツ)_/¯':
#        pass


    # 3. Filtro por tipo de trama favorita
    sinonimos_misterio = ['mystery', 'enigma', 'puzzle', 'riddle', 'conundrum', 'secret', 'intrigue', 'clue', 'suspense']
    sinonimos_aventura = ['thrill', 'adventure', 'expedition', 'journey', 'quest', 'explor', 'trek', 'voyage', 'clue', 'safari', 'exploration']
    sinonimos_fantasia = ['fantasy', 'imagination', 'imaginary', 'enchant', 'magic', 'tail', 'fairy', 'myth', 'wonder', 'dream', 'illusion', 'super', 'superhero']

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
    
    if respuestas['pregunta7'] == '¿A quién no le va a gustar?':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_final_feliz))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta7'] == 'No':
        comentarios_filtrados = comments[~comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_final_feliz))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta7'] == '¯\_(ツ)_/¯':
        pass


    # 8. Filtro por tema, época o lugar favorito
    if respuestas['pregunta8'] != '':
        tema = respuestas['pregunta4']
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, tema))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    else:
        pass


    # 9. Filtro por plataforma
#    if respuestas['pregunta9'] != '':
#        plat_favorito = respuestas['pregunta9']
#        df_filtrado = df_filtrado[df_filtrado['platform'].str.contains(plat_favorito, case=False)]
#    else:
#        pass
#
#    test_solution = df_filtrado['title'].tolist()
#
#    return test_solution




