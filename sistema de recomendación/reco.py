import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pylab as plt
import webbrowser
import base64
import io
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


st.set_option('deprecation.showPyplotGlobalUse', False)

st.image("../images/logo2.png", use_column_width=False)

st.write('# Bienvenidos a FlickPick🖖')


column1, column2 = st.columns(2)

with column1:
    st.write('### Responde el siguiente test para descubrir tus preferencias cinematográficas y obtener una lista de recomendaciones. ¡Empecemos a explorar juntos!')
with column2:
    st.image('../images/sodadef-removebg-preview.png', use_column_width=False)

titles = pd.read_csv('../data/clean/titles.csv', encoding='utf-8', encoding_errors='ignore')

fav_film= titles['title'].tolist()

# Preguntas

pregunta1 = st.selectbox('¿Cuál es tu película o serie favorita?', fav_film)

pregunta2 = st.radio('¿Prefieres los clásicos o las producciones contemporáneas?', ['Clásicas', 'Contemporáneas', 'Ambas'])

pregunta3 = st.radio('¿Qué tipo de trama te resulta más interesante?', ['Misterio', 'Aventura','Fantasía', 'Comedia', 'Romance', 'Terror','¡Cualquiera!'])

#pregunta4 = st.text_input('Escribe el nombre del actor o la actriz que deba aparecer en tu lista (o no escribas ninguno)')

pregunta5 = st.radio('¿Prefieres que sean basadas en hechos reales o ficción?', ['Hechos reales', 'Ficción', '¡Cualquiera!'])

duracion_minima, duracion_maxima = st.slider('¿Cuánto debería durar?', 0, 300, (0, 300))

pregunta7 = st.radio('¿Te gustan los finales felices?', ['Sí', 'No', '¯\_(ツ)_/¯'])

pregunta8 = st.text_input('¿Tienes alguna temática, época o lugar favorito?')

pregunta4 = st.text_input('Escribe el nombre del actor o la actriz que deba aparecer en tu lista (o no escribas ninguno)')


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
}

def buscar_sinonimos(critica, sinonimos):
    for sinonimo in sinonimos:
        if sinonimo in critica:
            return True
    return False


#filtrar recomendaciones

def generar_recomendaciones(respuestas):

    comments = pd.read_csv('../data/clean/com_group.csv', encoding='utf-8', encoding_errors='ignore')

    # 1. Filtro por peli o serie fav
    tfidf = TfidfVectorizer(stop_words= 'english')              # Definir objeto vectorizador TF_IDF
    tfidf_matrix = tfidf.fit_transform(titles['description'])   # contruir matriz TF-IDF

    cosine_sim= linear_kernel(tfidf_matrix, tfidf_matrix)       # similitud de cosenos

    indices= pd.Series(titles.index, index= titles['title']).drop_duplicates()  # construir mapa inverso de indices y titulos de peliculas

    idx = indices[respuestas['pregunta1']]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_id = [i[0] for i in sim_scores]
    df_filtrado = titles.iloc[movie_id]
    df_filtrado = df_filtrado.iloc[1:]


    # 2. Filtro por antiguedad
    if respuestas['pregunta2'] == 'Clásicas':
        df_filtrado = df_filtrado[df_filtrado['release_year'] <= 1999]
    elif respuestas['pregunta2'] == 'Contemporáneas':
        df_filtrado = df_filtrado[df_filtrado['release_year'] > 1999]
    elif respuestas['pregunta2'] == 'Ambas':
        pass


# 3. Filtro por tipo de trama favorita
    sinonimos_misterio = ['mystery', 'enigma', 'puzzle', 'riddle', 'conundrum', 'secret', 'intrigue', 'clue', 'suspense']
    sinonimos_aventura = ['thrill', 'adventure', 'expedition', 'journey', 'quest', 'explor', 'trek', 'voyage', 'clue', 'safari', 'exploration']
    sinonimos_fantasia = ['fantasy', 'imagination', 'imaginary', 'enchant', 'magic', 'tail', 'fairy', 'myth', 'wonder', 'dream', 'illusion', 'super', 'superhero']
    sinonimos_romance = ['romance', 'love', 'affection', 'passion', 'relationship']
    sinonimos_comedia = ['comedy', 'humor', 'funny', 'laughter', 'laugh', 'crack up']
    sinonimos_terror = ['fear', 'scary', 'fright', 'terrify', 'scare']

    if respuestas['pregunta3'] == 'Comedia':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_comedia))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta3'] == 'Terror':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_terror))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta3'] == 'Romance':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_romance))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta3'] == 'Misterio':
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
    palabras_clave_ficcion = ['fiction', 'imaginary', 'anime', 'cartoon', 'fantasy', 'fictional', 'otherworld', 'extraterrestrial']
    
    if respuestas['pregunta5'] == "Ficción":
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, palabras_clave_ficcion))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta5'] == "Hechos reales":
        comentarios_filtrados = comments[~comments['review'].apply(lambda x: buscar_sinonimos(x, palabras_clave_ficcion))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta5'] == '¡Cualquiera!':
        pass


    # 6. Filtro por duración de la película
    df_filtrado = df_filtrado[(df_filtrado['runtime'] >= duracion_minima) & (df_filtrado['runtime'] <= duracion_maxima)]


    # 7. Filtro por final feliz
    sinonimos_final_feliz = ['happy ending', 'positive outcome', 'pleasant conclusion', 'satisfying resolution', 'joyful finale', 'contented ending', 'delightful outcome', 'cheerful conclusion', 'pleasant ending', 'uplifting finale']
    
    if respuestas['pregunta7'] == 'Sí':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_final_feliz))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta7'] == 'No':
        comentarios_filtrados = comments[~comments['review'].apply(lambda x: buscar_sinonimos(x, sinonimos_final_feliz))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta7'] == '¯\_(ツ)_/¯':
        pass


    # 8. Filtro por tema, época o lugar favorito
    if respuestas['pregunta8'] != '':
        tema = respuestas['pregunta8']
        comentarios_filtrados = comments[comments['review'].str.contains(tema, case=False)]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    else:
        pass


    return df_filtrado


# Generar recomendaciones
df_filtrado = None

if st.button("Generar recomendaciones"):
    loading_image = st.image('../images/eating-popcorn-ms-chalice.gif', use_column_width=True) 
    df_filtrado = generar_recomendaciones(respuestas)
    loading_image.empty()

    column1, column2 = st.columns(2)

    with column1:
        st.image('../images/popdef-removebg-preview.png', use_column_width=False)
    with column2:
        st.write('### Ahora podrás explorar diferentes gráficos sobre las películas y series que más se adaptan a ti')

    plt.style.use('dark_background')

    # Crear dos columnas en el layout de Streamlit
    column1, column2 = st.columns(2)

    # Gráfico de barras del año de lanzamiento
    with column1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='release_year', data=df_filtrado, ax=ax1)
        ax1.set_xlabel('Año de lanzamiento')
        ax1.set_ylabel('Número de películas')
        ax1.set_title('Distribución por año de lanzamiento')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax1.grid(False)  # Quitar la malla
        st.pyplot(fig1)

    # Histograma de duración de las películas
    with column2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df_filtrado, x='runtime', bins=20, color='red', ax=ax2)
        ax2.set_xlabel('Duración (minutos)')
        ax2.set_ylabel('Número de películas')
        ax2.set_title('Distribución de duración')
        ax2.grid(False)  # Quitar la malla
        st.pyplot(fig2)


    column1, column2 = st.columns(2)

    # Gráfico polar géneros
    with column1:
        df_filtrado['genres'] = df_filtrado['genres'].str.split(',').str[0]
        data = df_filtrado.groupby('genres').count().T.iloc[0]

        etiquetas=list(df_filtrado.genres.unique())
        angulos = np.linspace(0, 2*np.pi, len(etiquetas), endpoint=False)
        angulos=np.concatenate((angulos, [angulos[0]]))
        data = np.concatenate((data, [data[0]]))
        fig3 = plt.figure()

        ax = fig3.add_subplot(111, polar=True)
        ax.plot(angulos, data, 'o-', linewidth=2, color= 'red') 
        ax.fill(angulos, data, alpha=0.25, color= 'red') 
        ax.set_xticklabels([]) 
        ax.set_thetagrids(angulos * 180/np.pi, etiquetas+[etiquetas[0]])  
        ax.set_title('Distribución de géneros') 
        ax.grid(True, color= '#444444' )
        st.pyplot(fig3)

    with column2:
        platform_counts = df_filtrado['platform'].value_counts()
        labels = platform_counts.index.tolist()
        values = platform_counts.values.tolist()

        fig4, ax3 = plt.subplots(figsize=(8, 6))

        # Definir un diccionario de colores personalizados
        colors = {'Amazon': 'blue', 'Netflix': 'red', 'HBO': 'purple'}

        # Obtener el color correspondiente para cada etiqueta
        pie_colors = [colors.get(label, 'gray') for label in labels]

        ax3.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)

        ax3.set_title('Distribución de Plataformas')
        st.pyplot(fig4)



st.subheader('¡Prepara palomitas, aquí vienen tus recomendaciones!🍿')
if df_filtrado is not None and not df_filtrado.empty:
    recomendaciones_10 = df_filtrado['title'].tolist()[:10]
    column1, column2, column3 = st.columns(3)  # Divide el espacio en dos columnas
    with column1:
        st.subheader('Recomendaciones')
        for i, recomendacion in enumerate(recomendaciones_10, start=1):
            st.write(f'{i}. {recomendacion}')
else:
    st.write('Lo siento, no se encontraron recomendaciones para tus respuestas.')


if df_filtrado is not None and not df_filtrado.empty:
    actor_counts = df_filtrado['actors'].str.split(',').explode().str.strip().value_counts()
    actores_mas_comunes = actor_counts.sort_values(ascending=False).head(10)
    with column2:
        st.subheader('Actores más comunes')
        for actor, count in actores_mas_comunes.items():
            st.write(f'{actor}: {count} apariciones')
else:
    pass
    #st.write('No se encontraron datos para mostrar.')

if df_filtrado is not None and not df_filtrado.empty:
    recom_imdb = df_filtrado.sort_values('imdb_score', ascending=False)
    recom_10_imdb = recom_imdb['title'].tolist()[:10]
    with column3:
        st.subheader('Aclamadas por la crítica')
        for i, recomendacion in enumerate(recom_10_imdb, start=1):
            st.write(f'{i}. {recomendacion}')
else:
    pass













