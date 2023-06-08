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
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud

from googletrans import Translator
translator = Translator()

st.set_option('deprecation.showPyplotGlobalUse', False)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('../images/3968016.png')


st.image("../images/logo2.png", use_column_width=False)
st.title('Bienvenidos a FlickPick🖖')


column1, column2 = st.columns(2)

with column1:
    st.write('### Responde el siguiente test para descubrir tus preferencias cinematográficas y obtener una lista de recomendaciones.')
    st.write('### ¡Empecemos a explorar juntos!')
with column2:
    st.image('../images/sodadef-removebg-preview.png', use_column_width=False)

# Cargar el dataframe de títulos
titles = pd.read_csv('../data/clean/titles.csv', encoding='utf-8', encoding_errors='ignore')
fav_film = titles['title'].tolist()

# Preguntas
pregunta1 = st.selectbox('¿Cuál es tu película o serie favorita?', [''] + fav_film)
pregunta2 = st.radio('¿Prefieres los clásicos o las producciones contemporáneas?', ['Clásicas', 'Contemporáneas', 'Ambas'])
pregunta3 = st.radio('¿Qué tipo de trama te resulta más interesante?', ['Misterio', 'Aventura', 'Fantasía', 'Comedia', 'Romance', 'Terror', '¡Cualquiera!'])
pregunta5 = st.radio('¿Prefieres que sean basadas en hechos reales o ficción?', ['Hechos reales', 'Ficción', '¡Cualquiera!'])
duracion_minima, duracion_maxima = st.slider('¿Cuánto debería durar?', 0, 300, (0, 300))
pregunta7 = st.radio('¿Te gustan los finales felices?', ['Sí', 'No', '¯\_(ツ)_/¯'])
pregunta8 = st.text_input('¿Tienes alguna temática favorita?')
#pregunta4 = st.text_input('Escribe el nombre del actor o la actriz que deba aparecer en tu lista (o no escribas ninguno)')
pregunta9 = st.selectbox('Selecciona el método de búsqueda', ['Búsqueda rápida', 'Búsqueda exhaustiva'])


# Recolectar las respuestas
respuestas = {
    'pregunta1': pregunta1,
    'pregunta2': pregunta2,
    'pregunta3': pregunta3,
    #'pregunta4': pregunta4,
    'pregunta5': pregunta5,
    'pregunta6': (duracion_minima, duracion_maxima),
    'pregunta7': pregunta7,
    'pregunta8': pregunta8,
    'pregunta9': pregunta9,
}

def buscar_sinonimos(critica, sinonimos):
    for sinonimo in sinonimos:
        if sinonimo in critica:
            return True
    return False

# Filtrar recomendaciones
def generar_recomendaciones(respuestas):
    comments = pd.read_csv('../data/clean/critics.csv', encoding='utf-8', encoding_errors='ignore')
    if respuestas['pregunta9'] == 'Búsqueda rápida':

        # Filtrar por película o serie favorita
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(titles['description'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(titles.index, index=titles['title']).drop_duplicates()
        idx = indices[respuestas['pregunta1']]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_id = [i[0] for i in sim_scores]
        df_filtrado = titles.iloc[movie_id]
        df_filtrado = df_filtrado.iloc[1:]

    if respuestas['pregunta9'] == 'Búsqueda exhaustiva':

        # Filtrar por película o serie favorita
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(comments['review'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(comments.index, index=comments['title']).drop_duplicates()
        idx = indices[respuestas['pregunta1']]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_id = [i[0] for i in sim_scores]
        df_filtrado = comments.iloc[movie_id]
        df_filtrado = df_filtrado.iloc[1:]
        df_filtrado = df_filtrado.merge(titles.rename(columns={'title': 'title_original'}), on='imdb_id', how='inner')



    # 2. Filtro por antiguedad
    if respuestas['pregunta2'] == 'Clásicas':
        df_filtrado = df_filtrado[df_filtrado['release_year'] <= 1999]
    elif respuestas['pregunta2'] == 'Contemporáneas':
        df_filtrado = df_filtrado[df_filtrado['release_year'] > 1999]
    elif respuestas['pregunta2'] == 'Ambas':
        pass


    # 3. Filtro por tipo de trama favorita
    palabras_clave_trama = {
        'Comedia': ['comedy', 'humor', 'funny', 'laughter', 'laugh', 'crack up'],
        'Terror': ['fear', 'scary', 'fright', 'terrify', 'scare'],
        'Romance': ['romance', 'affection', 'passion', 'relationship'],
        'Misterio': ['mystery', 'enigma', 'puzzle', 'riddle', 'conundrum', 'secret', 'intrigue', 'clue', 'suspense'],
        'Aventura': ['thrill', 'adventure', 'expedition', 'journey', 'quest', 'explor', 'trek', 'voyage', 'clue', 'safari', 'exploration'],
        'Fantasía': ['fantasy', 'imagination', 'imaginary', 'enchant', 'magic', 'tail', 'fairy', 'myth', 'wonder', 'dream', 'illusion', 'super', 'superhero']
    }

    tipo_trama = respuestas['pregunta3']
    if tipo_trama in palabras_clave_trama:
        comentarios_filtrados = comments[comments['review'].apply(lambda x: any(palabra_clave in x for palabra_clave in palabras_clave_trama[tipo_trama]))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')



    # 6. Filtro para determinar ficción
    palabras_clave_real = ['based on true events', 'true story', 'real story', 'real-life', 'factual', 'documentary']

    if respuestas['pregunta5'] == "Ficción":
        comentarios_filtrados = comments[~comments['review'].apply(lambda x: any(palabra_clave in x for palabra_clave in palabras_clave_real))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta5'] == "Hechos reales":
        comentarios_filtrados = comments[comments['review'].apply(lambda x: any(palabra_clave in x for palabra_clave in palabras_clave_real))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')



    # 6. Filtro por duración
    df_filtrado = df_filtrado[(df_filtrado['runtime'] >= duracion_minima) & (df_filtrado['runtime'] <= duracion_maxima)]


    # 7. Filtro por final feliz
    sinonimos_final_feliz = ['happy ending', 'positive outcome', 'pleasant conclusion', 'satisfying resolution', 'joyful finale', 'contented ending', 'delightful outcome', 'cheerful conclusion', 'pleasant ending', 'uplifting finale']

    if respuestas['pregunta7'] == 'Sí':
        comentarios_filtrados = comments[comments['review'].apply(lambda x: any(sinonimo in x for sinonimo in sinonimos_final_feliz))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    elif respuestas['pregunta7'] == 'No':
        comentarios_filtrados = comments[~comments['review'].apply(lambda x: any(sinonimo in x for sinonimo in sinonimos_final_feliz))]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')



    # 8. Filtro por tema, época o lugar favorito
    if respuestas['pregunta8'] != '':
        tema_espanol = respuestas['pregunta8']
        tema_ingles = translator.translate(tema_espanol, src='es', dest='en').text
        comentarios_filtrados = comments[comments['review'].str.contains(tema_ingles, case=False)]
        df_filtrado = df_filtrado.merge(comentarios_filtrados[['imdb_id']], on='imdb_id', how='inner')
    else:
        pass

    # 4. Filtro por actor o actriz favorito
    #if respuestas['pregunta4'] != '':
        #actor_favorito = respuestas['pregunta4']
        #df_filtrado = df_filtrado[df_filtrado['actors'].str.contains(actor_favorito, case=False)]
    #else:
        #pass
    

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

    column1, column2 = st.columns(2)

    # Gráfico para año de lanzamiento
    with column1:
        movie_counts = df_filtrado[df_filtrado['type'] == 'MOVIE']['release_year'].value_counts().sort_index()
        series_counts = df_filtrado[df_filtrado['type'] == 'SHOW']['release_year'].value_counts().sort_index()
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.fill_between(movie_counts.index, movie_counts.values, color='orange', alpha=0.5)
        ax1.fill_between(series_counts.index, series_counts.values, color='green', alpha=0.5)
        sns.lineplot(x=movie_counts.index, y=movie_counts.values, color='orange', label='Películas')
        sns.lineplot(x=series_counts.index, y=series_counts.values, color='green', label='Series')
        
        max_movie_year = movie_counts.idxmax()
        max_series_year = series_counts.idxmax()
        max_movie_count = movie_counts.max()
        max_series_count = series_counts.max()
        
        ax1.axvline(x=max_movie_year, color='orange', linestyle='--', label=f'Año ideal de producción de películas ({max_movie_year})')
        ax1.axvline(x=max_series_year, color='green', linestyle='--', label=f'Año ideal de producción de series ({max_series_year})')
        
        ax1.set_xlabel('Año de lanzamiento')
        ax1.set_ylabel('Número de títulos')
        ax1.set_title('Distribución de películas y series por año de lanzamiento')
        ax1.legend()
        ax1.grid(False)
        
        st.pyplot(fig1)



    # Gráfico para distribución de duración de las películas
    with column2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        movies_data = df_filtrado[df_filtrado['type'] == 'MOVIE']
        series_data = df_filtrado[df_filtrado['type'] == 'SHOW']
        
        ax2.hist(movies_data['runtime'], bins=20, color='orange', alpha=0.7, label='Películas')
        ax2.hist(series_data['runtime'], bins=20, color='green', alpha=0.7, label='Series')
        
        ax2.set_xlabel('Duración (minutos)')
        ax2.set_ylabel('Número de títulos')
        ax2.set_title('Distribución de duración')
        ax2.legend()
        ax2.grid(False)
        
        # Resaltar duración con mayor número de películas
        max_movies_duration = movies_data['runtime'].value_counts().idxmax()
        ax2.axvline(max_movies_duration, color='orange', linestyle='--', linewidth=2, label=f'Duración ideal de películas({max_movies_duration} min)')
        max_series_duration = series_data['runtime'].value_counts().idxmax()
        ax2.axvline(max_series_duration, color='green', linestyle='--', linewidth=2, label=f'Duración ideal de series({max_series_duration} min)')
        
        ax2.legend()
        st.pyplot(fig2)


    column1, column2 = st.columns(2)

    # Gráfico por puntuación en imdb
    with column1:
        platforms = ['Netflix', 'HBO', 'Amazon']
        filtered_df = df_filtrado[df_filtrado['platform'].isin(platforms)]
        platform_ratings = filtered_df.groupby('platform')['imdb_score'].mean()

        fig5, ax5 = plt.subplots(figsize=(8, 6))
        colors = ['purple' if p == 'HBO' else 'red' if p == 'Netflix' else 'blue' for p in platform_ratings.index]
        bars = ax5.bar(platform_ratings.index, platform_ratings.values, color=colors)
        ax5.set_xlabel('Plataforma')
        ax5.set_ylabel('Valoración media')
        ax5.set_title('Valoración media por plataforma')

        # Agregar etiquetas de puntuación a cada barra
        for bar in bars:
            height = bar.get_height()
            ax5.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom')

        st.pyplot(fig5)




    # Gráfico de tarta por plataforma
    with column2:
        platform_counts = df_filtrado['platform'].value_counts()
        labels = platform_counts.index.tolist()
        values = platform_counts.values.tolist()

        fig4, ax3 = plt.subplots(figsize=(8, 6))
        colors = {'Amazon': 'blue', 'Netflix': 'red', 'HBO': 'purple'}
        pie_colors = [colors.get(label, 'gray') for label in labels]

        explode = [0.1 if label == labels[0] else 0 for label in labels]
        ax3.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=pie_colors, explode=explode)
        ax3.set_title('Distribución de Plataformas')
        st.pyplot(fig4)

    
    # Gráfico polar géneros
    df_filtrado['genres'] = df_filtrado['genres'].str.split(',').str[0]
    data = df_filtrado.groupby('genres').count().T.iloc[0]

    etiquetas=list(df_filtrado.genres.unique())[:50]
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




if df_filtrado is not None and not df_filtrado.empty:
    st.subheader('¡Prepara palomitas, aquí vienen tus recomendaciones!🍿')
    recomendaciones_10 = df_filtrado['title'].tolist()[:10]
    column1, column2, column3 = st.columns(3)  # Divide el espacio en dos columnas
    with column1:
        st.subheader('Basadas en tu perfil')
        for i, recomendacion in enumerate(recomendaciones_10, start=1):
            st.write(f'{i}. {recomendacion}')
else:
    pass
    #st.write('Lo siento, no se encontraron recomendaciones para tus respuestas.')


if df_filtrado is not None and not df_filtrado.empty:
    titulos_sup8 = df_filtrado[df_filtrado['imdb_score'] > 8]
    recom_10_imdb = titulos_sup8['title'].tolist()[:10]
    with column2:
        st.subheader('Aclamadas por la crítica')
        for i, recomendacion in enumerate(recom_10_imdb, start=1):
            st.write(f'{i}. {recomendacion}')
else:
    pass


if df_filtrado is not None and not df_filtrado.empty:
    actor_counts = df_filtrado['actors'].str.split(',').explode().str.strip().value_counts()
    actores_mas_comunes = actor_counts.sort_values(ascending=False).head(10)
    with column3:
        st.subheader('Actores más comunes')
        for actor, count in actores_mas_comunes.items():
            st.write(f'{actor}: {count} apariciones')
else:
    pass
    #st.write('No se encontraron datos para mostrar.')















