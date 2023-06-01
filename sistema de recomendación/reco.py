import streamlit as st
import pandas as pd
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

st.set_option('deprecation.showPyplotGlobalUse', False)


image_path = "../images/Sweet_Popcorn__1_-removebg-preview (1).png"

st.image(image_path, use_column_width=False)


st.write('# Bienvenidos a FlickPick🖖')
st.write('## Tu recomendador de películas personalizado, MUY PERSONALIZADO.')
st.write('### Nuestro objetivo es ayudarte a descubrir películas y series que se adapten perfectamente a tí. ¡Empecemos a explorar juntos!')

st.sidebar.header('FlickPick Navigator')
st.sidebar.subheader('Streamlit Recom')
st.sidebar.info('Aquí puedes poner una barra de navegación o zonas para cargar archivos')


# Preguntas

pregunta1 = st.text_input('¿Cuál es tu película o serie favorita?')

pregunta2 = st.radio('¿Prefieres los clásicos o las producciones contemporáneas?', ['Clásicas', 'Contemporáneas', 'Ambas'])

pregunta3 = st.radio('¿Qué tipo de trama te resulta más interesante?', ['Misterio', 'Aventura','Fantasía', '¡Cualquiera!'])

pregunta4 = st.text_input('Escribe el nombre del actor o la actriz que deba aparecer en tu lista (o no escribas ninguno)')

pregunta5 = st.radio('¿Prefieres películas basadas en hechos reales o ficción?', ['Hechos reales', 'Ficción', '¡Cualquiera!'])

duracion_minima, duracion_maxima = st.slider('¿Cuánto debería durar?', 0, 300, (0, 300))

pregunta7 = st.radio('¿Te gustan los finales felices?', ['¿A quién no le va a gustar?', 'No', '¯\_(ツ)_/¯'])

pregunta8 = st.text_input('¿Tienes algún tema concreto, época o lugar favorito?')


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
        df_filtrado = df_filtrado[df_filtrado['release_year'] <= 1990]
    elif respuestas['pregunta2'] == 'Contemporáneas':
        df_filtrado = df_filtrado[df_filtrado['release_year'] > 1990]
    elif respuestas['pregunta2'] == 'Ambas':
        pass


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
    df_filtrado = df_filtrado[(df_filtrado['runtime'] >= duracion_minima) & (df_filtrado['runtime'] <= duracion_maxima)]


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


    return df_filtrado


# Generar recomendaciones
df_filtrado = None

if st.button("Generar recomendaciones"):
    df_filtrado = generar_recomendaciones(respuestas)

st.subheader('¡Prepara palomitas, aquí vienen tus recomendaciones!🍿')
if df_filtrado is not None and not df_filtrado.empty:
    recomendaciones_10 = df_filtrado['title'].tolist()[:10]
    for recomendacion in recomendaciones_10:
        st.write(recomendacion)
else:
    st.write('Lo siento, no se encontraron recomendaciones para tus respuestas.')

    

# Gráfico de barras del año de lanzamiento
plt.figure(figsize=(8, 6))
sns.countplot(x='release_year', data=df_filtrado)
plt.xlabel('Año de lanzamiento')
plt.ylabel('Número de películas')
plt.title('Distribución de películas por año de lanzamiento')
st.pyplot()

# Histograma de duración de las películas
plt.figure(figsize=(8, 6))
sns.histplot(data=df_filtrado, x='runtime', bins=20)
plt.xlabel('Duración (minutos)')
plt.ylabel('Número de películas')
plt.title('Distribución de duración de las películas')
st.pyplot()





