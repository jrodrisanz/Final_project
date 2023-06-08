# Proyecto Final - Sistema de Recomendación de Películas y Series

[![logo2.png](https://i.postimg.cc/h47fr08n/logo2.png)](https://postimg.cc/kDqJMKhY)

## Descripción

¡Hola a todos!

Este es mi proyecto final para el Bootcamp de Data Analytics en Ironhack. Espero que os guste.

En este proyecto he desarrollado un sistema de recomendación de películas y series basado en el contenido y las preferencias del usuario. El sistema utiliza algoritmos de filtrado colaborativo y filtrado basado en contenido para generar recomendaciones personalizadas.

El usuario debe responder un listado de preguntas tipo test para generar sus recomnedaciones. Una vez se hayan ordenado las películas y series en función de sus gustos, apareceran una serie de gráficos que muestran su personalidad cinematográfica.

## Metodología

Preguntas del tipo test:

1. ¿Cuál es tu película o serie favorita?
2. ¿Prefieres los clásicos o las producciones contemporáneas?
3. ¿Qué tipo de trama te resulta más interesante?
4. ¿Prefieres que sean basadas en hechos reales o ficción?
5. ¿Cuánto debería durar?
6. ¿Te gustan los finales felices?
7. ¿Tienes alguna temática favorita?
8. Selecciona el método de búsqueda


A través de la primera pregunta, ordenamos el dataframe en función de la correlación que tenga el contenido de la película o serie favorita con el resto de películas y series. Esto lo conseguimos vectorizando las palabras contenidas en la descripción y calculando la similitud entre ellas. 

De la segunda a la sexta pregunta hemos establecido una serie de parámetros que filtran el dataframe resultante de la respuesta anterior.

La séptima pregunta permite al usuario elegir un tema, lugar o época determinado y nos escoge las películas o series que contengan esa temática en la descripción o en algún comentario.

La última pregunta permite escoger la búsqueda rápida (calculando la similitud de cosenos con las descripciones) o la búsqueda exhaustiva (analizando la similitud de cosenos de comentarios y críticas asociadas a cada película o serie). El primer método es más rápido pero resulta menos preciso.

Finalmente nos aparecerán 5 gráficos:

1. El año de producción de películas y series que más se ajusta a nuestros gustos
2. La duración de películas y series que más se ajusta a nuestros gustos
3. La valoración de nuestras recomendaciones en las diferentes plataformas
4. La cantidad de contenido que hay que coincide con nuestros gustos en las diferentes plataformas
5. Cuáles son nuestros géneros favoritos


Por último, nos mostrará una lista con las 10 mejores recomendaciones ordenadas de mayor a menos correlación con nuestros gustos, en la segunda columna nos mostrará la misma lista filtrada con aquellas series y películas con buena puntuación en IMDB y en la tercera columna aparecerán el nombre de los actores que más aparecen en nuestras recomendaciones

## Créditos

Agradezco a mis compañeros por haber convertido este curso en una experiencia única, por haberme ayudado en todo momento y por haber creado un gran ambiente de retroalimentación y motivación que ha conseguido sacar lo mejor de cada uno de nosotros. Estoy especialmente agradecido con mis profesores, Yonatan, Ori y Carlos, los cuáles han conseguido transmitir su pasión por la enseñanza y la materia.

