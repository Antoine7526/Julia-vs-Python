# Comparaison Julia, Python et R sur les CNN (Convutionnal Neural Network)

<div style="text-align:justify;"> 

## Contexte du projet :
Nous avons voulu confectionner un algorithme nous permettant de faire de la classification d'images.
Pour cela, nous avons utilisé une base de données provenant de Kaggle. Cette dernière nous permettra de classifier des 
images de rugby et de football.

Nous avons réalisé ce projet sur Python et sur Julia. Comme nous avons fait de la classification d'images à l'aide de 
réseau de neurones, nous n'avons pas pu implementer un modèle sur ```R```, car le langage utilise des packages ```Python```.
Nous avons constaté que c'était plus simple d'implementer ce projet sur Python que sur Julia. En effet, 
les images sur ```Julia```, pour être utilisées, doivent subir de plus lourdes modifications sur la qualité et le format
que sur Python.

Il est important de préciser que dans les deux langages, il y a de nombreux packages permettant d'effectuer cette tâche. 
Étant donné nos connaissances sur ```Keras``` et ```Tensorflow```, nous avons choisi de travailler avec le package 
le plus ressemblant sur ```Julia```, à savoir ```Flux```.

Constatons les similitudes et les différences de ces deux langages.

## Ressemblances :

Tout d'abord, nous avons constaté que la manière d'implémenter un réseau de neurones sur Python et Julia est quasiment identique. 
En effet, la façon d'utiliser la méthode ```Chain```de ```Flux``` est la même que ```Sequential``` de ```keras```.
Enfin, la partie syntaxique reste globalement la même d'un langage à l'autre.

## Différences : 

Contrairement à ```Python```, ```Julia``` propose une vectorization, ce qui facilite l'application d'une fonction 
à un ensemble d'objets. Cela permet de réduire considérablement le nombre de boucles, rendant le code plus fluide et 
rapide. Nous avons pu voir cette différence lors de la partie "preprocessing" de notre projet.  
En revanche, ```Python``` est beaucoup plus simple à utiliser dans le sens où ce langage possède des packages complets 
et adaptés aux tâches de preprocess et deep learning. De plus, les documentations et les forums (par ex StackOverflow),
sont bien plus complets que pour ```Julia```. En effet, il était parfois difficile de trouver certaines informations.


## Pour aller plus loin :

```FastAI.jl``` est la librairie ```Julia``` pensée pour effectuer des tâches de deep learning complexe. La logique reste la même 
qu’avec les autres langages de programmation (former un jeu de données, déterminer une tâche d'apprentissage, procéder 
à un apprentissage, l’adapter à un modèle et finir par des prédictions et des évaluations). Pour répondre à notre 
problématique, ```FastAI``` aurait pu nous être fortement utile, mais par manque de temps, d'expérience sur ```Julia``` nous avons 
fait le choix de ne pas l’utiliser. En effet, comme nous l’avons mentionné plus haut, le manque de documentation et d’exemples 
de ```Julia``` et de ```FastAI``` a représenté pour nous une barrière quant à l’application des méthodes de cette librairie 
à notre jeu de données. Nous regrettons le fait de ne pas avoir pu aller plus loin avec ```FastAI``` car cette extension 
paraissait plus adaptée et plus performante que ```Flux```. Ces faiblesses et forces sont à mettre en lien avec la jeunesse 
du package. Sûrement que dans les années futures, l’utilisation de ```FastAI``` sera une évidence quant à la réponse à 
notre problématique.

## Auteurs :
Marion Borel, Bastien Gévaudan, Antoine Grancher & Julien Soufflet

</div>