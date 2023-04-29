# README

Bienvenue au mon dépôt du projet "Génération automatique de la musique". Il contient l'ensemble de code qui permet de générer de la musique en utilisant un réseau "apprentissage profond".

## Avant de lancer...

Il y a plusieurs fichiers dans le dépôt. Des programmes sont écrits en `Python`, le dataset est sous le format `midi`. Il existe des fichiers qui permet d'économiser le temps de fonctionnement après le premier lancement avec l'information extraité et le modèle "checkpoint", lisible par le framework `TensorFlow`. À cause de la limite de GitHub, certains gros fichiers ne sont pas mis dans le dépôt.

Afin d'avoir tout fonctionner, veuillez créer :

- un dossier nommé `train_data`
- un dossier nommé `checkpoints`

D'ailleurs, le programme utilse ces modules suivants :

- TensorFlow
- numpy
- mido
- matplotlib

Pour visualizer le résultat, il faut aussi l'application `Lilypond`.

## Ordre du lancement

Il faut premièrement lancer le script `training_data_gen.py`, ensuite `training_data_gen2.py`. Ces scripts créent des datasets sous le format de `npy`, lisible pour le module `numpy` de Python.

Ensuite, il faut lancer le programme `model_1.py` et `model_2.py`. Les programmes entraînent des modèles et les sauvegardent comme "checkpoint".

Finalement, on peut lancer le programme `score_generation` pour générer des partitions.
