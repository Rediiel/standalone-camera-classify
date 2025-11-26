# Standalone Camera Classifier

[](https://www.google.com/search?q=LICENSE)

Cette application Python accède à la caméra de votre ordinateur pour effectuer une classification d'images en temps réel. Elle identifie l'objet en face de la caméra en utilisant un modèle pré-entraîné stocké dans le dossier `model/`.

## Classes Identifiables

Le système a été entraîné pour reconnaître les classes suivantes :

  * backpack
  * bag
  * gabriel (human with white skin tone)
  * glasses
  * headset
  * keyboard
  * laptop
  * monitor
  * mouse
  * pen

-----

## Prérequis

Vous devez avoir **Python 3.12+** et **Poetry** installés sur votre système.

### Installation

Clonez le dépôt et installez les dépendances :

```bash
git clone https://github.com/Rediiel/standalone-camera-classify.git
cd standalone-camera-classify
# Installe toutes les dépendances de production et de développement
poetry install
```

-----

## Utilisation de l'Application

### Lancer la Classification en Temps Réel

Exécutez l'application principale via Poetry. Une fenêtre OpenCV s'ouvrira, affichant la vidéo en direct et le nom de la classe identifiée.

```bash
poetry run camera_classify
```

### Créer le Package (Build)

Pour construire le package distribuable (wheel et sdist) :

```bash
poetry build
```

-----

## Qualité du Code (Analyse Statique & Tests)

Ce projet utilise des outils d'analyse statique et de test pour garantir la qualité et la fiabilité du code.

### 1\. Analyse Statique (Black & Flake8)

Les outils d'analyse statique sont configurés pour une longueur de ligne maximale de 88 caractères.

  * **Vérification manuelle :**
    ```bash
    poetry run black .
    poetry run flake8 .
    ```

### 2\. Automatisation (Hooks de Pré-commit)

Vous pouvez automatiser l'exécution de ces outils avant chaque `git commit`.

1.  **Installer le hook :**
    ```bash
    poetry run pre-commit install
    ```
2.  **Fonctionnement :** Dès lors, Black et Flake8 s'exécuteront automatiquement. Si une erreur est détectée, le commit sera bloqué jusqu'à ce que les problèmes soient corrigés.

### 3\. Tests Unitaires

Les tests unitaires vérifient la logique d'extraction et d'ajustement des caractéristiques à travers 16 tests.

  * **Exécuter les tests :**
    ```bash
    poetry run test
    ```

### 4\. Journalisation (Logging)

La fonction `print()` est remplacée par le module `logging` pour une meilleure traçabilité :

  * Les étapes importantes sont tracées avec `INFO`.
  * Les erreurs critiques (ex. : caméra non trouvée) sont signalées avec `ERROR`.

-----

## Documentation

La documentation de référence est générée à l'aide de **Sphinx** à partir des docstrings du code.

### Générer la documentation de référence

1.  Naviguez vers le dossier de documentation :
    ```bash
    cd docs
    ```
2.  Générez les pages HTML :
    ```bash
    make html
    ```
    Les fichiers générés se trouveront dans `docs/_build/html`.

### Tutoriel

Un tutoriel pas-à-pas simple pour le projet se trouve ici : [docs/tutorials/tutorial.md](./docs/tutorials/tutorial.md).



## Licence

Ce projet est distribué sous licence **MIT**. Voir le fichier [LICENSE](https://www.google.com/search?q=./LICENSE) pour plus de détails.