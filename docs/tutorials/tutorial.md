## Tutoriel d'Utilisation : Standalone Camera Classifier

Ce guide pas-à-pas explique comment configurer, lancer et vérifier l'application de classification d'images en temps réel.

### 1\. Préparation de l'Environnement

Avant de commencer, assurez-vous que **Python 3.12+** et le gestionnaire de paquets **Poetry** sont installés sur votre système.

#### A. Clonage du Dépôt

Ouvrez votre terminal (PowerShell ou Bash) et téléchargez le projet :

```bash
git clone https://github.com/Rediiel/standalone-camera-classify.git
cd standalone-camera-classify
```

#### B. Installation des Dépendances

Installez toutes les bibliothèques requises (y compris OpenCV, scikit-learn, etc.) en utilisant Poetry. Cette commande configure également l'environnement virtuel du projet :

```bash
poetry install
```

### 2\. Lancer l'Application de Classification

L'application est lancée directement via Poetry. Elle utilise le modèle pré-entraîné qui se trouve dans le dossier `model/`.

#### A. Démarrage de la Caméra

Utilisez la commande suivante pour exécuter le script principal :

```bash
poetry run camera_classify
```

#### B. Utilisation

  * Une fenêtre **OpenCV** s'ouvrira, affichant le flux vidéo de votre caméra par défaut.
  * Pointez la caméra vers un objet appartenant aux classes entraînées (ex. : `laptop`, `mouse`, `headset`).
  * Si le modèle est suffisamment confiant, une **bande noire** en bas de la fenêtre affichera le nom de la classe reconnue.
  * **Pour quitter l'application**, appuyez sur la touche `q` ou `ESC`.

### 3\. Vérification de la Qualité et des Tests

Pour les développeurs et les contributeurs, il est essentiel de s'assurer que le code est conforme au style défini et que tous les tests unitaires passent.

#### A. Exécution de l'Analyse Statique (Style et Erreurs)

Pour vérifier manuellement la conformité au formatage **Black** et aux règles de qualité **Flake8** (longueur de ligne max 88), utilisez :

```bash
poetry run black .
poetry run flake8 .
```

#### B. Lancer les Tests Unitaires

Les tests sont configurés pour vérifier les fonctions d'extraction et d'ajustement des caractéristiques. Pour les exécuter :

```bash
poetry run test
```

#### C. Activer l'Automatisation

Pour exécuter l'analyse statique **automatiquement** avant chaque `git commit`, installez le hook de pré-commit :

```bash
poetry run pre-commit install
```

### 4\. Générer la Documentation de Référence

Si vous avez modifié les docstrings du code, vous pouvez générer une documentation HTML complète :

1.  Placez-vous dans le dossier de documentation :
    ```bash
    cd docs
    ```
2.  Générez les pages HTML :
    ```bash
    make html
    ```
    La documentation navigable se trouve désormais dans le dossier `docs/_build/html`.