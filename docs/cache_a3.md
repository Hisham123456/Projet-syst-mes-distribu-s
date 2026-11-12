# A3 — Cache des images décodées

La configuration `configs/cpu-cache.yaml` active un cache NumPy en uint8
après conversion RGB et redimensionnement. Chaque ensemble utilise son
propre fichier. Le prétraitement est appliqué après lecture du cache.

La clé dépend des chemins ordonnés, des dimensions cibles, de la taille
et de la date de modification des fichiers. La création utilise huit
threads et publie le fichier complet par renommage. Les processus de
chargement Windows rouvrent les fichiers en lecture seule.

## Vérification et mesure du chargement

```powershell
.venv/Scripts/python.exe -m pytest tests/test_data_pipeline.py -v
.venv/Scripts/python.exe benchmarks/benchmark_cache.py
```

Le benchmark mesure le chargement direct et la lecture du cache sur les
mêmes 64 images, avec trois répétitions. Il enregistre aussi la durée
de préparation des caches des trois ensembles, leur taille et le coût
de réouverture dans `results/cache_loading.json`. Il compare les pixels
des 64 images et un lot de validation obtenu avec les deux processus
de chargement habituels.

Le champ `cache_files_present_before` indique si des caches existaient
avant le lancement. Le système d'exploitation peut conserver des pages
en mémoire. Les temps du cache réutilisé doivent être interprétés dans
ces conditions.

## Mesure de l'entraînement

Avec le cache déjà construit par le benchmark, lancer cinq époques.

```powershell
.venv/Scripts/python.exe src/train.py --config configs/cpu-cache.yaml --device cpu --run cpu-cache-warm --monitor
```

Pour mesurer également la construction pendant un entraînement, copier
la configuration et donner à `data.cache_dir` un nouveau dossier vide.
Attribuer un nom distinct à cet essai. Comparer `setup_time_s`,
`wall_time_s` et `steady_epoch_s`, et vérifier les scores obtenus.
Le gain sur le chargement seul et le gain sur l'entraînement complet
sont deux mesures distinctes.
