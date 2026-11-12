# A3 : cache sur les données filtrées

Branche : `experiences/sans-labels-contradictoires`.

`configs/cpu-filtre-cache.yaml` reprend `configs/cpu-filtre.yaml` et ajoute
uniquement `data.cache_dir: data/cache`. Les images aux labels contradictoires
restent exclues. Les partitions figées sont identiques avec et sans cache :
1996 images d'entraînement, 445 de validation et 434 de test.

Le cache mémorise le décodage RGB et le redimensionnement. Le prétraitement
reste exécuté à chaque passage. Le code A3 et ses tests proviennent de `main`
(commit `fb2a361`), en conservant le chargement des partitions et le filtrage
propres à cette branche.

Depuis la racine du projet, préparer le cache avant les entraînements :

```powershell
.\.venv\Scripts\python.exe benchmarks/benchmark_cache.py --config configs/cpu-filtre-cache.yaml --out results/cache_loading_filtre.json
```

Ce benchmark vérifie aussi les pixels de 64 images et un lot de validation
chargé avec les workers. Le coût de création du cache est mesuré séparément.

Exécuter ensuite les quatre entraînements séquentiellement, dans cet ordre :

```powershell
.\.venv\Scripts\python.exe src/train.py --config configs/cpu-filtre.yaml --device cpu --run a3-filtre-sans-cache-1 --monitor
.\.venv\Scripts\python.exe src/train.py --config configs/cpu-filtre-cache.yaml --device cpu --run a3-filtre-avec-cache-1 --monitor
.\.venv\Scripts\python.exe src/train.py --config configs/cpu-filtre-cache.yaml --device cpu --run a3-filtre-avec-cache-2 --monitor
.\.venv\Scripts\python.exe src/train.py --config configs/cpu-filtre.yaml --device cpu --run a3-filtre-sans-cache-2 --monitor
```

Brancher le PC et conserver les mêmes conditions pendant les quatre mesures.
Chaque exécution utilise cinq époques et écrit dans un dossier distinct sous
`results/`. Pour une nouvelle série, utiliser de nouveaux noms `--run` afin de
conserver les résultats précédents.

Comparer `wall_time_s`, `steady_epoch_s` et les métriques de validation dans
les fichiers `history.json`. Le gain du cache se mesure entre ces configurations
filtrées. La comparaison directe avec les anciens essais sur `main` ne permet
pas d'isoler l'effet des labels : leurs partitions diffèrent.
