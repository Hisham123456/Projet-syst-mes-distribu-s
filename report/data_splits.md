# Données figées pour les expériences CPU et GPU

Les fichiers `data/splits/train.csv`, `val.csv` et `test.csv` constituent
la référence commune de toutes les expériences. Ne pas recréer une partition
pour chaque modèle ou matériel. `read_split("data/splits", "train")` (ou `val`,
`test`) vérifie l'empreinte SHA-256 du CSV avant de le lire.

La livraison figée comprend 2 999 images lisibles et 1 918 patients supposés :

| Ensemble | Images | Proportion | Patients supposés |
| --- | ---: | ---: | ---: |
| Entraînement | 2 094 | 69,82 % | 1 351 |
| Validation | 457 | 15,24 % | 285 |
| Test | 448 | 14,94 % | 282 |

Les effectifs totaux des classes 0 à 4 sont respectivement 504, 701, 583,
632 et 579. Les cinq classes figurent dans chaque ensemble. Aucun patient,
identifiant d'image ni fichier identique ne chevauche les ensembles.

Chaque ligne contient le chemin relatif à `data/FDR26_classification/images`,
l'identifiant de l'image, la classe DR (0 à 4), le patient supposé,
l'empreinte SHA-256 de l'image et son groupe de partitionnement.

L'Excel `data/FDR26_classification/classification DR.xlsx` reste inchangé.
Ses 3 000 entrées correspondent à 2 999 images disponibles : `2T2SALA702`
(classe 0) est explicitement exclue parce que son fichier est absent.
Cette exclusion est consignée dans `audit.json`. Toute autre absence,
image illisible ou image sans label bloque la génération.

**Hypothèse provisoire : les huit premiers caractères du nom identifient
un patient.** Elle est inférée des noms et n'est pas une identité clinique
validée. Toutes les images d'un même préfixe restent dans un seul ensemble.
Les patients reliés par des fichiers identiques sont également regroupés
pour empêcher une fuite entre ensembles.
Les doublons dont les labels diffèrent sont signalés dans l'audit. Les labels
originaux sont conservés, sans arbitrage automatique ; ces conflits restent
une limite de qualité des annotations à examiner avant d'interpréter les scores.
Cette livraison comporte 160 groupes de fichiers identiques, dont 62 avec
des labels contradictoires ; les identifiants concernés figurent dans l'audit.

La graine est 42. Le script sélectionne parmi 256 partitions par groupes
à chaque étape pour approcher les proportions 70/15/15 et conserver les cinq
classes dans chaque ensemble. Les proportions exactes et les effectifs par
classe et ensemble sont enregistrés dans `audit.json`, avec les contrôles
de lecture intégrale des images, de couverture et d'absence de chevauchement.

La commande PowerShell de génération initiale est :

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m dr_project.splits
```

Elle refuse d'écraser des partitions existantes. Pour examiner une nouvelle
version du jeu de données, utiliser explicitement un autre dossier `--output`
et conserver les partitions de référence.
