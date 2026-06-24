# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Les instructions globales (langue française, venvs, CUDA par défaut, barre de progression `tqdm`, pas de `pip install` sans accord, checkpoint git avant gros refactor, jamais de `—`) restent valables. Ce fichier ne couvre que ce qui est **spécifique à ce projet**.

## Objectif du projet

Construire **from scratch** un *world model* de style **JEPA (Joint-Embedding Predictive Architecture, LeCun)** et l'appliquer à **Snake**. La particularité : le modèle **agit lui-même par planification (MPC) dans l'espace latent** - il n'y a **pas de policy apprise séparée** (ce n'est donc pas Dreamer). L'action émerge directement de la prédiction du futur.

En une phrase : *un mini-Dreamer sans décodeur (le bloc prédictif est un JEPA), piloté par une recherche MPC dans l'espace latent au lieu d'un acteur appris.*

## Décisions de design (verrouillées)

- **Observation** = grille brute multi-canaux `(3, 8, 16)` : canaux tête / corps / pomme. **Ne PAS** utiliser le vecteur de 16 distances hand-crafted de `snake.py` (il tue l'intérêt de l'apprentissage de représentation).
- **Contrôle** = MPC latent, aucune policy entraînée. À chaque pas : encoder l'état, imaginer les 4 actions (puis des séquences sur un horizon court), scorer chaque futur, jouer l'argmax.
- **Critère de décision** = têtes **reward + done prédites** depuis le latent. Le planner maximise la récompense imaginée en évitant la mort.
- **Fidélité** = version maison simplifiée, inspirée des papiers (I-JEPA, V-JEPA 2-AC, DINO-WM) sans les reproduire.

## Architecture cible

```
  observation (3×8×16)                                ┌─► RewardHead(s_t, a) ─► r̂
        │                                             ├─► DoneHead(s_t, a)   ─► p(mort)
        ▼                                             │
   Encoder (CNN) ─► s_t ─► Predictor(s_t, a_t) ─► ŝ_{t+1}
        ▲                          │
o_{t+1} ┤                          └── loss latente vs ──┐
        ▼                                                │
  TargetEncoder (EMA, stop-grad) ─► s_{t+1} ─────────────┘   + VICReg (anti-collapse)
```

Loss totale (dans `train.py`) : `L_pred (latente) + λ·VICReg + reward (MSE) + done (BCE)`, avec mise à jour EMA du target encoder à chaque step.

### Prédiction multi-pas (important pour le planning)

Le prédicteur s'applique de façon **autorégressive** dans le latent (`ŝ_{t+2} = Pred(ŝ_{t+1}, a)`), sans jamais re-décoder. Pour que les rollouts longs restent fiables, deux mécanismes sont prévus :
- **Entraînement multi-step** : déplier le prédicteur sur K pas (def. K=5) et superviser chaque `ŝ` contre sa cible encodée → réduit la dérive (compounding error). Inspiré de Dreamer / TD-MPC.
- **MPC en boucle fermée (receding horizon)** : planifier H pas, n'exécuter que le 1er, ré-encoder l'observation réelle, re-planifier. Corrige la dérive à chaque pas.

Limite intrinsèque : au-delà d'un **repas**, la position de la nouvelle pomme est aléatoire → futur imprévisible. On planifie donc vers la pomme courante (H court : 3-5 au départ).

### ⚠️ Règle d'or : le collapse

Un JEPA peut « tricher » en faisant sortir une **constante** par l'encodeur (loss latente nulle, modèle inutile). Toute implémentation de l'encodeur/loss DOIT conserver les 3 parades : **stop-gradient** sur le target encoder, **EMA** du target encoder, **VICReg** (terme de variance). Monitorer en continu la **variance / le rang des embeddings** dans `validation/metrics.py` - une chute vers 0 = collapse.

## Résultats & apprentissages clés (pipeline validé)

Pipeline entraîné de bout en bout. Modèle = 1.4M params, ~1 min sur GPU. **MPC mean ≈ 7.8, max 17, survie ~132 pas** (vs random 0.07, plafond heuristique greedy 21.3) - le world model joue **sans aucune policy apprise**. `emb_std ≈ 1.0`, effective_rank ≈ 83/128 (jamais de collapse).

Quatre pièges résolus - **ne pas régresser dessus** :

1. **Sélection du modèle sur la loss des têtes (reward+done), PAS la loss totale.** La MSE de prédiction latente n'est **pas invariante à l'échelle** : elle grossit quand VICReg étend le latent (emb_std 0→1), donc sélectionner dessus récompense l'état dégénéré du début. Les cibles reward/done sont à échelle fixe → fiables.
2. **Têtes reward/done supervisées sur le VRAI latent encodé** (`model.encode(obs_k)`), pas sur le latent prédit `ŝ` du rollout (qui a dérivé). Sinon la mort (en fin de fenêtre, là où `ŝ` dérive le plus) est mal apprise. Ce fix : done recall 0.05 → 0.77.
3. **Reward shaping dense obligatoire** (`R_SHAPING=0.1` × réduction de distance Manhattan à la pomme, dans `snake_env.py`). Sans lui, le planner n'a aucun signal vers une pomme hors horizon → il erre. Avec : MPC mean 1.1 → 7.8.
4. **`done` déséquilibré** (~2% de morts) : BCE avec `pos_weight` (cap 15) + métriques **recall/precision/F1**, jamais l'accuracy (trompeuse à 98%).

Horizon MPC optimal = **5** (balayé) ; H≥6 régresse à cause du drift.

## Structure du code

Tous les modules sont **implémentés et vérifiés**. Lancer dans cet ordre : `make_dataset` → `train` → `metrics` (qui appelle le `planner`).

| Module | Rôle | État |
|---|---|---|
| `src/envs/snake_env.py` | Env headless type Gym (`reset()` / `step(a) → obs, reward, done`), obs grille `3×8×16`, **reward shaping** vers la pomme. | ✅ |
| `src/data/make_dataset.py` | Trajectoires `(o,a,r,done,o')` via politique mixte ε-greedy → `data/2-processed/snake_transitions.npz`. | ✅ |
| `src/models/model.py` | `Encoder` (CNN), `Predictor` (MLP résiduel), `TargetEncoder` (EMA), `reward_head`, `done_head`, `ema_update`. | ✅ |
| `src/models/train.py` | Loss JEPA + VICReg + EMA + multi-step, têtes sur vrai latent, sélection sur head loss, MLflow. | ✅ |
| `src/models/planner.py` | Contrôleur **MPC latent** (recherche exhaustive `4^H` vectorisée, survival-weighted return). | ✅ |
| `src/validation/metrics.py` | Collapse monitor (std + rang), drift multi-pas, qualité reward/done, éval agent vs baselines. | ✅ |
| `src/config.py` | Config Pydantic `BaseSettings` : tous les hyperparamètres + paths + MLflow. | ✅ |

## `snake.py` - état actuel et pièges

C'est le **moteur de jeu existant**, déjà utilisé pour entraîner d'autres IA (NEAT). À connaître avant de le refactorer :

- **Pas exécutable seul** : aucun bloc `if __name__ == "__main__"`. `game_loop(...)` exige des arguments NEAT (`net, genome, i, Neat`) et n'est appelé nulle part. Le fichier est un **module à importer**, pas un script.
- **Grille** : `width=800`, `height=400`, cases de `50` → **16 colonnes × 8 lignes** (128 cases).
- **Actions** : `0=UP`, `1=RIGHT`, `2=DOWN`, `3=LEFT`. Contrainte **demi-tour interdit** (UP↔DOWN, LEFT↔RIGHT ignorés).
- **Stochasticité** : la pomme réapparaît à une position **aléatoire** ([`generated_food`](snake.py)) - c'est la seule source d'incertitude, et un argument fort pour prédire dans le latent plutôt qu'en pixels.
- **Observation actuelle** : vecteur de **16 features hand-crafted** (8 distances bords/corps + 8 distances pomme). Conservé pour mémoire / baseline, **mais le world model utilise la grille**.
- **Flags globaux** : `show` (rendu pygame), `player` (contrôle clavier), `info` (debug), `vitesse`, `stop_iteration=500`. Le refactor headless doit pouvoir tourner avec `show=False` et sans pygame initialisé.

## Commandes

```powershell
# Activer l'environnement (IA / PyTorch + CUDA)
& c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1

# Tests
pytest                                   # toute la suite
pytest tests/test_model.py::test_placeholder   # un seul test

# Entraînement / data (une fois les stubs implémentés)
python -m src.models.train
python -m src.data.make_dataset

# Suivi des expériences
mlflow ui                                # dashboard MLflow sur http://localhost:5000
```

- **Device** : `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Le modèle est minuscule (grille 8×16, petit CNN) → tourne très vite, CPU possible.
- **Artefacts** : modèles → `outputs/models/`, logs → `outputs/logs/`, résultats/plots → `outputs/results/`. Données : `data/1-raw` → `data/2-processed` → `data/3-external`. (Tout ce contenu est git-ignoré sauf les `.gitkeep`.)
