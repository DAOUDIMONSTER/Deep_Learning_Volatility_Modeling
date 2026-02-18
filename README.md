# Modélisation de la Volatilité par Deep Learning

## Vue d’ensemble

Ce dépôt contient un notebook Jupyter (`PRRE.ipynb`) implémentant une approche de deep learning pour la modélisation de la volatilité implicite (IV) dans les modèles stochastiques de Black-Scholes (BS) et Heston, telle que décrite dans l’article *Deep Learning Volatility: A unified approach for implied volatility modeling under different stochastic processes* de H. Buehler et al. (2019).

Le notebook entraîne des réseaux de neurones artificiels (ANN) pour prédire la volatilité implicite à partir de données de marché d’options, en utilisant des caractéristiques comme le moneyness ($S/K$), le temps à maturité ($\tau$) et les paramètres du processus stochastique. Il reproduit la méthodologie du papier (génération de données, entraînement et évaluation) et l’enrichit avec des visualisations supplémentaires.

Ce projet a été réalisé dans le cadre du module Deep Learning & Financial Modeling du semestre 4 à l’ENSIIE (École Nationale Supérieure d’Informatique pour l’Industrie et l’Entreprise) par DAOUDI ALI && YASSINE BENTAYFORE. L’implémentation repose sur TensorFlow (avec accélération GPU), Python (pandas, seaborn, etc.) et Docker pour garantir un environnement reproductible.

---

## Lien avec l’article

Le notebook suit de près la méthodologie présentée dans le papier :

- **Objectif** : Entraîner des ANN pour prédire la volatilité implicite ($\sigma^*$) pour les modèles BS et Heston, en utilisant des caractéristiques comme le moneyness, le temps à maturité, et la valeur temps logarithmique ($\ln((V - \text{intrinsic})/K)$).

- **Génération des données** :
  - **BS** : Génère 1 million d'échantillons en utilisant la formule de Black-Scholes et l'échantillonnage par hypercube latin (LHS), avec les entrées ($S/K$, $\tau$, $r$, $\ln((V - \text{intrinsic})/K)$) et la sortie IV ($\sigma^* = \sigma$) (Section 3.1.1).
  - **Heston** : Génère 100 000 échantillons en utilisant la méthode COS pour le pricing et la méthode de Brent pour l'inversion de la VI, avec les entrées ($S/K$, $\tau$, $r$, $\rho$, $\kappa$, $\theta$, $\sigma_v$, $v_0$, $\ln((V - \text{intrinsic})/K)$) (Section 3.1.2).

- **Architecture du réseau** : Implémente un réseau *feedforward* à 4 couches (400 neurones par couche, activation ReLU, optimiseur Adam, fonction de perte MSE), comme spécifié dans la Section 3.2.1.

- **Entraînement** : Utilise un taux d'apprentissage décroissant par paliers (0.001 pour 100 époques, 0.0001 pour 50, 0.00001 pour 50) et un prétraitement avec `StandardScaler` (Section 3.2.3).

- **Évaluation** : Reproduit les graphiques du papier (Figures 5–7, 9b, 11) et atteint des MSE comparables (BS ~1.55e-8, Heston ~1.14e-6) (Section 4).

- **Extensions** : Ajoute de nouvelles visualisations (prix de l'option vs. strike/moneyness, sourire de volatilité Heston) non explicitement montrées dans le papier.

- **Omission** : N'implémente pas le pipeline optionnel Heston en deux étapes (Figure 10), qui entraîne un second ANN pour mapper la VI de Black-Scholes à la VI de Heston.

---

## Contenu du Dépôt

- `PRRE.ipynb` : Le notebook Jupyter principal contenant l'implémentation (19 cellules de code avec des explications en markdown).
- `Dockerfile` : Définit l'image Docker `grok-prre-gpu` pour exécuter le notebook avec le support GPU.
- `README.md` : Ce fichier, fournissant la documentation du projet.

**Sorties** (générées lors de l'exécution du notebook) :
- **Données** : `bs_data.npz`, `heston_data.npz`, `bs_wide_test.npz`, `heston_wide_test.npz` (jeux de données d'entraînement/test).
- **Graphiques** :
  - Figures du papier : `fig5_bs_pred_vs_actual.png`, `fig6_bs_loss_curves.png`, `fig7_bs_iv_vs_moneyness.png`, `fig11_bs_error_dist.png`, etc.
  - Nouveaux graphiques : `bs_price_vs_strike.png`, `bs_price_vs_moneyness.png`, `heston_price_vs_strike.png`, `heston_price_vs_moneyness.png`, `heston_vol_smile.png`.

---

## Prérequis

Pour exécuter le notebook, vous avez besoin de :
- **Matériel** : Un ordinateur avec un GPU NVIDIA (pour l'accélération GPU de TensorFlow).
- **Logiciels** :
  - Docker (pour l'environnement conteneurisé).
  - Support NVIDIA Docker (pour l'accès au GPU).
  - Git (pour cloner le dépôt).
- **Système d'exploitation** : Testé sur WSL2 (Windows Subsystem for Linux 2) avec Ubuntu, mais devrait fonctionner sur Linux ou macOS avec Docker.

---
## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/DAOUDIMONSTER/Deep_learning_volatility_Modeling.git
cd Deep_learning_volatility_Modeling 
```


### 2. Build the Docker Image
The notebook runs in a custom Docker image (`grok-prre-gpu`) based on `tensorflow:2.15.0-gpu-jupyter`.

```bash
docker build -t grok-prre-gpu .
```

**Dockerfile Contents**:
```dockerfile
FROM tensorflow/tensorflow:2.15.0-gpu-jupyter
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    pandas numpy scipy matplotlib scikit-learn seaborn tqdm
ENV JUPYTER_NOTEBOOK_DIR=/tf/notebooks
RUN rm -f /app/*.ipynb
WORKDIR /tf/notebooks
```

**Note**: If the build fails due to hash mismatches, try:
- Adding `--index-url https://pypi.python.org/simple/` to the `pip install` command.
- Installing packages one at a time (e.g., `RUN pip install pandas && pip install numpy ...`).

### 3. Run the Docker Container
Mount your local project directory to `/tf/notebooks` in the container and start Jupyter Notebook.

```bash
docker run --gpus all -it -p 8888:8888 -v $(pwd):/tf/notebooks grok-prre-gpu
```

- `--gpus all`: Enables GPU access.
- `-p 8888:8888`: Maps Jupyter’s port to your host.
- `-v $(pwd):/tf/notebooks`: Mounts the project directory.

### 4. Access Jupyter Notebook
- Copy the Jupyter URL from the terminal (e.g., `http://127.0.0.1:8888/?token=...`).
- Open it in a web browser.
- Open `PRRE.ipynb`.

## Running the Notebook

1. **Verify Environment**:
   - Run Cell 1 to install dependencies and check GPU availability.
   - Expected output:
     ```
     Pandas: 2.2.3
     NumPy: 1.26.2
     TensorFlow: 2.15.0
     GPUs Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
     ```

2. **Execute All Cells**:
   - Run Cells 1–19 to:
     - Generate BS (1M samples) and Heston (100K samples) datasets.
     - Train ANNs for BS and Heston IV prediction.
     - Evaluate performance with MSEs (BS ~1.55e-8, Heston ~1.14e-6).
     - Produce plots (Figures 5–7, 9b, 11, plus new price/volatility plots).
   - Outputs are saved to the mounted directory (`./bs_data.npz`, `./bs_price_vs_strike.png`, etc.).

3. **Monitor GPU Usage**:
   - In another terminal:
     ```bash
     nvidia-smi
     ```
   - Expect Python/TensorFlow using ~2–4 GB during training (Cells 14–15).

**Runtime**:
- Data generation: ~10–15 minutes (Heston is slower due to COS method).
- ANN training: ~30–40 minutes (BS ~20 min, Heston ~15 min).
- Plotting: ~2 minutes.

## Notebook Structure

The notebook is organized into 19 code cells with markdown explanations:

1. **Environment Setup**: Installs dependencies and verifies GPU.
2. **Library Imports**: Imports `pandas`, `tensorflow`, etc.
3–4. **Pricing Functions**: BS call pricing and Heston COS method with IV inversion.
5–6. **Data Generation**: LHS for BS (1M samples) and Heston (100K samples).
7. **Data Exploration**: Plots feature distributions and correlations.
8. **Data Preprocessing**: Standardizes features with `StandardScaler`.
9. **ANN Model Definition**: 4-layer ANN (400 neurons, ReLU).
10. **ANN Training**: Trains with piecewise learning rate (200 epochs).
11. **Evaluation and Visualization**: Generates Figures 5–7, 9b (predicted IV, loss curves, IV vs. moneyness, BS price vs. volatility).
12–13. **Wide/Narrow Test Sets**: Evaluates generalization for Figure 11.
14. **Main Execution: BS**: Runs BS pipeline (MSE ~1.55e-8).
15. **Main Execution: Heston**: Runs Heston pipeline (MSE ~1.14e-6).
16–19. **Additional Visualizations**: Plots option price vs. strike/moneyness and Heston volatility smile.

**Markdown Cells**: Added before each section to explain purpose and paper relation (see notebook for details).

## Results

- **Performance**:
  - BS MSE: ~1.55e-8, matching the paper’s accuracy.
  - Heston MSE: ~1.14e-6, consistent with the paper.
- **Plots**:
  - Replicates paper’s Figures 5–7 (predicted vs. actual IV, loss curves, IV vs. moneyness), 9b (BS volatility vs. price), and 11 (error distributions).
  - New plots: BS/Heston price vs. strike/moneyness, Heston volatility smile (U-shaped IV curve, ~0.2–0.4).
- **Validation**:
  - BS prices: Convex, decreasing curve (~20 at K=80, ~2 at K=120).
  - Heston prices: Similar to BS with stochastic volatility effects.
  - Volatility smile: Captures Heston’s characteristic U-shape.

## Troubleshooting

- **Docker Build Fails**:
  - Check for hash mismatch errors in `pip install`.
  - Modify Dockerfile to use a specific PyPI mirror:
    ```dockerfile
    RUN pip install --no-cache-dir --index-url https://pypi.python.org/simple/ pandas ...
    ```
  - Install packages individually to isolate issues.
- **Imports Fail**:
  - Verify kernel uses container’s Python (`/usr/local/bin/python3`):
    ```bash
    docker run --gpus all -it grok-prre-gpu /bin/bash
    jupyter kernelspec list
    ```
  - Reinstall kernel:
    ```bash
    python3 -m pip install ipykernel
    python3 -m ipykernel install --user --name python3
    ```
- **GPU Not Used**:
  - Confirm GPU detection:
    ```bash
    docker run --gpus all -it grok-prre-gpu python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```
  - Ensure NVIDIA drivers and Docker GPU support are installed.
- **Plot Errors**:
  - Ensure Cells 14–15 (training) ran successfully, creating `model_bs` and `model_heston`.
  - Check for missing dependencies (`matplotlib`, `seaborn`).

## Future Improvements

- **Heston Two-Step Pipeline**: Implement the paper’s optional second ANN to map BS IV to Heston IV (Figure 10).
- **Additional Models**: Extend to other stochastic processes (e.g., SABR, Merton).
- **Sensitivity Analysis**: Plot IV sensitivity to Heston parameters (e.g., ρ, κ).
- **Interactive Plots**: Use Plotly for dynamic visualizations.
- **Performance Optimization**: Reduce Heston data generation time with GPU-accelerated COS method.

## Acknowledgments

- Based on *Deep Learning Volatility* by H. Buehler et al. (2019).
- Developed for ENSIIE S4 coursework.
- Uses TensorFlow, Docker, and Python libraries for implementation.

## Contact

For questions or issues, please open a GitHub issue or contact [yassine.bentayfor2015@gmail.com].
