# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import time
from mini_pacman_rl import (
    MiniPacmanEnv,
    QLearningAgent,
    train_agent,
    play_episode_images,   
)

st.set_page_config(page_title="Mini-Pacman RL", layout="wide")

st.title("🟡 Mini-Pacman avec Q-Learning")
st.write("Interface web Streamlit : entraînement, résultats et démo de la politique apprise.")

if "trained" not in st.session_state:
    st.session_state.trained = False
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "coins" not in st.session_state:
    st.session_state.coins = []
if "params" not in st.session_state:
    st.session_state.params = {}
if "agent" not in st.session_state:
    st.session_state.agent = None
if "env_conf" not in st.session_state:
    st.session_state.env_conf = {}

tab_train, tab_results, tab_demo = st.tabs(["⚙ Training", "📊 Résultats", "🎮 Démo"])


with tab_train:
    st.header("Configuration de l'entraînement")

    col1, col2 = st.columns(2)

    with col1:
        width = st.number_input("Largeur de la grille", min_value=7, max_value=31, value=15, step=2)
        height = st.number_input("Hauteur de la grille", min_value=7, max_value=31, value=11, step=2)
        coin_prob = st.slider("Proba de pièce par case", 0.0, 0.5, 0.15, 0.01)
        max_steps = st.number_input("Nombre max de pas par épisode", min_value=10, max_value=10000, value=100, step=10)
        regenerate_maze = st.checkbox("Nouveau labyrinthe à chaque épisode ?", value=False)

    with col2:
        n_episodes = st.number_input("Nombre d'épisodes", min_value=10, max_value=10000, value=500, step=10)
        alpha = st.slider("Learning rate (alpha)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.slider("Facteur de discount (gamma)", 0.0, 0.999, 0.99, 0.001)
        epsilon = st.slider("Epsilon initial (exploration)", 0.0, 1.0, 1.0, 0.05)
        epsilon_decay = st.slider("Décroissance epsilon", 0.90, 0.999, 0.995, 0.001)
        epsilon_min = st.slider("Epsilon min", 0.0, 0.5, 0.05, 0.01)

    if st.button("🚀 Lancer l'entraînement"):
        st.write("Entraînement en cours...")

        env = MiniPacmanEnv(width=width, height=height, max_steps=max_steps, coin_prob=coin_prob)
        agent = QLearningAgent(
            n_actions=4,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay
        )

        rewards, coins = train_agent(
            env,
            agent,
            n_episodes=int(n_episodes),
            max_steps=int(max_steps),
            regenerate_maze=regenerate_maze
        )

        st.session_state.trained = True
        st.session_state.rewards = rewards
        st.session_state.coins = coins
        st.session_state.agent = agent
        st.session_state.env_conf = {
            "width": width,
            "height": height,
            "max_steps": max_steps,
            "coin_prob": coin_prob
        }
        st.session_state.params = {
            "n_episodes": n_episodes,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "regenerate_maze": regenerate_maze
        }
        st.session_state.env = env

        st.success("✅ Entraînement terminé ! Va voir l'onglet Résultats et Démo.")


with tab_results:
    st.header("Résultats de l'entraînement")

    if not st.session_state.trained:
        st.info("Lance d'abord un entraînement dans l'onglet Training.")
    else:
        rewards = np.array(st.session_state.rewards)
        coins = np.array(st.session_state.coins)
        episodes = np.arange(1, len(rewards) + 1)

        window = max(1, len(rewards) // 20)

        def moving_average(x, w):
            if w <= 1:
                return x
            return np.convolve(x, np.ones(w) / w, mode="valid")

        avg_rewards = moving_average(rewards, window)
        avg_coins = moving_average(coins, window)

        df_rewards = pd.DataFrame({
            "Episode": episodes[:len(avg_rewards)],
            "Récompense moyenne": avg_rewards
        })
        df_coins = pd.DataFrame({
            "Episode": episodes[:len(avg_coins)],
            "Pièces ramassées": avg_coins
        })

        st.subheader("Récompense par épisode (moyenne glissante)")
        st.line_chart(df_rewards, x="Episode", y="Récompense moyenne")

        st.subheader("Pièces ramassées par épisode (moyenne glissante)")
        st.line_chart(df_coins, x="Episode", y="Pièces ramassées")



with tab_demo:
    st.header("Démonstration de la politique apprise (animation graphique)")

    if not st.session_state.trained or st.session_state.agent is None:
        st.info("Pas encore d'agent entraîné. Lance un training d'abord.")
    else:
        env = st.session_state.env
        agent = st.session_state.agent



        max_steps_demo = st.number_input("Nombre max de pas pour la démo", 10, 500, 80, 10)
        speed = st.slider("Vitesse de l'animation (sec entre frames)", 0.02, 0.5, 0.12, 0.01)
        cell_size = st.slider("Taille d'une case (pixels)", 16, 64, 32, 4)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶ Générer une partie graphique"):
                frames = play_episode_images(env, agent,
                                             max_steps=int(max_steps_demo),
                                             cell_size=int(cell_size))
                st.session_state.demo_frames_img = frames
                st.success(f"Partie générée avec {len(frames)} frames.")

        with col2:
            if st.button("🎬 Lancer l'animation"):
                if "demo_frames_img" not in st.session_state:
                    st.warning("Génère d'abord une partie avec le bouton de gauche.")
                else:
                    frames = st.session_state.demo_frames_img
                    placeholder = st.empty()

                    for frame in frames:
                        placeholder.image(frame)
                        time.sleep(speed)

        if "demo_frames_img" in st.session_state:
            st.markdown("---")
            st.write("Navigation manuelle image par image :")
            frames = st.session_state.demo_frames_img
            idx = st.slider("Étape", 0, len(frames) - 1, 0)
            st.image(frames[idx])
