import random
from dataclasses import dataclass
from typing import List

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="One-Armed Bandit Demo", page_icon="🎰", layout="wide")

st.title("🎰 One-Armed Bandit Demo")
st.caption(
    "A simple reinforcement learning classroom demo showing exploration, exploitation, "
    "hidden reward probabilities, and cumulative performance."
)


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class RoundResult:
    round_no: int
    chosen_arm: int
    reward: int
    cumulative_reward: int
    estimated_value: float
    pulls_of_arm: int


# ----------------------------
# Helpers
# ----------------------------
def init_state() -> None:
    if "n_arms" not in st.session_state:
        st.session_state.n_arms = 5

    if "true_probs" not in st.session_state:
        st.session_state.true_probs = [round(random.uniform(0.1, 0.9), 2) for _ in range(st.session_state.n_arms)]

    if "history" not in st.session_state:
        st.session_state.history = []

    if "counts" not in st.session_state:
        st.session_state.counts = [0] * st.session_state.n_arms

    if "values" not in st.session_state:
        st.session_state.values = [0.0] * st.session_state.n_arms

    if "round_no" not in st.session_state:
        st.session_state.round_no = 0

    if "total_reward" not in st.session_state:
        st.session_state.total_reward = 0

    sync_state_lengths()


def sync_state_lengths() -> None:
    n_arms = st.session_state.n_arms

    if "true_probs" not in st.session_state or len(st.session_state.true_probs) != n_arms:
        st.session_state.true_probs = [round(random.uniform(0.1,(n_arms: int) -> None:
    st.session_state.n_arms = n_arms
    st.session_state.true_probs = [round(random.uniform(0.1, 0.9), 2) for _ in range(n_arms)]
    reset_progress(keep_bandits=True)


def reset_progress(keep_bandits: bool = True) -> None:
    n_arms = st.session_state.get("n_arms", 5)

    if not keep_bandits or "true_probs" not in st.session_state:
        st.session_state.true_probs = [round(random.uniform(0.1, 0.9), 2) for _ in range(n_arms)]

    st.session_state.history = []
    st.session_state.counts = [0] * n_arms
    st.session_state.values = [0.0] * n_arms
    st.session_state.round_no = 0
    st.session_state.total_reward = 0


def pull_arm(arm_index: int) -> int:
    p = st.session_state.true_probs[arm_index]
    return 1 if random.random() < p else 0


def update_estimate(arm_index: int, reward: int) -> None:
    st.session_state.counts[arm_index] += 1
    n = st.session_state.counts[arm_index]
    current_value = st.session_state.values[arm_index]
    new_value = current_value + (reward - current_value) / n
    st.session_state.values[arm_index] = new_value


def play_round(arm_index: int) -> None:
    reward = pull_arm(arm_index)
    update_estimate(arm_index, reward)
    st.session_state.round_no += 1
    st.session_state.total_reward += reward

    result = RoundResult(
        round_no=st.session_state.round_no,
        chosen_arm=arm_index + 1,
        reward=reward,
        cumulative_reward=st.session_state.total_reward,
        estimated_value=st.session_state.values[arm_index],
        pulls_of_arm=st.session_state.counts[arm_index],
    )
    st.session_state.history.append(result)


def select_arm_epsilon_greedy(epsilon: float) -> int:
    # Ensure each arm gets tried at least once for a better classroom demo.
    for i, c in enumerate(st.session_state.counts):
        if c == 0:
            return i

    if random.random() < epsilon:
        return random.randint(0, st.session_state.n_arms - 1)

    best_value = max(st.session_state.values)
    best_arms = [i for i, v in enumerate(st.session_state.values) if v == best_value]
    return random.choice(best_arms)


def autoplay(n_rounds: int, epsilon: float) -> None:
    for _ in range(n_rounds):
        arm_index = select_arm_epsilon_greedy(epsilon)
        play_round(arm_index)


def history_df() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame(
            columns=[
                "Round",
                "Arm",
                "Reward",
                "Cumulative Reward",
                "Estimated Value of Chosen Arm",
                "Pulls of Chosen Arm",
            ]
        )

    return pd.DataFrame(
        [
            {
                "Round": r.round_no,
                "Arm": r.chosen_arm,
                "Reward": r.reward,
                "Cumulative Reward": r.cumulative_reward,
                "Estimated Value of Chosen Arm": round(r.estimated_value, 3),
                "Pulls of Chosen Arm": r.pulls_of_arm,
            }
            for r in st.session_state.history
        ]
    )


def arm_summary_df() -> pd.DataFrame:
    rows = []
    reveal = st.session_state.get("reveal_probs", False)
    best_prob = max(st.session_state.true_probs)

    for i in range(st.session_state.n_arms):
        rows.append(
            {
                "Arm": f"Arm {i + 1}",
                "Pulls": st.session_state.counts[i],
                "Estimated Reward Rate": round(st.session_state.values[i], 3),
                "True Reward Probability": st.session_state.true_probs[i] if reveal else "Hidden",
                "Best Arm": "Yes" if st.session_state.true_probs[i] == best_prob and reveal else "",
            }
        )
    return pd.DataFrame(rows)


def plot_cumulative_reward(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty:
        ax.plot(df["Round"], df["Cumulative Reward"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Over Time")
    ax.grid(True, alpha=0.3)
    return fig


def plot_arm_estimates() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(1, st.session_state.n_arms + 1))
    ax.bar(x, st.session_state.values)
    ax.set_xticks(x)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Estimated Reward Rate")
    ax.set_title("Current Estimated Value by Arm")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


# ----------------------------
# Initialize
# ----------------------------
init_state()


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

n_arms_input = st.sidebar.slider("Number of arms", min_value=2, max_value=10, value=st.session_state.n_arms)
if n_arms_input != st.session_state.n_arms:
    regenerate_bandits(n_arms_input)

st.session_state.reveal_probs = st.sidebar.checkbox("Reveal true reward probabilities", value=False)

epsilon = st.sidebar.slider("Autoplay epsilon (exploration rate)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
autoplay_rounds = st.sidebar.slider("Autoplay rounds", min_value=1, max_value=500, value=50, step=1)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("Reset rounds"):
        reset_progress(keep_bandits=True)
        st.rerun()
with col_sb2:
    if st.button("New bandits"):
        regenerate_bandits(st.session_state.n_arms)
        st.rerun()

if st.sidebar.button("Run autoplay"):
    autoplay(autoplay_rounds, epsilon)
    st.rerun()


# ----------------------------
# Top metrics
# ----------------------------
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Rounds played", st.session_state.round_no)
with metric_cols[1]:
    st.metric("Total reward", st.session_state.total_reward)
with metric_cols[2]:
    avg_reward = (
        round(st.session_state.total_reward / st.session_state.round_no, 3)
        if st.session_state.round_no > 0
        else 0.0
    )
    st.metric("Average reward", avg_reward)
with metric_cols[3]:
    best_arm = st.session_state.true_probs.index(max(st.session_state.true_probs)) + 1
    st.metric("Best arm", f"Arm {best_arm}" if st.session_state.reveal_probs else "Hidden")

st.divider()


# ----------------------------
# Manual play section
# ----------------------------
st.subheader("Manual Play")
st.write("Each button pull gives either reward **1** or **0**. The true probabilities are hidden unless you choose to reveal them.")

button_cols = st.columns(st.session_state.n_arms)
for i in range(st.session_state.n_arms):
    label = f"Pull Arm {i + 1}"
    if button_cols[i].button(label, use_container_width=True):
        play_round(i)
        st.rerun()


# ----------------------------
# Charts and tables
# ----------------------------
df_hist = history_df()
df_summary = arm_summary_df()

chart_cols = st.columns(2)
with chart_cols[0]:
    st.pyplot(plot_cumulative_reward(df_hist))
with chart_cols[1]:
    st.pyplot(plot_arm_estimates())

st.subheader("Arm Summary")
st.dataframe(df_summary, use_container_width=True)

st.subheader("Round-by-Round History")
st.dataframe(df_hist, use_container_width=True, height=320)


# ----------------------------
# Teaching notes
# ----------------------------
with st.expander("Teaching Notes"):
    st.markdown(
        """
### What students should notice
- At the beginning, the agent knows nothing.
- Early experimentation is **exploration**.
- Repeatedly choosing the arm with the best observed average is **exploitation**.
- The estimated values improve with more pulls, but they are noisy early on.
- A small epsilon helps the algorithm keep checking whether a seemingly weaker arm might actually be better.

### Good classroom prompts
- Why not always pick the arm with the best current estimate?
- What happens if epsilon is too low?
- What happens if epsilon is too high?
- Why can a bad arm look good early on?
- How is this different from supervised learning?
        """
    )

with st.expander("How to run this app"):
    st.code("pip install streamlit pandas matplotlib\nstreamlit run app.py", language="bash")
