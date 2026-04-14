
import random
from dataclasses import dataclass

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


@dataclass
class RoundResult:
    round_no: int
    chosen_arm: int
    reward: int
    cumulative_reward: int
    estimated_value: float
    pulls_of_arm: int


KEY_N_ARMS = "bandit_n_arms"
KEY_TRUE_PROBS = "bandit_true_probs"
KEY_HISTORY = "bandit_history"
KEY_COUNTS = "bandit_counts"
KEY_ESTIMATES = "bandit_estimates"
KEY_ROUND_NO = "bandit_round_no"
KEY_TOTAL_REWARD = "bandit_total_reward"
KEY_REVEAL = "bandit_reveal_probs"


def random_probs(n_arms: int) -> list[float]:
    return [round(random.uniform(0.1, 0.9), 2) for _ in range(n_arms)]


def sync_state_lengths() -> None:
    n_arms = int(st.session_state.get(KEY_N_ARMS, 5))

    true_probs = st.session_state.get(KEY_TRUE_PROBS)
    if not isinstance(true_probs, list) or len(true_probs) != n_arms:
        st.session_state[KEY_TRUE_PROBS] = random_probs(n_arms)

    counts = st.session_state.get(KEY_COUNTS)
    if not isinstance(counts, list):
        st.session_state[KEY_COUNTS] = [0] * n_arms
    elif len(counts) != n_arms:
        st.session_state[KEY_COUNTS] = counts[:n_arms] + [0] * max(0, n_arms - len(counts))

    estimates = st.session_state.get(KEY_ESTIMATES)
    if not isinstance(estimates, list):
        st.session_state[KEY_ESTIMATES] = [0.0] * n_arms
    elif len(estimates) != n_arms:
        st.session_state[KEY_ESTIMATES] = estimates[:n_arms] + [0.0] * max(0, n_arms - len(estimates))

    history = st.session_state.get(KEY_HISTORY)
    if not isinstance(history, list):
        st.session_state[KEY_HISTORY] = []

    if not isinstance(st.session_state.get(KEY_ROUND_NO), int):
        st.session_state[KEY_ROUND_NO] = 0

    if not isinstance(st.session_state.get(KEY_TOTAL_REWARD), int):
        st.session_state[KEY_TOTAL_REWARD] = 0

    if not isinstance(st.session_state.get(KEY_REVEAL), bool):
        st.session_state[KEY_REVEAL] = False


def init_state() -> None:
    if KEY_N_ARMS not in st.session_state or not isinstance(st.session_state.get(KEY_N_ARMS), int):
        st.session_state[KEY_N_ARMS] = 5
    sync_state_lengths()


def reset_progress(keep_bandits: bool = True) -> None:
    n_arms = int(st.session_state.get(KEY_N_ARMS, 5))
    if not keep_bandits:
        st.session_state[KEY_TRUE_PROBS] = random_probs(n_arms)

    st.session_state[KEY_HISTORY] = []
    st.session_state[KEY_COUNTS] = [0] * n_arms
    st.session_state[KEY_ESTIMATES] = [0.0] * n_arms
    st.session_state[KEY_ROUND_NO] = 0
    st.session_state[KEY_TOTAL_REWARD] = 0
    sync_state_lengths()


def regenerate_bandits(n_arms: int) -> None:
    st.session_state[KEY_N_ARMS] = int(n_arms)
    st.session_state[KEY_TRUE_PROBS] = random_probs(int(n_arms))
    reset_progress(keep_bandits=True)


def pull_arm(arm_index: int) -> int:
    sync_state_lengths()
    p = st.session_state[KEY_TRUE_PROBS][arm_index]
    return 1 if random.random() < p else 0


def update_estimate(arm_index: int, reward: int) -> None:
    sync_state_lengths()
    counts = st.session_state[KEY_COUNTS]
    estimates = st.session_state[KEY_ESTIMATES]

    counts[arm_index] += 1
    n = counts[arm_index]
    current_value = estimates[arm_index]
    estimates[arm_index] = current_value + (reward - current_value) / n

    st.session_state[KEY_COUNTS] = counts
    st.session_state[KEY_ESTIMATES] = estimates


def play_round(arm_index: int) -> None:
    reward = pull_arm(arm_index)
    update_estimate(arm_index, reward)

    st.session_state[KEY_ROUND_NO] += 1
    st.session_state[KEY_TOTAL_REWARD] += reward

    history = st.session_state[KEY_HISTORY]
    history.append(
        RoundResult(
            round_no=st.session_state[KEY_ROUND_NO],
            chosen_arm=arm_index + 1,
            reward=reward,
            cumulative_reward=st.session_state[KEY_TOTAL_REWARD],
            estimated_value=st.session_state[KEY_ESTIMATES][arm_index],
            pulls_of_arm=st.session_state[KEY_COUNTS][arm_index],
        )
    )
    st.session_state[KEY_HISTORY] = history


def select_arm_epsilon_greedy(epsilon: float) -> int:
    sync_state_lengths()
    counts = st.session_state[KEY_COUNTS]
    estimates = st.session_state[KEY_ESTIMATES]
    n_arms = st.session_state[KEY_N_ARMS]

    for i, c in enumerate(counts):
        if c == 0:
            return i

    if random.random() < epsilon:
        return random.randint(0, n_arms - 1)

    best_value = max(estimates)
    best_arms = [i for i, v in enumerate(estimates) if v == best_value]
    return random.choice(best_arms)


def autoplay(n_rounds: int, epsilon: float) -> None:
    for _ in range(n_rounds):
        arm_index = select_arm_epsilon_greedy(epsilon)
        play_round(arm_index)


def history_df() -> pd.DataFrame:
    sync_state_lengths()
    history = st.session_state[KEY_HISTORY]
    if not history:
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
            for r in history
        ]
    )


def arm_summary_df() -> pd.DataFrame:
    sync_state_lengths()
    reveal = st.session_state[KEY_REVEAL]
    true_probs = st.session_state[KEY_TRUE_PROBS]
    counts = st.session_state[KEY_COUNTS]
    estimates = st.session_state[KEY_ESTIMATES]
    best_prob = max(true_probs)

    rows = []
    for i in range(st.session_state[KEY_N_ARMS]):
        rows.append(
            {
                "Arm": f"Arm {i + 1}",
                "Pulls": counts[i],
                "Estimated Reward Rate": round(estimates[i], 3),
                "True Reward Probability": true_probs[i] if reveal else "Hidden",
                "Best Arm": "Yes" if reveal and true_probs[i] == best_prob else "",
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


def plot_arm_estimates():
    sync_state_lengths()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(1, st.session_state[KEY_N_ARMS] + 1))
    ax.bar(x, st.session_state[KEY_ESTIMATES])
    ax.set_xticks(x)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Estimated Reward Rate")
    ax.set_title("Current Estimated Value by Arm")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def main() -> None:
    st.set_page_config(page_title="One-Armed Bandit Demo", page_icon="🎰", layout="wide")
    init_state()

    st.title("🎰 One-Armed Bandit Demo")
    st.caption(
        "A simple reinforcement learning classroom demo showing exploration, "
        "exploitation, hidden reward probabilities, and cumulative performance."
    )

    st.sidebar.header("Controls")

    n_arms_input = st.sidebar.slider(
        "Number of arms",
        min_value=2,
        max_value=10,
        value=int(st.session_state[KEY_N_ARMS]),
    )
    if n_arms_input != st.session_state[KEY_N_ARMS]:
        regenerate_bandits(int(n_arms_input))
        st.rerun()

    st.session_state[KEY_REVEAL] = st.sidebar.checkbox(
        "Reveal true reward probabilities",
        value=bool(st.session_state.get(KEY_REVEAL, False)),
    )

    epsilon = st.sidebar.slider(
        "Autoplay epsilon (exploration rate)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
    )
    autoplay_rounds = st.sidebar.slider(
        "Autoplay rounds",
        min_value=1,
        max_value=500,
        value=50,
        step=1,
    )

    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        if st.button("Reset rounds"):
            reset_progress(keep_bandits=True)
            st.rerun()
    with col_sb2:
        if st.button("New bandits"):
            regenerate_bandits(st.session_state[KEY_N_ARMS])
            st.rerun()

    if st.sidebar.button("Run autoplay"):
        autoplay(autoplay_rounds, epsilon)
        st.rerun()

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Rounds played", st.session_state[KEY_ROUND_NO])
    with metric_cols[1]:
        st.metric("Total reward", st.session_state[KEY_TOTAL_REWARD])
    with metric_cols[2]:
        avg_reward = (
            round(st.session_state[KEY_TOTAL_REWARD] / st.session_state[KEY_ROUND_NO], 3)
            if st.session_state[KEY_ROUND_NO] > 0
            else 0.0
        )
        st.metric("Average reward", avg_reward)
    with metric_cols[3]:
        best_arm = st.session_state[KEY_TRUE_PROBS].index(max(st.session_state[KEY_TRUE_PROBS])) + 1
        st.metric("Best arm", f"Arm {best_arm}" if st.session_state[KEY_REVEAL] else "Hidden")

    st.divider()

    st.subheader("Manual Play")
    st.write(
        "Each button pull gives either reward **1** or **0**. "
        "The true probabilities are hidden unless you choose to reveal them."
    )

    button_cols = st.columns(st.session_state[KEY_N_ARMS])
    for i in range(st.session_state[KEY_N_ARMS]):
        if button_cols[i].button(f"Pull Arm {i + 1}", use_container_width=True):
            play_round(i)
            st.rerun()

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
        st.code("pip install streamlit pandas matplotlib\nstreamlit run one_armed_bandit_demo_fixed.py", language="bash")


if __name__ == "__main__":
    main()
