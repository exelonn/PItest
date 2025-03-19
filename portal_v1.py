import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
from scipy.stats import poisson
import matplotlib.font_manager as fm
from streamlit_extras.add_vertical_space import add_vertical_space


###########################################
# 1) Style Function for Matplotlib
###########################################
def apply_matplotlib_style(dark_bg=True, font_name="Arial"):
    """
    Sets up Matplotlib so:
    - All figure backgrounds are transparent (invisible).
    - Font is set to font_name.
    - Text color is white if dark_bg=True, black if dark_bg=False.
    """
    text_color = "white" if dark_bg else "black"

    mpl.rcParams["font.family"] = font_name

    # Make all backgrounds transparent
    mpl.rcParams["figure.facecolor"] = "none"
    mpl.rcParams["axes.facecolor"] = "none"
    mpl.rcParams["savefig.facecolor"] = "none"

    # Set text and axis colors
    mpl.rcParams["text.color"] = text_color
    mpl.rcParams["axes.labelcolor"] = text_color
    mpl.rcParams["xtick.color"] = text_color
    mpl.rcParams["ytick.color"] = text_color
    mpl.rcParams["axes.titlecolor"] = text_color


###########################################
# 2) Call the style function
###########################################
# Example: We assume you want dark background in Streamlit => white text in charts
apply_matplotlib_style(dark_bg=True, font_name="Calibri")


###########################################
# 3) Poisson-based no-payout function
###########################################
def probability_of_no_payout_poisson(num_events, coverage_int, event_probabilities):
    """
    Approximates the probability of no payout using Poisson.
    coverage_int = integer # of events covered
    If actual events > coverage_int, no payout occurs.
    """
    lambda_val = sum(event_probabilities)
    pmf_sum = sum(poisson.pmf(k, lambda_val) for k in range(coverage_int + 1))
    return 1 - pmf_sum


# ---------------
# Initialize
# ---------------
if "step" not in st.session_state:
    st.session_state.step = 1

st.title("Parametric Insurance")
st.subheader("Pool Risk & Return Calculator")


st.markdown("""---""")
add_vertical_space(1)
# -------------------
# 1. Collect Inputs
# -------------------
st.sidebar.header("User Inputs")

coverage_per_event = st.sidebar.number_input(
    "Coverage per Event ($)", min_value=1_000_000, value=10_000_000, step=1_000_000
)
core_probability = st.sidebar.number_input(
    "Core Probability (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.5
)
prob_range = st.sidebar.number_input(
    "Probability Range (+/- %)", min_value=0.0, max_value=5.0, value=0.0, step=0.1
)
number_of_events = st.sidebar.slider(
    "Number of Events", min_value=5, max_value=100, value=10, step=5
)
premium_multiplier = st.sidebar.number_input(
    "Premium Multiplier", min_value=1.0, max_value=10.0, value=3.0, step=0.1
)
pool_coverage = st.sidebar.number_input(
    "Pool Coverage Ratio (%)", min_value=0.0, max_value=100.0, value=100.0, step=10.0
)

# Generate probabilities
if prob_range > 0:
    random_probabilities = np.random.uniform(
        (core_probability - prob_range) / 100,
        (core_probability + prob_range) / 100,
        number_of_events,
    )
else:
    random_probabilities = np.full(number_of_events, core_probability / 100)

# Calculate premiums
premium_per_event = premium_multiplier * random_probabilities * coverage_per_event

# Build DataFrame for Step 1 display
df = pd.DataFrame(
    {
        "Event Number": np.arange(1, number_of_events + 1),
        "Assigned Probability (%)": [f"{p * 100:.2f}%" for p in random_probabilities],
        "Premium per Event ($)": [f"${p:,.2f}" for p in premium_per_event],
    }
)

# ----------------------
# Step 1: Probabilities & Premiums
# ----------------------
if st.session_state.step >= 1:
    st.subheader("Step 1: Calculated Probabilities & Premiums")

    total_premium_collected = np.sum(premium_per_event)
    coverage_int = int(number_of_events * pool_coverage / 100)

    # Dynamic text example
    if prob_range == 0:
        st.markdown(
            f"""
            **Number of events in the pool**: {number_of_events}  
            **Pool Coverage Ratio**: {pool_coverage:.0f}% → {coverage_int} events  
            **Coverage per Event**: ${coverage_per_event:,.0f}  
            **Core Probability**: {core_probability:.1f}%  
            **Probability Range**: ±{prob_range:.1f}%  

            Each event's probability is **fixed** at **{core_probability:.1f}%**.  

            $$\\text{{Premium}} = {premium_multiplier} \\times \\text{{probability}} \\times {coverage_per_event:,.0f}$$
            """
        )
    else:
        st.markdown(
            f"""
            **Number of events in the pool**: {number_of_events}  
            **Pool Coverage Ratio**: {pool_coverage:.0f}% → {coverage_int} events  
            **Coverage per Event**: ${coverage_per_event:,.0f}  
            **Core Probability**: {core_probability:.1f}%  
            **Probability Range**: ±{prob_range:.1f}%  

            Each event's probability is randomly sampled between 
            **{(core_probability - prob_range):.1f}%** and **{(core_probability + prob_range):.1f}%**.  
            
            $$\\text{{Premium}} = {premium_multiplier} \\times \\text{{probability}} \\times {coverage_per_event:,.0f}$$
            """
        )

    st.dataframe(df, hide_index=True)
    st.markdown(f"**Total Premium Collected**: ${total_premium_collected:,.2f}")
    add_vertical_space(2)
    # Next button
    if st.session_state.step == 1:
        if st.button("Next"):
            st.session_state.step = 2
            st.rerun()
    st.markdown("""---""")
    add_vertical_space(5)
# --------------------------------------------------------
# Step 2: Net Return Matrix & Risk Visualizations (Matplotlib)
# --------------------------------------------------------
if st.session_state.step >= 2:
    st.subheader("Step 2: Net Return Matrix & Risk Analysis")

    actual_events = np.arange(number_of_events + 1)
    covered_events = np.arange(1, number_of_events + 1)
    premium_collected = np.sum(premium_per_event)

    net_return_matrix = pd.DataFrame(
        index=covered_events, columns=actual_events, dtype=float
    )
    for covered in covered_events:
        for actual in actual_events:
            payout = coverage_per_event * min(actual, covered)
            net_return = (premium_collected - payout) / (coverage_per_event * covered)
            net_return_matrix.at[covered, actual] = net_return

    net_return_matrix.index.name = "Covered_Events"
    net_return_matrix.columns.name = "Actual_Events"

    st.markdown(
        """
    Net Return = (Total Premium Collected - Payout) / (Coverage per Event × Covered Events)
    """
    )

    # Prepare masked data for the heatmap (if you prefer to mask actual>covered)
    plot_data = net_return_matrix.copy()
    for c in plot_data.index:
        for a in plot_data.columns:
            if a > c:
                plot_data.at[c, a] = np.nan

    # Matplotlib Heatmap
    fig, ax = plt.subplots(figsize=(10, 7))

    cax = ax.imshow(plot_data, cmap="RdYlGn", aspect="auto", origin="upper")
    plt.colorbar(cax, ax=ax, label="Net Return")

    ax.set_title("Net Return Matrix Heatmap")
    ax.set_xlabel("Actual Events")
    ax.set_ylabel("Covered Events")

    # Set ticks
    ax.set_xticks(np.arange(len(plot_data.columns)))
    ax.set_xticklabels(plot_data.columns)
    ax.set_yticks(np.arange(len(plot_data.index)))
    ax.set_yticklabels(plot_data.index)

    # Optionally annotate cells if number_of_events is small
    if number_of_events <= 20:
        for i, covered in enumerate(plot_data.index):
            for j, actual in enumerate(plot_data.columns):
                val = plot_data.iloc[i, j]
                if not pd.isna(val):
                    ax.text(
                        j,
                        i,
                        f"{val*100:.1f}%",
                        ha="center",
                        va="center",
                        color="black",  # Correct parameter is 'color', not 'colour'
                        fontsize=12,
                    )

    st.pyplot(fig)
    add_vertical_space(2)
    # Navigation
    if st.session_state.step == 2:
        if st.button("Next"):
            st.session_state.step = 3
            st.rerun()
    if st.button("Back to Step 1"):
        st.session_state.step = 1
        st.rerun()
    st.markdown("""---""")
    add_vertical_space(5)
# -------------------------------------------------------
# Step 3: Expected Return Calculation (Hybrid Approach)
# -------------------------------------------------------
if st.session_state.step >= 3:
    st.subheader("Step 3: Expected Return Calculation")

    # If N <= 20 => exact combos, else Poisson
    if number_of_events <= 20:
        st.write("Using exact combination-based approach (N ≤ 20).")
        actual_event_counts = np.arange(number_of_events + 1)
        actual_event_probabilities = np.zeros(number_of_events + 1)
        for k in actual_event_counts:
            prob_sum = 0.0
            for combo in combinations(range(number_of_events), k):
                prob_product = 1.0
                for i in range(number_of_events):
                    if i in combo:
                        prob_product *= random_probabilities[i]
                    else:
                        prob_product *= 1 - random_probabilities[i]
                prob_sum += prob_product
            actual_event_probabilities[k] = prob_sum
    else:
        st.write("Switched to Poisson approximation (N > 20).")
        lambda_val = np.sum(random_probabilities)
        actual_event_counts = np.arange(number_of_events + 1)
        actual_event_probabilities = poisson.pmf(actual_event_counts, lambda_val)

    covered_events = np.arange(1, number_of_events + 1)
    expected_returns = []
    for covered in covered_events:
        e_return = 0.0
        for actual in actual_event_counts:
            e_return += (
                net_return_matrix.loc[covered, actual]
                * actual_event_probabilities[actual]
            )
        expected_returns.append(round(e_return, 6))

    net_return_matrix["Expected_Return"] = expected_returns

    # Simple line chart with Matplotlib
    fig3, ax3 = plt.subplots()
    ax3.plot(
        covered_events, np.array(expected_returns) * 100, marker="o", color="orange"
    )
    ax3.set_xlabel("Number of Events Covered", size=10)
    ax3.set_ylabel("Expected Return (%)", size=10)
    ax3.set_title("Expected Return by Covered Events", size=12)

    st.pyplot(fig3)
    add_vertical_space(2)
    if st.session_state.step == 3:
        if st.button("Next"):
            st.session_state.step = 4
            st.rerun()
    if st.button("Back to Step 2"):
        st.session_state.step = 2
        st.rerun()
    st.markdown("""---""")
    add_vertical_space(5)
# -------------------------
# Step 4: Risk Profiles Analysis
# -------------------------
if st.session_state.step >= 4:
    st.subheader("Step 4: Risk Profiles Analysis")

    # Original list of pool sizes
    event_pool_sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 500, 750, 1000]

    # Ensure user's chosen number_of_events is included
    if number_of_events not in event_pool_sizes:
        event_pool_sizes.append(number_of_events)
    event_pool_sizes.sort()  # keep it sorted for neatness

    st.markdown(
        f"""
    In this section, we analyze how different pool sizes affect overall risk and return, 
    using a Poisson approximation for the number of events that actually occur.
    
    For each tested pool size (including your chosen value {number_of_events}):
    - The pool covers {pool_coverage}% of the total events based on your selection.
    - We compute the expected number of actual events (lambda) as the sum of all event probabilities.
    - We approximate the distribution of actual events with a Poisson(lambda).
    - We calculate the net return for each possible actual event count, then derive:
      - **Mean Expected Return**
      - **Standard Deviation** of returns
      - **Risk/Return Ratio** (std dev / |mean|)
      - **Probability of Loss** (chance that net return is < 0)
    """
    )

    # Prepare lists to store metrics
    mean_returns = []
    std_devs = []
    risk_return_ratios = []
    probability_of_loss = []

    # Loop over each pool size in the updated list
    for num_events in event_pool_sizes:
        covered = int(num_events * pool_coverage / 100)

        # Generate probabilities (same logic as Step 1)
        if prob_range > 0:
            event_probabilities = np.random.uniform(
                (core_probability - prob_range) / 100,
                (core_probability + prob_range) / 100,
                num_events,
            )
        else:
            event_probabilities = np.full(num_events, core_probability / 100)

        # Compute per-event premiums
        premium_per_event = (
            premium_multiplier * coverage_per_event * event_probabilities
        )
        premium_collected = premium_per_event.sum()
        allocated_capital = coverage_per_event * covered

        # Poisson approximation
        from scipy.stats import poisson

        lambda_val = event_probabilities.sum()
        actual_event_counts = np.arange(num_events + 1)
        actual_event_probabilities = poisson.pmf(actual_event_counts, lambda_val)

        # Compute net return for each actual event count
        returns = []
        for actual in actual_event_counts:
            payout = coverage_per_event * min(actual, covered)
            net_ret = (premium_collected - payout) / allocated_capital
            returns.append(net_ret)

        # Mean (expected) return
        mean_return = np.sum(
            [r * p for r, p in zip(returns, actual_event_probabilities)]
        )
        # Variance & std dev
        variance = np.sum(
            [
                ((r - mean_return) ** 2) * p
                for r, p in zip(returns, actual_event_probabilities)
            ]
        )
        std_dev = np.sqrt(variance)
        rr_ratio = std_dev / abs(mean_return) if mean_return != 0 else None
        prob_loss = np.sum(
            [p for r, p in zip(returns, actual_event_probabilities) if r < 0]
        )

        # Store
        mean_returns.append(mean_return)
        std_devs.append(std_dev)
        risk_return_ratios.append(rr_ratio)
        probability_of_loss.append(prob_loss)

    # Build DataFrame
    metrics_df_2 = pd.DataFrame(
        {
            "Number of Events": event_pool_sizes,
            "Mean Expected Return": mean_returns,
            "Standard Deviation": std_devs,
            "Risk/Return Ratio": risk_return_ratios,
            "Probability of Loss": probability_of_loss,
        }
    )

    st.dataframe(metrics_df_2)
    add_vertical_space(2)
    # --------------------------------------------------
    # Visualization 1: Mean Return vs. Std Dev (Dual-Axis)
    # --------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Mean Expected Return on left axis
    ax1.set_xlabel("Number of Events in Pool", size=20)
    ax1.set_ylabel("Mean Expected Return", color="skyblue", size=20)
    ax1.plot(
        event_pool_sizes,
        mean_returns,
        marker="o",
        # linestyle="-",
        color="skyblue",
        label="Mean Return",
    )
    ax1.tick_params(axis="y", labelcolor="skyblue")

    # Dynamically set y-limits for the left axis
    mean_min, mean_max = min(mean_returns), max(mean_returns)
    padding = (mean_max - mean_min) * 0.5  # adjust factor as needed
    # ax1.set_ylim(mean_min - padding, mean_max + padding)
    ax1.set_ylim(-0.02, 0.5)

    # Add secondary y-axis for Standard Deviation
    ax2 = ax1.twinx()
    ax2.set_ylabel("Standard Deviation (Risk)", color="orange", size=20)
    ax2.plot(
        event_pool_sizes,
        std_devs,
        marker="s",
        # linestyle="--",
        color="orange",
        label="Risk (Std Dev)",
    )
    ax2.tick_params(axis="y", labelcolor="orange", size=12)
    ax2.set_ylim(0, max(std_devs) * 1.2)  # 20% extra padding

    # Title and Legends
    fig.suptitle("Impact of Increasing Event Pool Size on Returns & Risk", size=24)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # ax1.grid()
    st.pyplot(fig)
    add_vertical_space(2)
    # --------------------------------------------------
    # Visualization 2: Probability of Loss vs. Pool Size
    # --------------------------------------------------
    fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
    ax_loss.plot(
        event_pool_sizes,
        probability_of_loss,
        marker="o",
        # linestyle="-",
        linewidth=3,
        color="orange",
    )
    ax_loss.set_title("Probability of Loss vs. Pool Size", size=24)
    ax_loss.set_xlabel("Number of Events in Pool", size=20)
    ax_loss.set_ylabel("Probability of Negative Return", size=20)
    # ax_loss.grid(True)
    st.pyplot(fig_loss)

    # Navigation
    if st.session_state.step == 4:
        if st.button("Final Figures"):
            st.session_state.step = 5
            st.rerun()
    if st.button("Back to Step 3"):
        st.session_state.step = 3
        st.rerun()
    st.markdown("""---""")
    add_vertical_space(5)


############################
# Helper for color grading
############################
def color_grade_percentage(value):
    """
    Returns a CSS color based on a simple threshold logic for percentages.
    Lower is generally better.
    """
    if value < 2:
        return "green"
    elif value < 5:
        return "orange"
    else:
        return "red"


def color_grade_return(value):
    """
    Returns a CSS color for net return (higher is better).
    """
    if value < 0:
        return "red"
    elif value < 5:
        return "orange"
    else:
        return "green"


def color_grade_risk_sigma(value):
    """
    Returns a CSS color for risk (σ). Lower is better.
    """
    if value < 0.05:
        return "green"
    elif value < 0.15:
        return "orange"
    else:
        return "red"


######################################
# Step 5: Final Metrics & Insights
######################################
if st.session_state.step >= 5:
    st.subheader("Step 5: Final Metrics & Insights")

    # 1. Determine how many events are covered (integer)
    coverage_int = int(number_of_events * pool_coverage / 100)

    # 2. Expected Net Return from net_return_matrix (Step 3)
    if coverage_int in net_return_matrix.index:
        expected_net_return = (
            net_return_matrix.loc[coverage_int, "Expected_Return"] * 100
        )
    else:
        expected_net_return = 0.0

    # 3. Risk (σ) & Probability of Loss from metrics_df_2 (Step 4)
    row_mask = metrics_df_2["Number of Events"] == number_of_events
    if any(row_mask):
        risk_sigma = metrics_df_2.loc[row_mask, "Standard Deviation"].values[0]
        probability_of_loss = (
            metrics_df_2.loc[row_mask, "Probability of Loss"].values[0] * 100
        )
    else:
        risk_sigma = 0.0
        probability_of_loss = 0.0

    # 4. Cost for $1,000 coverage
    cost_of_insurance_1000 = random_probabilities.mean() * premium_multiplier * 1000

    # 5. Probability of no payout (Poisson or exact combos).
    #    If you prefer Poisson:
    # prob_no_payout = probability_of_no_payout_poisson(
    #     number_of_events, coverage_int, random_probabilities
    # ) * 100
    #    If you want the exact combos approach (feasible for smaller N):
    prob_no_payout = (
        probability_of_no_payout_poisson(
            number_of_events, coverage_int, random_probabilities
        )
        * 100
    )

    # --- Apply color grading ---
    net_return_color = color_grade_return(expected_net_return)
    risk_color = color_grade_risk_sigma(risk_sigma)
    prob_loss_color = color_grade_percentage(probability_of_loss)
    prob_no_payout_color = color_grade_percentage(prob_no_payout)

    # Build two sections: Investor metrics & Purchaser metrics
    # We'll use HTML in st.markdown with unsafe_allow_html=True to color the text.

    st.markdown("#### Investor Metrics")
    st.markdown(
        f"""
    <ul>
      <li><strong>Net Return:</strong> <span style="color:{net_return_color}">{expected_net_return:.2f}%</span></li>
      <li><strong>Risk (σ):</strong> <span style="color:{risk_color}">{risk_sigma:.4f}</span></li>
      <li><strong>Probability of Loss:</strong> <span style="color:{prob_loss_color}">{probability_of_loss:.2f}%</span></li>
    </ul>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Purchaser Metrics")
    st.markdown(
        f"""
    <ul>
      <li><strong>Cost of $1,000 Coverage:</strong> ${cost_of_insurance_1000:.2f}</li>
      <li><strong>Probability of Not Getting Paid:</strong> 
          <span style="color:{prob_no_payout_color}">{prob_no_payout:.2f}%</span>
      </li>
    </ul>
    """,
        unsafe_allow_html=True,
    )

    # Buttons for navigation
    if st.button("Back to Step 4"):
        st.session_state.step = 4
        st.rerun()
