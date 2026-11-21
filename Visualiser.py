import matplotlib.pyplot as plt


class Visualiser:
    """2x3 dashboard built only from real data and real model output."""

    def show_dashboard(self, matches, actual, predicted, model_mae, elo):
        """
        matches   : DataFrame from the preprocessor (needs first_innings_runs, date, toss_decision)
        actual    : true first-innings totals for the test set
        predicted : model predictions for the test set
        model_mae : dict {label: MAE}
        elo       : Series of team Elo ratings
        """
        fig, axs = plt.subplots(2, 3, figsize=(17, 9))
        fig.suptitle("PitchVision: Cricket Analytics & Model Performance", fontsize=16, fontweight="bold")

        ax = axs[0, 0]
        ax.hist(matches["first_innings_runs"], bins=25, color="skyblue", edgecolor="black")
        ax.set(title="1. First-Innings Score Distribution", xlabel="Runs", ylabel="Matches")

        ax = axs[0, 1]
        yearly = matches.groupby(matches["date"].dt.year)["first_innings_runs"].mean()
        ax.plot(yearly.index, yearly.values, marker="o", color="forestgreen")
        ax.set(title="2. Average First-Innings Score by Year", xlabel="Year", ylabel="Runs")
        ax.grid(True, linestyle="--", alpha=0.6)

        ax = axs[0, 2]
        ax.scatter(actual, predicted, alpha=0.5, color="purple", edgecolors="k", s=25)
        lims = [min(min(actual), min(predicted)), max(max(actual), max(predicted))]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set(title="3. Actual vs Predicted (test set)", xlabel="Actual", ylabel="Predicted")
        ax.legend()
