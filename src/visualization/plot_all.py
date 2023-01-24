from ._number_of_positive import plot_number_of_positive
from ._curves import plot_curves_all, plot_curves_average


if __name__ == '__main__':
    run_ids = list(range(1, 1 + 100))

    plot_number_of_positive(
        run_ids,
        active_learning_strategies=['uncertain', 'stratified'],
        path_save="outputs/visualization/number_of_positive.svg")
    plot_curves_all(
        run_ids=run_ids,
        active_learning_strategies=['uncertain'],
        path_save="outputs/visualization/curves_uncertain_all.svg")
    plot_curves_average(
        run_ids=run_ids,
        active_learning_strategies=['uncertain'],
        path_save="outputs/visualization/curves_uncertain_average.svg")
