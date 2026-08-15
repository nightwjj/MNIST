import json
import os


def save_experiment_config(
    experiment_config,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    config_path = os.path.join(
        output_dir,
        "config.json",
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_config,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return config_path

def save_experiment_results(
    experiment_results,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    results_path = os.path.join(
        output_dir,
        "results.json",
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return results_path