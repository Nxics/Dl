from projects.image_captioning.experiments import (
    get_experiment_configs,
    summarize_experiment_configs,
)


def test_when_experiment_configs_loaded_then_baseline_is_present():
    configs = get_experiment_configs()

    names = {config.name for config in configs}

    assert 'baseline_vgg16_lstm' in names
    assert 'augmented_vgg16_lstm' in names


def test_when_experiment_configs_summarized_then_report_fields_exist():
    rows = summarize_experiment_configs()

    assert rows
    assert {
        'name',
        'image_preprocessing',
        'augmentation',
        'word_embedding',
        'encoder',
        'decoder',
        'expected_tradeoff',
    }.issubset(rows[0])
