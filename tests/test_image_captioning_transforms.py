from projects.image_captioning.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    describe_image_preprocessing,
)


def test_when_preprocessing_described_then_imagenet_normalization_is_reported():
    description = describe_image_preprocessing(image_size=224)

    assert description['resize'] == (224, 224)
    assert description['normalization_mean'] == IMAGENET_MEAN
    assert description['normalization_std'] == IMAGENET_STD
    assert description['training_augmentation']
