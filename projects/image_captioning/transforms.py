from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD),
    ])


def describe_image_preprocessing(image_size: int = 224) -> dict[str, object]:
    """Return the preprocessing settings used by VGG16 training/inference."""

    return {
        'image_size': image_size,
        'resize': (image_size, image_size),
        'normalization_mean': IMAGENET_MEAN,
        'normalization_std': IMAGENET_STD,
        'training_augmentation': [
            'RandomHorizontalFlip(p=0.5)',
            'RandomRotation(degrees=10)',
        ],
        'eval_augmentation': [],
    }
