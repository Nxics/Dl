from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from projects.image_captioning.data import read_flickr8k_captions
from projects.image_captioning.vocab import tokenize


def summarize_flickr8k(images_dir: str | Path, captions_path: str | Path) -> dict[str, object]:
    images_dir = Path(images_dir)
    captions = read_flickr8k_captions(captions_path)
    caption_lengths = captions['caption'].map(lambda text: len(tokenize(text)))
    image_names = sorted(set(captions['image']))

    widths: list[int] = []
    heights: list[int] = []
    missing: list[str] = []
    for image_name in image_names:
        image_path = images_dir / image_name
        if not image_path.exists():
            missing.append(image_name)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        widths.append(width)
        heights.append(height)

    return {
        'num_caption_rows': int(len(captions)),
        'num_unique_images_in_captions': int(len(image_names)),
        'num_existing_images': int(len(widths)),
        'num_missing_images': int(len(missing)),
        'caption_length_min': int(caption_lengths.min()),
        'caption_length_mean': float(caption_lengths.mean()),
        'caption_length_max': int(caption_lengths.max()),
        'image_width_mean': float(pd.Series(widths).mean()) if widths else None,
        'image_height_mean': float(pd.Series(heights).mean()) if heights else None,
        'missing_images': missing[:20],
    }


def save_basic_eda_plots(images_dir: str | Path, captions_path: str | Path,
                         output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captions = read_flickr8k_captions(captions_path)
    caption_lengths = captions['caption'].map(lambda text: len(tokenize(text)))

    plt.figure()
    plt.hist(caption_lengths, bins=30)
    plt.xlabel('Caption length in words')
    plt.ylabel('Count')
    plt.title('Flickr8k caption length distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'caption_lengths.png')
    plt.close()

    words = captions['caption'].str.lower().str.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?").explode()
    top_words = words.value_counts().head(20)
    plt.figure(figsize=(10, 5))
    top_words.plot(kind='bar')
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.title('Top 20 caption words')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_words.png')
    plt.close()
