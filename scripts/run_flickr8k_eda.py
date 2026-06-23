import argparse
from pprint import pprint

from projects.image_captioning.eda import save_basic_eda_plots, summarize_flickr8k


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--captions-path', required=True)
    parser.add_argument('--output-dir', default='reports/eda')
    args = parser.parse_args()

    summary = summarize_flickr8k(args.images_dir, args.captions_path)
    pprint(summary)
    save_basic_eda_plots(args.images_dir, args.captions_path, args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
