"""Run a TF-IDF caption-text experiment for the Flickr8k project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.image_captioning.text_baselines import compute_tfidf_terms  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute TF-IDF caption terms.')
    parser.add_argument('--captions', type=Path, default=PROJECT_ROOT / 'DATA' / 'flickr8k' / 'splits' / 'train.csv')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'reports' / 'model' / 'tfidf_caption_terms.csv')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--min-document-frequency', type=int, default=5)
    parser.add_argument('--keep-stop-words', action='store_true')
    args = parser.parse_args()

    frame = pd.read_csv(args.captions)
    terms = compute_tfidf_terms(
        frame['caption'].tolist(),
        top_k=args.top_k,
        min_document_frequency=args.min_document_frequency,
        stop_words=set() if args.keep_stop_words else None,
    )
    output_frame = pd.DataFrame([
        {
            'term': term.term,
            'tfidf': term.tfidf,
            'frequency': term.frequency,
            'document_frequency': term.document_frequency,
        }
        for term in terms
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output, index=False)
    print(output_frame.head(20).to_string(index=False))
    print(f'Saved TF-IDF terms to {args.output}')


if __name__ == '__main__':
    main()
