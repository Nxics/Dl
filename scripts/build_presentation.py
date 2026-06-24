"""Build a PowerPoint presentation from PRESENTATION.md."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_markdown_slides(markdown_text: str) -> list[tuple[str, list[str]]]:
    slides: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_bullets: list[str] = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if line.startswith('## '):
            if current_title is not None:
                slides.append((current_title, current_bullets))
            current_title = line.removeprefix('## ').strip()
            current_bullets = []
        elif line.startswith('- ') and current_title is not None:
            current_bullets.append(line.removeprefix('- ').strip())

    if current_title is not None:
        slides.append((current_title, current_bullets))

    return slides


def build_presentation(input_path: Path, output_path: Path) -> None:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Missing dependency: python-pptx. Install dev requirements with '
            '`python -m pip install -r requirements-dev.txt`.'
        ) from exc

    slides = _parse_markdown_slides(input_path.read_text(encoding='utf-8'))
    presentation = Presentation()

    for index, (title, bullets) in enumerate(slides):
        layout = presentation.slide_layouts[0] if index == 0 else presentation.slide_layouts[1]
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title

        if index == 0:
            subtitle = slide.placeholders[1]
            subtitle.text = 'Flickr8k image captioning project'
            continue

        body = slide.placeholders[1]
        body.text = ''
        text_frame = body.text_frame
        text_frame.margin_left = Inches(0.1)
        text_frame.margin_right = Inches(0.1)

        for bullet_index, bullet in enumerate(bullets):
            paragraph = text_frame.paragraphs[0] if bullet_index == 0 else text_frame.add_paragraph()
            paragraph.text = bullet
            paragraph.level = 0
            paragraph.font.size = Pt(24)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build PowerPoint slides from PRESENTATION.md.')
    parser.add_argument('--input', type=Path, default=Path('PRESENTATION.md'))
    parser.add_argument('--output', type=Path, default=Path('presentation/image_captioning_presentation.pptx'))
    args = parser.parse_args()

    build_presentation(args.input, args.output)
    print(f'Saved presentation to {args.output}')


if __name__ == '__main__':
    main()
