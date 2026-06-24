from scripts.build_presentation import _parse_markdown_slides


def test_when_markdown_slides_parsed_then_titles_and_bullets_are_extracted():
    markdown = """# Presentation

## Slide 1

- first point
- second point

## Slide 2

- final point
"""

    slides = _parse_markdown_slides(markdown)

    assert slides == [
        ('Slide 1', ['first point', 'second point']),
        ('Slide 2', ['final point']),
    ]
