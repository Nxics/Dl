from projects.image_captioning.text_baselines import compute_tfidf_terms


def test_when_tfidf_terms_computed_then_distinctive_terms_are_returned():
    captions = [
        'a dog runs through grass',
        'a dog jumps over grass',
        'a child sits on a bench',
        'a child plays on grass',
    ]

    terms = compute_tfidf_terms(captions, top_k=5, min_document_frequency=1)

    assert terms
    assert all(term.tfidf > 0 for term in terms)
    assert {term.term for term in terms}.intersection({'dog', 'child', 'grass'})
