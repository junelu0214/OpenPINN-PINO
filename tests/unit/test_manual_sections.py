"""Prevent content silently disappearing during future manual regrouping."""
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / 'docs/_ext/manual_sections.py'
    spec = importlib.util.spec_from_file_location('manual_sections_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_nested_anchors_have_their_owning_page():
    m = module()
    m.MANUAL_PAGES['test'] = [('first', 'First', '', ('a',)), ('second', 'Second', '', ('b',))]
    sections = m.manual_sections('<main><section id="a"><h2>A</h2><details><p id="deep">Details</p></details></section><section id="b"><h2>B</h2></section></main>')
    assert m.section_routes('test', sections) == {'a': 'test_guide_first.html', 'deep': 'test_guide_first.html', 'b': 'test_guide_second.html'}


@pytest.mark.parametrize('anchors', [('a',), ('a', 'b', 'b'), ('a', 'b', 'missing')])
def test_missing_or_repeated_assignment_fails(anchors):
    m = module()
    m.MANUAL_PAGES['test'] = [('first', 'First', '', anchors)]
    sections = {'a': ('A', '<section id="a"></section>'), 'b': ('B', '<section id="b"></section>')}
    with pytest.raises(ValueError, match='coverage mismatch'):
        m.section_routes('test', sections)


def test_duplicate_nested_anchor_fails():
    m = module()
    m.MANUAL_PAGES['test'] = [('first', 'First', '', ('a', 'b'))]
    sections = {'a': ('A', '<section id="a"><p id="same"></p></section>'), 'b': ('B', '<section id="b"><p id="same"></p></section>')}
    with pytest.raises(ValueError, match='Duplicate manual anchor'):
        m.section_routes('test', sections)
