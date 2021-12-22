from karmarkar_karp import _PartialSet


def test_partial():
    p1 = _PartialSet(m=3, first_element=10)
    p2 = _PartialSet(m=3, first_element=20)
    p3 = _PartialSet(m=3, first_element=5)
    p4 = _PartialSet(m=3, first_element=15)
    p5 = _PartialSet(m=3, first_element=7)

    p1 | p2
    assert p1.sets == [[], [10], [20]]
    assert p1.errors == [0, 10, 20]
    p1 | p3
    assert p1.sets == [[5], [10], [20]]
    assert p1.errors == [0, 5, 15]
    p4 | p5
    assert p4.sets == [[], [7], [15]]
    assert p4.errors == [0, 7, 15]
    p1 | p4
    assert p1.sets == [[10, 7], [5, 15], [20]]
    assert p1.errors == [0, 3, 3]

