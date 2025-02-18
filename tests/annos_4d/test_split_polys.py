
import pytest


def process_poly_with_vis(vis):
    output_list = []
    vis1 = [i and j for i, j in zip(vis[:-1], vis[1:])]
    vis2 = [False] + vis1 + [False]
    for i in range(len(vis2) - 1):
        v1 = vis2[i]
        v2 = vis2[i + 1]
        if v1 != v2:
            output_list += [i]
    output = []
    assert len(output_list) % 2 == 0
    for i in range(int(len(output_list) / 2)):
        a1 = output_list[2 * i]
        a2 = output_list[2 * i + 1] if 2 * i + 1 < len(output_list) else len(vis) - 1
        output += [(a1, a2)]
    return output

 
@pytest.mark.parametrize("vis,expected", [
    ([True, True, True, False, False, True, True], [(2, 3), (4, 5)]),
    ([True, True, True, True], []),
    ([False], []),
    ([True, False, True, False], [(0, 1), (2, 3)]),
    ([False, False, False, False], []),
    ([True, False, True, True, False, True], [(0, 1), (3, 4)])
])
def test_process_poly_with_vis(vis, expected):
    assert process_poly_with_vis(vis) == expected