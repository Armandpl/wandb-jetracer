import sys
sys.path.append("../")

from utils.utils import split_list_by_pct


def test_split_list_by_pct():
    # Arrange
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pcts = [0.8, 0.2]

    # Act
    new_list = split_list_by_pct(list, pcts)

    # Assert
    assert new_list == [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]
