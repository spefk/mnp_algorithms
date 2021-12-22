from typing import List


def Dynamic2MNP(data: List[int]) -> bool:
    """
    Dynamic programming algorithm to determine if
    there is a perfect partitioning of numbers in
    two sets.

    time complexity: O(S * N)
    memory complexity: O(S)
    where S = sum(data), N = len(data)
    """
    s = sum(data)
    if s % 2 == 1:
        return False

    perfect = s // 2
    arr = [0] * (perfect + 1)
    arr[0] = 1
    for x in data:
        for i in range(perfect, x - 1, -1):
            arr[i] = max([arr[i], arr[i - x]])

    return arr[-1] == 1


if __name__ == '__main__':
    print(Dynamic2MNP([1, 3, 2]))
    print(Dynamic2MNP([1, 3, 2, 1]))
    print(Dynamic2MNP([1, 3, 5, 1]))
    print(Dynamic2MNP([1, 2, 3, 4, 5, 6, 7]))
    print(Dynamic2MNP([1, 2, 3, 4, 5, 6, 7, 8]))
    print(Dynamic2MNP(list(range(0, 201))))
    print(Dynamic2MNP(list(range(0, 202))))
