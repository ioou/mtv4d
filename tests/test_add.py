from mtv4d.utils.main import add

def test_add_integers():
    """ 测试 add 函数能否正确地将两个整数相加 """
    assert add(1, 2) == 3

def test_add_floats():
    """ 测试 add 函数能否正确地将两个浮点数相加 """
    assert add(1.5, 2.7) == 4.2
