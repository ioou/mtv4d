from mtv4d.annos_4d.helper import format_floats  # 假设mtv4d是你的项目名，根据实际情况调整

def test_format_floats_single_float():
    assert format_floats(3.1415926) == 3.1416

def test_format_floats_dict():
    input_dict = {"pi": 3.1415926, "e": 2.71828}
    expected_dict = {"pi": 3.1416, "e": 2.7183}
    assert format_floats(input_dict) == expected_dict

def test_format_floats_list():
    input_list = [1.23456, 7.890123]
    expected_list = [1.2346, 7.8901]
    assert format_floats(input_list) == expected_list

def test_format_integers():
    assert format_floats(42) == 42

def test_format_strings():
    assert format_floats("hello") == "hello"

def test_format_tuples():
    input_tuple = (1.2345, 6.7890)
    expected_tuple = (1.2346, 6.789)
    assert format_floats(input_tuple) == expected_tuple