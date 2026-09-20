# char.py
charset_list = [
    # 大写（常规/脂肪族原子）
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "Ge", 
    # 小写（芳香族原子，新增了 se 和 si）
    "n", "c", "b", "o", "s", "p", "se", "si", 
    # 数字
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    # 括号
    "(", ")", "[", "]", 
    # 符号与起止符
    "-", "=", "#", "/", "\\", "+", "@", "%", "*", "^", ">" 
]

# 自动生成字符字典
charset_dict = {char: idx for idx, char in enumerate(charset_list)}

# 预先分类单字符和双字符，供其他模块直接调用
charset_list2 = [char for char in charset_list if len(char) == 2]
charset_list1 = [char for char in charset_list if len(char) == 1]