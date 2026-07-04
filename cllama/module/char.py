# char.py
charset_list = [
    
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "Ge", 
    
    "n", "c", "b", "o", "s", "p", "se", "si", 
    
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    
    "(", ")", "[", "]", 
    
    "-", "=", "#", "/", "\\", "+", "@", "%", "*", "^", ">" 
]


charset_dict = {char: idx for idx, char in enumerate(charset_list)}


charset_list2 = [char for char in charset_list if len(char) == 2]
charset_list1 = [char for char in charset_list if len(char) == 1]