def is_vec_of_ones(vec) -> bool:
    for i in vec:
        if not i == 1:
            print(f"{i} is not one")
            return False
    return True
