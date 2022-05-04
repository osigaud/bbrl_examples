def is_vec_of_ones(vec) -> bool:
    # print(vec)
    # print(vec.shape)
    for subvec in vec:
        for i in subvec:
            # print(f"{i} ")
            if not i == 1:
                # print(f"{i} is not one")
                return False
    return True
