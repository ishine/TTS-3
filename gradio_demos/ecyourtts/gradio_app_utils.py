def controller_reader(control_input):
    """
    Reads the control input and returns the corresponding values.

    Input format
        ```
            word1| m 0.9  // m = multiply, s = set
            w1_p1|100.0   // value_w1_p1_hat = 0.9 * 100.0
            w1_p2|120.0   // value_w1_p2_hat = 0.9 * 120.0
            ...
            wordK| s 100.0
            100.0         // values for <SPACE>
            #
            word2| m 1.0
            w2_p1|100.0
            ...
            wordL|100.0
            100.0
            #
        ```
    """
    word_flag = True
    word_coeff = None
    phone_coeffs = []
    for line in control_input.splitlines():
        # print(line)
        if line == "#":
            word_flag = True
            continue
        elif "|" in line and word_flag:
            word, value = line.split("|")
            word_coeff = value
            word_flag = False
        elif "|" in line:
            phone, value = line.split("|")

            if "m" in word_coeff:
                # multiply mode
                word_coeff_ = word_coeff.replace("m", "")
                phone_coeffs.append((phone, float(value) * float(word_coeff_)))
            elif "s" in word_coeff:
                # set mode
                word_coeff_ = word_coeff.replace("s", "")
                phone_coeffs.append((phone, float(word_coeff_)))
            else:
                # default mode - multiply
                phone_coeffs.append((phone, float(value) * float(word_coeff)))
        elif "|" not in line and len(line) != 0:
            phone_coeffs.append((" ", float(line)))
    return phone_coeffs


def controller_writer(phonemes, values):
    """Creates the string that is read by the `controller_reader()`"""
    control_data = ""
    word = ""
    word_values = []
    count = 0
    # breakpoint()
    for phone, value in zip(phonemes, values):
        # print(f"{phone} - {value}")
        if phone == " " or count == len(phonemes) - 1:
            if count == len(phonemes) - 1:
                word += phone
                word_values.append(value)
            control_data += f"{word}|1.0\n"
            for p_w in word:
                control_data += f"{p_w}|{word_values.pop(0)}\n"
            if phone == " ":
                control_data += f"{value}\n"
                control_data += "#\n"
                word = ""
        else:
            word += phone
            word_values.append(value)
        count += 1
    return control_data
