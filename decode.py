import decode_no_breakpoint
import decode_breakpoint


def decode(ciphertext, has_breakpoint):
    if has_breakpoint:
        return decode_breakpoint.decode(ciphertext)
    else:
        return decode_no_breakpoint.decode(ciphertext)
