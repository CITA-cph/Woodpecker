import sys, time, msvcrt, os

def readUserInput(caption, default, timeout = 2):

    start_time = time.time()
    print(int(timeout), "%s(%s):"%(caption, default))
    inp = ""

    while True:
        if msvcrt.kbhit():
            chr = msvcrt.getche()

            if ord(chr) == 13: # enter_key
                break
            elif ord(chr) >= 32: #space_char
                inp += chr.decode("utf-8") 

        if len(inp) == 0 and (time.time() - start_time) > timeout:
            break
 
    if len(inp) > 0:
        return inp
    else:
        return default