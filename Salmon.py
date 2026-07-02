#Main python file for Salmon chess engine
#Designed to interface through Universal Chess Interface (UCI) with a chess GUI
#The UCI loop itself lives in common.run_uci_loop; this entry point just binds
#it to the classical alpha-beta search.
#Built by Abhinav Brahmarouthu

import common
import minimax

VERSION = "0.1"

if __name__ == "__main__":
    common.run_uci_loop("Salmon" + VERSION, "Abhinav Brahmarouthu", minimax.bestMove)
