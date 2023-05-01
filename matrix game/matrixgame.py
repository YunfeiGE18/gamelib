import numpy as np
import nashpy as nash



def zerosum(A):
    """
    ----- Input -----

        A:   row player's utility - AxA np.array (minimizer)

    ----- Output -----

        pi_r:   row player's strategy - 1xA np.array
        pi_c:   colum player's strategy - 1xA np.array
        gameval:   game value - 1x2 np.array 
                   # gameval[0]: game value for row player
                     gameval[1]:  game value for column player
    """
    rps = nash.Game(A)
    eqs = rps.support_enumeration()
    equ = list(eqs)

    if equ !=[]:
        pi_r = equ[0][0]    # strategy for row player
        pi_c = equ[0][1]    # strategy for column player

        gameval = rps[pi_r, pi_c]
    else: 
        raise Exception("The game has no equilibrium.")
    
    return pi_r, pi_c, gameval





def generalsum(A,B):
    """
    ----- Input -----

        A:   row player's utility - AxB np.array (minimizer)
        B:   colum player's utility - AxB np.array, optional

    ----- Output -----

        pi_r:   row player's strategy - 1xA np.array
        pi_c:   colum player's strategy - 1xB np.array
        gameval:   game value - 1x2 np.array 
                   # gameval[0]: game value for row player
                     gameval[1]:  game value for column player
    """
    rps = nash.Game(A, B)
    eqs = rps.support_enumeration()
    equ = list(eqs)

    if equ !=[]:
        pi_r = equ[0][0]    # strategy for row player
        pi_c = equ[0][1]    # strategy for column player

        gameval = rps[pi_r, pi_c]
    else: 
        raise Exception("The game has no equilibrium.")
    
    return pi_r, pi_c,gameval




