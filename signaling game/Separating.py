import numpy as np

def separating(theta, sender_util, receiver_util):
    """
    ----- Input -----
        
        # For binary type/message/action
        
        theta:   prior beilief of type 0 - int between 0 and 1 
        sender_util:   sender's utility - TxMxA np.array
        receiver_util:   receiver's utility - TxMxA np.array

    ----- Output -----

        Equilibrium:  Eq - equilibrium category (3 = seperating equilibirum)
                      pi_s - sender's pure strategy (pi_s[type])
                      pi_r - receiver's pure strategy (pi_r[message])
                      p_range - feasible poesterior belief range for b(type0|M0)
                      q_range - feasible poesterior belief range for b(type0|M1)
                      EU_S - sender's expected utility
                      EU_R - receiver's expected utility
        # if return [], there is no equilibrum under this category

    """
    
    PBE = []
    # (M1, M2)
    S1 = ('M{}, M{}'.format(0, 1))
    p = 1
    q = 0

    Ai = 0 if receiver_util[0][0][0] > receiver_util[0][0][1] else 1
    Aj = 0 if receiver_util[1][1][0] > receiver_util[1][1][1] else 1

    if sender_util[0][0][Ai] >= sender_util[0][1][Aj] and sender_util[1][1][Aj] >= sender_util[1][0][Ai]:
        EU_R = theta * receiver_util[0][0][Ai] + (1 - theta) * receiver_util[1][1][Aj]
        EU_S = theta * sender_util[0][0][Ai] + (1 - theta) * sender_util[1][1][Aj]
        S2 = ('A{}, A{}'.format(Ai, Aj))
        PBE.append({'Eq':3, 'pi_s':S1, 'pi_r':S2, 'p_range':(p, p), 'q_range':(q, q),  'EU_S':EU_S, 'EU_R':EU_R})
    # (M2, M1)
    S1 = ('M{}, M{}'.format(1, 0))
    p = 0
    q = 1

    Ai = 0 if receiver_util[1][0][0] > receiver_util[1][0][1] else 1
    Aj = 0 if receiver_util[0][1][0] > receiver_util[0][1][1] else 1

    if sender_util[0][0][Ai] <= sender_util[0][1][Aj] and sender_util[1][1][Aj] <= sender_util[1][0][Ai]:
        EU_R = theta * receiver_util[0][1][Ai] + (1 - theta) * receiver_util[1][0][Aj]
        EU_S = theta * sender_util[0][1][Ai] + (1 - theta) * sender_util[1][0][Aj]
        S2 = ('A{}, A{}'.format(Ai, Aj))
        PBE.append({'Eq':3, 'pi_s':S1, 'pi_r':S2, 'p_range':(p, p),  'q_range':(q, q), 'EU_S':EU_S, 'EU_R':EU_R})

    return PBE