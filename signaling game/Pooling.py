import numpy as np

def pooling(m, theta, sender_util, receiver_util):
    """
    ----- Input -----
        
        # For binary type/message/action
        
        m: pooling message
        theta:   prior beilief of type 0 - int between 0 and 1 
        sender_util:   sender's utility - TxMxA np.array
        receiver_util:   receiver's utility - TxMxA np.array

    ----- Output -----

        Equilibrium:  Eq - equilibrium category (1/2 = pooling equilibirum with M0/M1)
                      pi_s - sender's pure strategy (pi_s[type])
                      pi_r - receiver's pure strategy (pi_r[message])
                      p_range - feasible poesterior belief range for b(type0|M0)
                      q_range - feasible poesterior belief range for b(type0|M1)
                      EU_S - sender's expected utility
                      EU_R - receiver's expected utility
        # if return [], there is no equilibrum under this category

    """
    
    PBE = []
    if m == 0:
        p = theta
        Util_receiverr_for_A1 = p * receiver_util[0][m][0] + (1-p) * receiver_util[1][m][0]
        Util_receiverr_for_A2 = p * receiver_util[0][m][1] + (1-p) * receiver_util[1][m][1]
        
        Aj = 0 if Util_receiverr_for_A1 > Util_receiverr_for_A2 else 1
        
        if sender_util[0][0][Aj] >= sender_util[0][1][0] and sender_util[1][0][Aj] >= sender_util[1][1][0]:
            temp = (receiver_util[0][1][0] + receiver_util[1][1][1] - receiver_util[1][1][0] - receiver_util[0][1][1])
            q = (receiver_util[1][1][1] - receiver_util[1][1][0]) / temp
            F1_empty = 0
            q_min = -1
            q_max = -1
            if temp > 0 and q > 1:
                F1_empty = 1
            elif temp > 0 and q <= 1:
                q_min = max(q, 0)
                q_max = 1 
            elif temp < 0 and q < 0:
                F1_empty = 1
            elif temp < 0 and q >= 0:
                q_min = 0
                q_max = min(q, 1)
            elif temp == 0 and receiver_util[1][1][0] >= receiver_util[1][1][1]:
                q_min = 0
                q_max = 1
            elif temp == 0 and receiver_util[1][1][0] < receiver_util[1][1][1]:
                F1_empty = 1
            
            if not F1_empty:
                EU_R = theta * receiver_util[0][0][Aj]  + (1 - theta) * receiver_util[1][0][Aj]
                EU_S = theta * sender_util[0][0][Aj] + (1 - theta) * sender_util[1][0][Aj]
                S1 = ('M{}, M{}'.format(m, m))
                S2 = ('A{}, A{}'.format(Aj, 0))
                PBE.append({'Eq':1, 'S1':S1, 'S2':S2,  'p_range':(p, p), 'q_range':(q_min, q_max), 'EU_S':EU_S, 'EU_R':EU_R})
    
        if sender_util[0][0][Aj] >= sender_util[0][1][1] and sender_util[1][0][Aj] >= sender_util[1][1][1]:
            temp = (receiver_util[0][1][0] + receiver_util[1][1][1] - receiver_util[1][1][0] - receiver_util[0][1][1])
            q = (receiver_util[1][1][1] - receiver_util[1][1][0]) / temp
            F2_empty = 0
            q_min = -1
            q_max = -1
            if temp > 0 and q < 0:
                F2_empty = 1
            elif temp > 0 and q >= 0:
                q_min = 0
                q_max = min(q ,1)
            elif temp < 0 and q > 1:
                F2_empty = 1
            elif temp < 0 and q <= 1:
                q_min = max(q, 0)
                q_max = 1
            elif temp == 0 and receiver_util[1][1][0] <= receiver_util[1][1][1]:
                q_min = 0
                q_max = 1
            elif temp == 0 and receiver_util[1][1][0] > receiver_util[1][1][1]:
                F2_empty = 1
            
            if not F2_empty:
                EU_R = theta * receiver_util[0][0][Aj]  + (1 - theta) * receiver_util[1][0][Aj]
                EU_S = theta * sender_util[0][0][Aj] + (1 - theta) * sender_util[1][0][Aj]
                S1 = ('M{}, M{}'.format(m, m))
                S2 = ('A{}, A{}'.format(Aj, 1))
                PBE.append({'Eq':1, 'S1':S1, 'S2':S2,  'p_range':(p, p), 'q_range':(q_min, q_max), 'EU_S':EU_S, 'EU_R':EU_R})
        return PBE
    elif m == 1:
        q = theta   
        Util_receiverr_for_A1 = q * receiver_util[0][m][0] + (1-q) * receiver_util[1][m][0]
        Util_receiverr_for_A2 = q * receiver_util[0][m][1] + (1-q) * receiver_util[1][m][1]
        
        Aj = 0 if Util_receiverr_for_A1 > Util_receiverr_for_A2 else 1
        
        if sender_util[0][1][Aj] >= sender_util[0][0][0] and sender_util[1][1][Aj] >= sender_util[1][0][0]:
            temp = (receiver_util[0][0][0] + receiver_util[1][0][1] - receiver_util[1][0][0] - receiver_util[0][0][1])
            p = (receiver_util[1][0][1] - receiver_util[1][0][0]) / temp
            F1_empty = 0
            p_min = -1
            p_max = -1
            if temp > 0 and p > 1:
                F1_empty = 1
            elif temp > 0 and p <= 1:
                p_min = max(p, 0)
                p_max = 1 
            elif temp < 0 and p < 0:
                F1_empty = 1
            elif temp < 0 and p >= 0:
                p_min = 0
                p_max = min(p, 1)
            elif temp == 0 and receiver_util[1][0][0] >= receiver_util[1][0][1]:
                p_min = 0
                p_max = 1
            elif temp == 0 and receiver_util[1][0][0] < receiver_util[1][0][1]:
                F1_empty = 1
            
            if not F1_empty:
                EU_R = theta * receiver_util[0][1][Aj]  + (1 - theta) * receiver_util[1][1][Aj]
                EU_S = theta * sender_util[0][1][Aj] + (1 - theta) * sender_util[1][1][Aj]
                S1 = ('M{}, M{}'.format(m, m))
                S2 = ('A{}, A{}'.format(0, Aj))
                PBE.append({'Eq':2, 'S1':S1, 'S2':S2,  'p_range':(p_min, p_max), 'q_range':(q, q), 'EU_S':EU_S, 'EU_R':EU_R})

        if sender_util[0][1][Aj] >= sender_util[0][0][1] and sender_util[1][1][Aj] >= sender_util[1][0][1]:
            temp = (receiver_util[0][0][0] + receiver_util[1][0][1] - receiver_util[1][0][0] - receiver_util[0][0][1])
            p = (receiver_util[1][0][1] - receiver_util[1][0][0]) / temp
            F2_empty = 0
            p_min = -1
            p_max = -1
            if temp > 0 and p < 0:
                F2_empty = 1
            elif temp > 0 and p >= 0:
                p_min = 0
                p_max = min(p ,1)
            elif temp < 0 and p > 1:
                F2_empty = 1
            elif temp < 0 and p <= 1:
                p_min = max(p, 0)
                p_max = 1
            elif temp == 0 and receiver_util[1][0][0] <= receiver_util[1][0][1]:
                p_min = 0
                p_max = 1
            elif temp == 0 and receiver_util[1][0][0] > receiver_util[1][0][1]:
                F2_empty = 1
            
            if not F2_empty:
                EU_R = theta * receiver_util[0][1][Aj]  + (1 - theta) * receiver_util[1][1][Aj]
                EU_S = theta * sender_util[0][1][Aj] + (1 - theta) * sender_util[1][1][Aj]
                S1 = ('M{}, M{}'.format(m, m))
                S2 = ('A{}, A{}'.format(1, Aj))
                PBE.append({'Eq':2, 'S1':S1, 'S2':S2, 'p_range':(p_min, p_max), 'q_range':(q, q),  'EU_S':EU_S, 'EU_R':EU_R})
        return PBE