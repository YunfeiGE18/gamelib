import numpy as np
import nashpy as nash
import copy
import gurobipy as gp

"""
----- Input -----

    # For binary attacker's type

    Transaction matrix:   T[state][action]
    Action:   actions[type][state][action_index]
    Reward:   rewards[type][state][action_rowidx_attack_min][action_columnidx_defense_max]
    Number of state:   num_state
    Initial Belief:   b[type]
    Finite time horizon:   K

----- Output -----

    Value of the attacker:   val_a[type][state][stage]
    Value of the attacker:   val_d[state][stage]
    Belief at each stage:   b[type][state][stage]
    Defender's stratege:   pid[state][stage]
    Attacker's stratege:   pia[type][state][stage]

"""
    


def SBNE(actions, rewards, num_state,b, K):

    
    val_a = [[] for i in range(2)]
    for i in range(2):
        val_a[i] = [[] for j in range(num_state)]
        for j in range(num_state):
            val_a[i][j] = [0]*(K+1) # val_a[type][state][stage]

    val_d = [[] for i in range(num_state)]            # val_d[state][stage]
    for i in range(num_state):
        val_d[i] = [0]*(K+1)

    pid = [[] for i in range(num_state-2)]             #pid[state][stage] (list[actions])
    for i in range(num_state - 2):
        pid[i] =  [[] for j in range(K+1)]


    pia= [[] for i in range(2)]
    for i in range(2):
        pia[i] = [[] for j in range(num_state-2)]
        for j in range(num_state-2):
            pia[i][j] = [[] for j in range(K+1)] # pia[type][state][stage](list[actions])




    for sidx in range(num_state-2):
        # For defender (need to consider belief)
        alpha = [1,1]
        num_a = len(actions[0][sidx])

        try:
            m = gp.Model('LP')
            m.setParam('OutputFlag', 0)
            m.setParam('LogFile', '')
            m.setParam('LogToConsole', 0)

            va = []
            for atype in range(2):
                n = "va_type_ "+str(atype)
                va.append(m.addVar(vtype=gp.GRB.CONTINUOUS, lb=-1*gp.GRB.INFINITY, name=n))

            vd = m.addVar(name='vd', vtype=gp.GRB.CONTINUOUS, lb=-1*gp.GRB.INFINITY)

            pd = {}
            for d in range(num_a):
                pd[d] = m.addVar(lb=0.0, ub=1.0, name='pd_action_'+str(d))

            pa = {}
            for atype in range(2):
                for a in range(num_a):
                    pa[atype,a] = m.addVar(lb=0.0, ub=1.0, name='pa_type_'+str(atype)+'_action_'+str(a))

            m.update()


            m.addConstr(gp.quicksum(pd[d] for d in range(num_a)) == 1, name='c_pd')
            for atype in range(2):
                m.addConstr(gp.quicksum(pa[atype,a] for a in range(num_a)) == 1, name='c_pa_type'+str(atype))


            for atype in range(2):
                for a in range(num_a):
                    ja = gp.LinExpr()
                    for d in range(num_a):
                        ja.add(-pd[d]*rewards[atype][sidx][a][d])
                    m.addConstr(ja <= -va[atype], name='c_va_type'+str(atype)+'_action_'+str(a))


            for d in range(num_a):
                jd = gp.LinExpr()
                for atype in range(2):
                    for a in range(num_a):
                        jd.add(b[atype][sidx][K]*pa[atype,a]*rewards[atype][sidx][a][d])
                m.addConstr(jd <= -vd,name='c_vd_'+str(d))

            obj = gp.QuadExpr()
            for atype in range(2):
                jattacker = gp.QuadExpr()
                for a in range(num_a):
                    for d in range(num_a):
                        jattacker.add(alpha[atype]*(-rewards[atype][sidx][a][d]*pa[atype,a]*pd[d]))
                obj.add(va[atype] + jattacker)

            jdefender = gp.QuadExpr()
            for a in range(num_a):
                for d in range(num_a):
                    for atype in range(2):
                        jdefender.add(rewards[atype][sidx][a][d]*pa[atype,a]*pd[d]*b[atype][sidx][K])
            obj.add(jdefender)

            m.setParam('DualReductions', 0)
            m.setParam("NonConvex", 2);

            m.setObjective(obj, sense=gp.GRB.MAXIMIZE)
            m.optimize()

        except gurobierror:
            print("Error reported")

            
        # [va0,va1,vd,(pd - actions),(pa0 - actions),(pa1 - actions)]
        results = [var.x for var in m.getVars()]
        val_a[0][sidx][K] = results[0]
        val_a[1][sidx][K] = results[1]
        val_d[sidx][K] = results[2]
        pid[sidx][K] = results[3:3+num_a]
        pia[0][sidx][K] = results[3+num_a:3+num_a*2]
        pia[1][sidx][K] = results[3+num_a*2:]
        
        
    return val_a, val_d, pid, pia




def DBNE(actions, rewards, T, num_state, K, val_a, val_d, pid, pia, b):
    

    for k in range(K):
#         print('------')
#         print(k)
#         print('------')
        for sidx in range(num_state-2):
#             print(sidx)
            # For defender (need to consider belief)
            alpha = [1,1]
            num_a = len(actions[0][sidx])

            try:
                m = gp.Model('LP')
                m.setParam('OutputFlag', 0)
                m.setParam('LogFile', '')
                m.setParam('LogToConsole', 0)

                va = []
                for atype in range(2):
                    n = "va_type_ "+str(atype)
                    va.append(m.addVar(vtype=gp.GRB.CONTINUOUS, lb=-1*gp.GRB.INFINITY, name=n))

                vd = m.addVar(name='vd', vtype=gp.GRB.CONTINUOUS, lb=-1*gp.GRB.INFINITY)

                pd = {}
                for d in range(num_a):
                    pd[d] = m.addVar(lb=0.0, ub=1.0, name='pd_action_'+str(d))

                pa = {}
                for atype in range(2):
                    for a in range(num_a):
                        pa[atype,a] = m.addVar(lb=0.0, ub=1.0, name='pa_type_'+str(atype)+'_action_'+str(a))

                m.update()


                m.addConstr(gp.quicksum(pd[d] for d in range(num_a)) == 1, name='c_pd')
                for atype in range(2):
                    m.addConstr(gp.quicksum(pa[atype,a] for a in range(num_a)) == 1, name='c_pa_type'+str(atype))


                for atype in range(2):
                    for a in range(num_a):
                        ja = gp.LinExpr()
                        for d in range(num_a):
                            if a == d:
                                sidx_next = sidx
                            else:
                                sidx_next = T[sidx][a]
                            ja.add(pd[d]*(-rewards[atype][sidx][a][d]+val_a[atype][sidx_next][0]))
                        m.addConstr(ja <= -va[atype], name='c_va_type'+str(atype)+'_action_'+str(a))


                for d in range(num_a):
                    jd = gp.LinExpr()
                    for atype in range(2):
                        for a in range(num_a):
                            if a == d:
                                sidx_next = sidx
                            else:
                                sidx_next = T[sidx][a]
                            jd.add(b[atype][sidx][K-k-1]*pa[atype,a]*(rewards[atype][sidx][a][d]+val_d[sidx_next][0]))
                    m.addConstr(jd <= -vd,name='c_vd_'+str(d))

                obj = gp.QuadExpr()
                for atype in range(2):
                    jattacker = gp.QuadExpr()
                    for a in range(num_a):
                        for d in range(num_a):
                            if a == d:
                                sidx_next = sidx
                            else:
                                sidx_next = T[sidx][a]
                            jattacker.add(alpha[atype]*(pa[atype,a]*pd[d]*(-rewards[atype][sidx][a][d]+val_a[atype][sidx_next][0])))
                    obj.add(va[atype] + jattacker)

                jdefender = gp.QuadExpr()
                for a in range(num_a):
                    for d in range(num_a):
                        if a == d:
                            sidx_next = sidx
                        else:
                            sidx_next = T[sidx][a]
                        for atype in range(2):
                            jdefender.add(pa[atype,a]*pd[d]*b[atype][sidx][K-k-1]*(rewards[atype][sidx][a][d]+val_d[sidx_next][0]))
                obj.add(jdefender)

                m.setParam('DualReductions', 0)
                m.setParam("NonConvex", 2);

                m.setObjective(obj, sense=gp.GRB.MAXIMIZE)
                m.optimize()

            except gurobierror:
                print("Error reported")


            # [va0,va1,vd,(pd - actions),(pa0 - actions),(pa1 - actions)]
            results = [var.x for var in m.getVars()]
            val_a[0][sidx][K-k-1] = results[0]
            val_a[1][sidx][K-k-1] = results[1]
            val_d[sidx][K-k-1] = results[2]
            pid[sidx][K-k-1] = results[3:3+num_a]
            pia[0][sidx][K-k-1] = results[3+num_a:3+num_a*2]
            pia[1][sidx][K-k-1] = results[3+num_a*2:]
        
    return val_a, val_d, pid, pia





def belief_update(b, actions, T, pid, pia, num_state, K):
    
    
    for sk in range(num_state-2):              #sk
        num_a = len(actions[0][sk])

        for k1 in range(1,K+1):                 # k+1

            for sk1 in range(num_state-2):     # s^k+1

                # compute denominator
                diff_theta_sum = [0]*2
                for theta in range(2):         
                    ad_sum = 0
                    for a in range(num_a):     # a
                        for d in range(num_a): # d
                            if a == d:         # T
                                sk_next = sk
                            else:
                                sk_next = T[sk][a]

                            if sk_next != sk1:
                                continue
                            else:
                                diff_theta_sum[theta] += pid[sk][k1-1][d]*pia[theta][sk][k1-1][a]*b[theta][sk][k1-1]

                adthe_sum = sum(diff_theta_sum)

                # belief update
                for theta in range(2):        # theta
                    if adthe_sum == 0:
                        b[theta][sk1][k1] = b[theta][sk1][k1]
                    else:
                        b[theta][sk1][k1] = diff_theta_sum[theta]/adthe_sum
                        
    return b