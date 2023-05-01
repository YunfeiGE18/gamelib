from gurobipy import *
import sys

def BSG(filename):
    try:
        # Create a new model
        m = Model("MIQP")

        f = open(filename, 'r')
            ## f.readline() read one line by one line
        """
        ------ Input file ------
        No. of message (X)
        No. of receivers (L)
        | Probability for receiver with type l (p_l)
        | No. of receiver's actions with type l (Q_l)
        | [
        |       Matrix ( X * Q_l) with
        |       values r, c (sender / receiver)
        | ]
        | where r,c are rewards for defender and attacker respectively
        | First: user / Second: attacker
        """

        # Add defender stategies to the model
        X = int(f.readline())
        x = []
        for i in range(X):
            n = "message-" + str(i)
            x.append(m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=n))
        m.update()

        # Add defender stategy constraints
        con = LinExpr()
        for i in range(X):
            con.add(x[i])
        m.addConstr(con == 1)

        """ Start processing for attacker types """

        L = int(f.readline())
        obj = QuadExpr()
        M = 100000000

        for l in range(L):

            # Probability of l-th attacker
            v = f.readline().strip()
            print()
            p = float(v)

            # Add l-th attacker info to the model
            Q = int(f.readline())
            q = []
            cve_names = f.readline().strip().split("|")
            for i in range(Q):
                n = "Attacker "+ str(l) + "-" + cve_names[i]
                q.append(m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name=n))

            a = m.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="value for attacker-" + str(l)
            )

            m.update()

            # Get reward for attacker and defender
            R = []
            C = []
            for i in range(X):
                rewards = f.readline().split()
                r = []
                c = []
                for j in range(Q):
                    r_and_c = rewards[j].split(",")
                    r.append(r_and_c[0])
                    c.append(r_and_c[1])
                R.append(r)
                C.append(c)

            # Update objective function
            for i in range(X):
                for j in range(Q):
                    r = p * float(R[i][j])
                    obj.add(r * x[i] * q[j])

            # Add constraints to make attaker have a pure strategy
            con = LinExpr()
            for j in range(Q):
                con.add(q[j])
            m.addConstr(con == 1)

            # Add constrains to make attacker select dominant pure strategy
            for j in range(Q):
                val = LinExpr()
                val.add(a)
                for i in range(X):
                    val.add(float(C[i][j]) * x[i], -1.0)
                m.addConstr(val >= 0, q[j].getAttr("VarName") + "lb")
                m.addConstr(val <= (1 - q[j]) * M, q[j].getAttr("VarName") + "ub")

        # Set objective funcion as all attackers have now been considered
        m.setObjective(obj, GRB.MAXIMIZE)

        # Solve MIQP
        m.optimize()

        # Print out values
        def printSeperator():
            print("---------------")

        printSeperator()
        for v in m.getVars():
            print("%s -> %g" % (v.varName, v.x))

        printSeperator()
        print("Obj -> %g" % m.objVal)
        printSeperator()

        # Prints constrains
        # printSeperator()
        # for c in m.getConstrs():
        #    if c.Slack == 0.0:
        #        print(str(c.ConstrName) + ': ' + str(c.Slack))
        # printSeperator()
    except GurobiError:
        print("Error reported")