#--IMPORTS------------------------------------------------------------------------------
import sys                    # za agrumente iz konzole

#--GLOBALS------------------------------------------------------------------------------
cook = 1
res = 1

clause_counter = 0

goal_string = ""

premises_to_skip= []
clauses_used = []

#--UTIL---------------------------------------------------------------------------------
# Pretvara liniju (string klauzule) u listu literala (implicirana disjunkcija)
def process_line(line : str):
    return [node.strip() for node in line.split(" v ")]

# Faktorizira klauzulu (uklanja duplikatne literale)
def factorize(clause : list):
    return list(dict.fromkeys(clause))

# Negira literal
def negate(elem : str):
    return "~" + elem if elem[0] != "~" else elem[1:]

# Vraća formatirani string klauzule
def clause_string(clause : list):
    cum_str = clause[0]
    for elem in clause[1:]:
        cum_str += " v " + elem
    return cum_str

# Provjera za subsumaciju jedne klauzule drugom
def is_subsumed(clause : list, clauses : set):
    clause_set = set(clause)

    for _clause in clauses:
        _clause = set(_clause)
        if(_clause.issubset(clause_set)):
            return 1
    return 0

# Provjera redundancije klauzule
def is_redundant(clause : list, clauses : set):
    return tuple(clause) in clauses or is_subsumed(clause, clauses)

# Provjera je li klauzula tautologija
def is_tautology(clause: list):
    return any(negate(node) in clause for node in clause)

# Učitavanje klauzula iz datoteke
def read_clauses(path : str):
    global goal_string

    clauses = []

    f = open(path, "r", encoding="utf-8")
    for line in f:
        line = line.lower().strip()
        if(line[0] == '#' or line == ""):
            continue
        goal_string = line
        clauses.append(process_line(line))
    f.close()

    return clauses

# Učitavanje komandi iz datoteke
def read_commands(path : str):
    commands = []

    f = open(path, "r", encoding="utf-8")
    for line in f:
        line = line.lower().strip()
        clause, op = process_line(line[:-2]), line[-1]
        commands.append(tuple([clause, op]))
    f.close()

    return commands

# Formatiranje klauzula za rezoluciju (odvajanje cilja od premisa)
def clauses_resolution(clauses : list):
    clauses[-1] = [[negate(clause)] for clause in clauses[-1]]
    return [clause for clause in clauses[:-1] if not is_tautology(clause)], clauses[-1]

# Stvaranje nove klauzule, kombinirajući dvije poznate
def resolve(clause_1 : list, clause_2 : list):
    result = factorize(clause_1 + clause_2)
    for i in range(len(result)):                
        for j in range(len(result)):            
            if(result[i] == negate(result[j])): # Uklanjanje razrješenih klauzula
                result[i] = result[j] = "-"
                break
    return [r for r in result if r != "-"]

#--STRATEGY-----------------------------------------------------------------------------
# Implementacija SoS strategije
def strategy(clause_list : list, goal : list):
    global premises_to_skip

    SoS = [goal_state for goal_state in goal]

    clause_list += goal
    
    clauses = dict()
    for i in clause_list:
        clauses.update({tuple(i) : None})

    premises_to_skip = clause_list[:]

    while(SoS):
        clause = SoS.pop()
        for known_clause in clause_list:
            if(clause == known_clause):
                continue
            
            # Određivanje novih klauzula
            new_clause = resolve(clause, known_clause)
            
            # Ako smo našli praznu klauzulu, znači da smo naišli na proturječje -> cilj je dokazan
            if(new_clause == []):
                clauses.update({"NIL" : [known_clause, clause]})
                if(tuple(clause) not in clause_list and tuple(clause) not in clauses_used):
                    clauses_used.append(tuple(clause))
                return clauses, 1
            
            # Odbacivanje nepotrebnih klauzula
            if(is_tautology(new_clause) or is_redundant(new_clause, clauses.keys())):
                continue

            # Dodavanje novih klauzula
            if(tuple(new_clause) not in clauses.keys()):
                clauses.update({tuple(new_clause) : [known_clause, clause]})
                if(tuple(clause) not in clause_list and tuple(clause) not in clauses_used): # Pamtimo samo one klauzule koje su korištene
                    clauses_used.append(tuple(clause))
                SoS.append(new_clause)

    return clauses, 0

# Formatiranje rezultata dobivenih SoS strategijom
def process_clauses(clauses : list, goals : list):
    global clause_counter, goal_string, premises_to_skip, clauses_used

    clauses = [factorize(clause) for clause in clauses]
    goals = [factorize(goal) for goal in goals]

    clause_map, concl = strategy(clauses, goals)
    index_map = dict()

    # Uklanjamo premise koje nismo koristili
    for key, values in clause_map.items():
        if(values != None and key in clauses_used):
            for value in values:
                if(value in clauses and value in premises_to_skip):
                    premises_to_skip.remove(value)
    if(concl):
        for value in clause_map["NIL"]:
            if(value in clauses and value in premises_to_skip):
                premises_to_skip.remove(value)

    # Numeriranje korištenih klauzula
    clause_counter = 0
    for clause in clauses:
        if(clause not in premises_to_skip):
            clause_counter += 1
            print(str(clause_counter) + ". " + clause_string(clause))
            index_map.update({tuple(clause) : clause_counter})
    print("===============")
    for clause in clauses_used:
        if(clause_map[tuple(clause)] != None):
            clause_counter += 1
            print(str(clause_counter) + ". " + clause_string(clause) + " (" + str(index_map[tuple(clause_map[tuple(clause)][0])]) + ", " + str(index_map[tuple(clause_map[tuple(clause)][1])])+ ")")
            index_map.update({tuple(clause) : clause_counter})
    if(concl):
        print(str(clause_counter + 1) + ". NIL " + " (" + str(index_map[tuple(clause_map["NIL"][0])]) + ", " + str(index_map[tuple(clause_map["NIL"][1])])+ ")")

    # Ispis konkluzije
    print("===============\n[CONCLUSION]:", goal_string, "is", "true" if concl else "unknown")

    return 0

# Driver funkcija za opciju "resolution"
def resolution(res_idx : int):
    clauses, goals = clauses_resolution(read_clauses(sys.argv[res_idx + 1]))
    return process_clauses(clauses, goals)

# Driver funkcija za opciju "cooking"
def cooking(cook_idx : int):
    global goal_string, premises_to_skip, clauses_used

    clauses = read_clauses(sys.argv[cook_idx + 1])
    commands = read_commands(sys.argv[cook_idx + 2])

    print("Constructed with knowledge:")
    for clause in clauses:
        print(clause_string(clause))

    for command in commands:
        print("\nUser's command:", clause_string(command[0]), command[1])
        # Komanda za dodavanje premisa
        if(command[1] == "+"):
            if(command[0] not in clauses):
                clauses.append(command[0])
        # Komanda za uklanjanje premisa
        elif(command[1] == "-"):
            clauses.remove(command[0])
        # Komanda za provjeru vrijednosti
        else:
            clauses_used.clear()
            premises_to_skip.clear()
            goal_string = clause_string(command[0])
            goals = [[negate(node)] for node in command[0]]
            print(clauses, "\n", goals)
            process_clauses(clauses[:], goals)
    return 0

#--MAIN---------------------------------------------------------------------------------
if(__name__ == "__main__"):
    argc = len(sys.argv)
    res_idx, cook_idx = None, None

    try:
        res_idx = sys.argv.index("resolution")
    except ValueError:
        res = 0

    try:
        cook_idx = sys.argv.index("cooking")
    except ValueError:
        cook = 0

    if(not res and not cook or res and argc < 3 or cook and argc < 4):
        print("Missing arguments!")
        exit(1)

    if(res):
        resolution(res_idx)
    else:
        cooking(cook_idx)