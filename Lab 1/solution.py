#--IMPORTS------------------------------------------------------------------------------
from collections import deque # za BFS
import heapq                  # za UCS i A*
import sys                    # za agrumente iz konzole
import os                     # za provjeru putanja

#--GLOBALS------------------------------------------------------------------------------
check_optimistic = 1
check_consistent = 1

goal_states = []

#--UTIL---------------------------------------------------------------------------------
# Uzima "sirove" podatke pročitane iz datoteke i formatira ih u tuple(str, float) oblik 
def file_data_to_node(data : str):
    data_vect = data.split(",")
    return tuple([data_vect[0], int(data_vect[1])])

# Čita datoteku s popisom nasljednika svakog stanja i slaže ih u dict. Sortira ih abecedno.
def load_successor_map(path : str):
    global goal_states

    successor = dict()
    f = open(path, "r", encoding='utf-8')

    state_0 = ""
    line_counter = 0
    for line in f:
        if(line[0] == '#' or line == ""): # Ignoriranje komentara
            continue
        line = line.rstrip("\n") # Uklanjanje newline znakova s kraja retka
        line_counter += 1
        if(line_counter == 1):   # Na prvoj liniji se nalazi početno stanje
            state_0 = line
        elif(line_counter == 2): # Na drugoj liniji se nalaze ciljna stanja
            goal_states = [line_el for line_el in line.split()]
        else:                    # Na ostalim linijama se nalaze stanja i njihovi prijelazi, skupa sa cijenom prijelaza
            line_elements = line.split()
            source = line_elements[0].rstrip(":")
            line_elements.pop(0)
            successor.update({source : sorted([file_data_to_node(element) for element in line_elements], key=lambda x : x[0][0])})

    f.close()
    return successor, state_0

# Čita datoteku s heurističkim informacijama svakog stanja i slaže ih u dict
def load_heuristic_map(path : str):
    heuristic = dict()

    f = open(path, "r", encoding='utf-8')
    for line in f:
        if(line[0] == '#' or line == ""): # Ignoriranje komentara
            continue
        line_elements = line.split(" ")
        heuristic.update({line_elements[0].rstrip(":") : int(line_elements[1])})

    f.close()
    return dict(sorted(heuristic.items()))

# Provjerava postojanje parametra --h, postojanje putanje za datoteku s heurističkim vrijednostima stanja i vraća rezultat čitanja datoteke kao dict
def get_heuristic():
    try:
        heur_idx = sys.argv.index("--h")
    except ValueError:
        print("Missing argument --h!")
        exit(1)
    try:
        if(not os.path.exists(sys.argv[heur_idx + 1])):
            print("Invalid heuristic path!")
            exit(1)
    except IndexError:
        print("Missing path argument following --h!")
        exit(1)
    return load_heuristic_map(sys.argv[heur_idx + 1]), heur_idx

# Funkcija koja rezultate algoritama pretrage oblikuje u univerzalni format za lakši ispis  
def format_return_vals(node : tuple, parent_dict : dict, closed_list : set):
    found_node = node
    path = [node]
    while(parent_dict[node] != None):
        path.append(parent_dict[node])
        node = parent_dict[node]
    return found_node, list(closed_list), path[::-1]

#--NODE-HANDLERS------------------------------------------------------------------------
# Ovaj cijeli blok funckija služi samo za čitljiviji kod. Struktura čvora je:
#   tuple(tuple(str, float), int)
#       - str -> stanje (ime stanja)
#       - float -> cijena stanja
#       - int -> dubina stabla 

def state(node : tuple):
    return node[0][0]

def cost(node : tuple):
    return node[0][1]

def depth(node : tuple):
    return int(node[1])

def initial(state_0 : str):
    return tuple([(tuple([state_0, 0])), 0])

def expand(node : tuple, successor : dict):
    return [tuple([tuple([state[0], state[1] + cost(node)]), depth(node) + 1]) for state in successor[node[0][0]]]

def goal(node : tuple):
    return node in goal_states

def heuristic_grade(node : tuple, heuristic : dict):
    return int(heuristic[state(node)] + cost(node))

#--PATH-FINDERS-------------------------------------------------------------------------
# Funkcija za BFS algoritam
#   Korišten set umjesto list za closed zbog brzine pristupa elementima, izvor odgovor korisnika "Martijn Pieters":
#   https://stackoverflow.com/questions/48386070/faster-way-of-getting-elements-in-list-by-index
def BFS(state_0 : str, successor : dict):
    # Korištene strukture podataka
    node_0 = initial(state_0)
    open_list = deque([node_0])
    closed_list = set()
    parent_dict = {node_0 : None}

    # Glavna petlja
    while(open_list):
        node = open_list.popleft()
        closed_list.add(state(node))
        
        # Provjera je li pronađeno ciljno stanje
        if(goal(state(node))):
            return format_return_vals(node, parent_dict, closed_list)
        
        # Odbacivanje već obiđenih čvorova pri proširivanju
        for succ_node in expand(node, successor):
            if(state(succ_node) not in closed_list):
                parent_dict.update({succ_node : node})
                open_list.append(succ_node)

    # Default povrat (ciljno stanje nije pronađeno)
    return None, closed_list, None

# Zajednička funkcija za UCS i A* algoritme (spojene u jednu funkciju zbog velike sličnosti implementacija)
#   Za sortiranje po cijeni pa nazivima u heapq, prilagođen odgovor korisnika "Chau Pham": 
#   https://stackoverflow.com/questions/3954530/how-to-make-heapq-evaluate-the-heap-off-of-a-specific-attribute
#   Također korišten set, ovdje kako bi funkcija za formatiranje povratnih vrijednosti bila konzistentna
def UCS_Astar(state_0 : str, successor : dict, heuristic : dict = None):
    # Ako je heuristic None -> UCS, inače -> A*
    cost_value_function = (lambda node : cost(node)) if heuristic == None else (lambda node : heuristic_grade(node, heuristic)) 

    # Korištene strukture podataka
    node_0 = initial(state_0)
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, (cost_value_function(node_0), state(node_0), node_0))
    parent_dict = {node_0 : None}
    best_cost_for_state = {state(node_0) : cost(node_0)}

    # Glavna petlja
    while(open_list):
        _, _, node = heapq.heappop(open_list)
        closed_list.add(node)

        # Provjera je li pronađeno ciljno stanje
        if(goal(state(node))):
            return format_return_vals(node, parent_dict, closed_list)

        # Odbacivanje već obiđenih čvorova, čija cijena nije bolja od prethodno nađene, pri proširivanju
        for succ_node in expand(node, successor):
            if(state(succ_node) not in best_cost_for_state.keys() or cost(succ_node) < best_cost_for_state[state(succ_node)]):
                parent_dict.update({succ_node : node})
                best_cost_for_state.update({state(succ_node) : cost(succ_node)})
                heapq.heappush(open_list, (cost_value_function(succ_node), state(succ_node), succ_node))
    
    # Default povrat (ciljno stanje nije pronađeno)
    return None, list(best_cost_for_state.keys()), None

# Lambde za lakšu sintaksu i poziv funkcije iznad
UCS = lambda state_0, successor : UCS_Astar(state_0, successor)
Astar = lambda state_0, successor, heuristic : UCS_Astar(state_0, successor, heuristic)

#--HEURISTIC-GRADERS--------------------------------------------------------------------
# Funkcija za provjeru optimističnosti heuristike
def optimism_check(successor : dict, heuristic : dict):
    result = 1
    costs = dict() # Ovdje se spremaju do sad pronađene cijene puta
    for possible_state in heuristic.keys():
        if(possible_state not in costs.keys()): # Ako već nismo našli cijenu za trenutno stanje, provodimo UCS kako bi pronašli optimalni put 
            found_node, _, path = UCS(possible_state, successor)
            if(found_node == None):
                print("[CONDITION]: [ERR] Final state unreachable.")
                result = 0
            for node in path:
                if(state(node) not in costs.keys()): # Dodajemo samo one koje već nismo dodali
                    costs.update({state(node) : cost(path[-1]) - cost(node)})
        true_cost = costs[possible_state]
        cond = heuristic[possible_state] <= true_cost # Uvjet optimističnosti
        if(not cond):
            result = 0
        print("[CONDITION]:", "[OK]" if cond else "[ERR]", "h(" + possible_state + ")", "<= h*:", float(heuristic[possible_state]), "<=", float(true_cost))
    return result

# Funkcija za provjeru konzistentnosti heuristike
def consistency_check(successor : dict, heuristic : dict):
    result = 1
    for state in heuristic.keys():
        for succ_state in successor[state]:
            cond = heuristic[state] <= heuristic[succ_state[0]] + succ_state[1] # Uvjet konzistentnosti
            if(not cond):
                result = 0
            print("[CONDITION]:", "[OK]" if cond else "[ERR]", "h(" + state + ")", "<=", "h(" + succ_state[0] + ") + c:", float(heuristic[state]), "<=", float(heuristic[succ_state[0]]), "+", float(succ_state[1]))
    return result

#--MAIN---------------------------------------------------------------------------------
if(__name__ == "__main__"):
    argc = len(sys.argv)
    alg_idx, succ_idx, heur_idx, cost_accu = 0, 0, 0, 0.0
    heuristic, successor, state_0, found_node, closed_list, path = None, None, None, None, None, None
#------------------------------------------------------------
    # Prva provjera za --ss argument (univerzalan je za sve provjere / pozive algoritama)
    try:
        succ_idx = sys.argv.index("--ss")
    except ValueError:
        print("Missing argument --ss!")
        exit(1)
    try:
        if(not os.path.exists(sys.argv[succ_idx + 1])):
            print(sys.argv[succ_idx + 1])
            print("Invalid successor path!")
            exit(1)
    except IndexError:
        print("Missing path argument following --ss!")
        exit(1)
    successor, state_0 = load_successor_map(sys.argv[succ_idx + 1])
#------------------------------------------------------------
    # Provjera traži li se neka od provjera za heuristike
    try:
        _ = sys.argv.index("--check-optimistic")
    except ValueError:
        check_optimistic = 0

    try:
        _ = sys.argv.index("--check-consistent")
    except ValueError:
        check_consistent = 0
#------------------------------------------------------------
    # Ako su se tražile provjere, poziv za onu koja je tražena
    if(check_consistent or check_optimistic):
        heuristic, heur_idx = get_heuristic()
        print("# HEURISTIC-", "OPTIMISTIC " if check_optimistic else "CONSISTENT ", sys.argv[heur_idx + 1], sep="")
        concl = optimism_check(successor, heuristic) if(check_optimistic) else consistency_check(successor, heuristic)
        print("[CONCLUSION]: Heuristic is ", "" if concl else "not ", "optimistic." if check_optimistic else "consistent.", sep="")
#------------------------------------------------------------
    else:
        # Provjera koji algoritam se želi koristiti
        try:
            alg_idx = sys.argv.index("--alg")
        except ValueError:
            print("Missing argument --alg!")
            exit(1)
        try:
            if(sys.argv[alg_idx + 1] not in ["astar", "bfs", "ucs"]):
                print("Invalid algorithm!")
                exit(1)
        except IndexError:
            print("Missing algorithm argument following --alg!")
            exit(1)
        alg = sys.argv[alg_idx + 1]
#------------------------------------------------------------
        # Dodatno učitavanje heuristike za slučaj da je pozvan A*
        if(alg == "astar"):
            heuristic, heur_idx = get_heuristic()
#------------------------------------------------------------
        # Poziv odabranog algoritma
        if(alg == "bfs"):
            print("# BFS")
            found_node, closed_list, path = BFS(state_0, successor)
        elif(alg == "ucs"):
            print("# UCS")
            found_node, closed_list, path = UCS(state_0, successor)
        else:
            print("# A-STAR", sys.argv[succ_idx + 1])
            found_node, closed_list, path = Astar(state_0, successor=successor, heuristic=heuristic)

        # Ispisi dobivenih rezultata
        print("[FOUND_SOLUTION]:","yes" if found_node != None else "no")
        print("[STATES_VISITED]:", len(closed_list))
        if(found_node != None):
            path_str = state_0
            for i in path[1:]:
                path_str += " => " + i[0][0]
            cost_accu = path[-1][0][1]
            print("[PATH_LENGTH]:", len(path))
            print("[TOTAL_COST]: {}".format(float(cost_accu)))
            print("[PATH]:", path_str)