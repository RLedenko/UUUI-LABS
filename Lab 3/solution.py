# IMPORTS
import math
import sys

# CONSTANTS
MODE_LEAF = 0
MODE_NODE = 1

# GLOBALS VARS
V = dict()
varname_convert = dict()
default_val = ""

# Klasa node (čvor) služi za predstavljanje stabla koje ID3 algoritam generira
class Node:
    # Konstruktor
    def __init__(self, label : str, value : dict | str, mode : int):
        self.label = label # Labela -> ime čvora
        self.value = value # Vrijednost -> krajnja vrijednost (mapa čvorova ili listna vrijednost)
        self.mode = mode   # Način rada -> list ili čvor
    
    # Print funkcija
    # -poziva rekurzivni ispis grana stabla
    def print_tree(self):
        self.print_tree_i(1, "")

    # Evaluacijska funkcija za ulaz iz testnog skupa
    def evaluate(self, varnames : list, data : list):
        # Listovi vraćaju svoju vrijednost
        if(self.mode == MODE_LEAF):
            return self.value
        
        # Inače tražimo sljedeću granu
        elem = data[varnames.index(self.label)]

        # Ako varijable nema, vraćaj najčešću labelu
        if(elem not in self.value.keys()):
            return default_val

        # Rekurzivni poziv
        return self.value[elem].evaluate(varnames, data)

    # Rekurzivni korak ispisa
    def print_tree_i(self, depth : int, accustr : str):
        # List -> kraj, ispiši akumulirani string
        if(self.mode == MODE_LEAF):
            print(accustr + " " + self.value)
            return

        basestr = str(depth) + ":" + self.label + "="
        if(not depth == 1):
            basestr = accustr + " " + basestr
        
        for k, v in self.value.items():
            str_key = basestr + k
            v.print_tree_i(depth + 1, str_key)

# Pretvori zapis iz datoteke u iskoristivu strukturu podataka
def parse_to_datapoint(raw : str):
    data = raw.split(",")
    return tuple([data[:-1], data[-1]])

# Izračun skupa V -> svih varijabli za određenu labelu
def count_all(data : list):
    all_vars = dict()

    # Setup mape za podatke
    for i in range(len(data[0][0])):
        all_vars.update({i : []})
    all_vars.update({"y" : []})

    # Izvlačenje svih varijabli iz podataka i sortiranje po labelama
    for datapoint in data:
        i = 0
        for elem in datapoint[0]:
            _t = all_vars[i]
            if elem not in _t:
                _t.append(elem)
                all_vars.update({i : _t})
            i += 1
        _t = all_vars["y"]
        if datapoint[1] not in _t:
            _t.append(datapoint[1])
            all_vars.update({"y" : _t})
    
    all_vars["y"].sort(key=lambda x : x)

    return all_vars

# Generiranje skupa D iz učitanih podataka, i izvlačenje svih labela
def generate_D(path : str):
    D, varnames = [], []
    with open(path) as f:
        varnames = f.readline().strip().split(",")
        i = 0
        for var in varnames:
            varname_convert.update({i : var})
            i += 1
        for line in f:
            D.append(parse_to_datapoint(line.strip()))
    return D, varnames

# Izračun najčešće labele
def decide_default(D):
    global default_val

    all_elems = [k[1] for k in D]
    elem_ctr = dict()
    for j in all_elems:
        elem_ctr[j] = elem_ctr.get(j, 0) + 1
    
    maxval = max(elem_ctr.values())
    default_val = sorted([k for k in elem_ctr if elem_ctr[k] == maxval], key=lambda x : x)[0]

# Implementacija formule za entropiju
def entropy(D):
    ctr = dict()
    for _, label in D:
        ctr[label] = ctr.get(label, 0) + 1
    n = len(D)
    calc_entropy = 0.
    for label in ctr:
        p = ctr[label] / n
        calc_entropy -= p * math.log2(p)
    return calc_entropy

# Implementacija formule za informacijsku dobit
def IG(D : list, idx : int):
    global varname_convert

    D_entropy = entropy(D)
    
    vals = dict()
    for X, y in D:
        key = X[idx]
        if key not in vals:
            vals.update({key : []})
        vals[key].append((X, y))

    sum_entropies = 0.

    n = len(D)
    for subset in vals.values():
        p = len(subset) / n
        sum_entropies += p * entropy(subset)
   
    return D_entropy - sum_entropies

# Argmax funkcija za skup D 
def argmax_D(data : list):
    ctr = dict()
    for _, label in data:
        ctr[label] = ctr.get(label, 0) + 1
    
    maxval = max(ctr.values())
    return sorted([k for k in ctr if ctr[k] == maxval], key=lambda x : x)[0]

# Argmax funkcija za argument s najvećim IG iz D
def argmax(D : list, X : list):
    global varname_convert

    IGs = [(i, IG(D, i)) for i in X]
    IGs.sort(key=lambda x : (-x[1], varname_convert[x[0]]))
    
    for idx, ig in IGs:
        print(f"IG({varname_convert[idx]})={ig:.4f}", end=" ")
    print()

    return IGs[0][0]

# Bool funkcija za provjeru jesu li sve labele u D jednake
def all_v(D : list, v : str):
    for i in D:
        if not i[1] == v:
            return False
    return True

# Implementacija ID3 algoritma s opcionalnom maksimalnom dubinom
def id3(D, Dp, X, y, depth=0, max_depth=None):
    global V

    if(max_depth is not None and depth == max_depth):
        return Node(None, argmax_D(D if D else Dp), MODE_LEAF)

    if not D:
        v = argmax_D(Dp)
        return Node(None, v, MODE_LEAF)
    v = argmax_D(D)
    if not X or all_v(D, v):
        return Node(None, v, MODE_LEAF)
    
    x = argmax(D, X)

    subtrees = dict()
    for v in V[x]:
        t = id3([k for k in D if v == k[0][x]], D, [k for k in X if not(k == x)], y, depth + 1, max_depth)
        subtrees.update({v : t})
    
    return Node(varname_convert[x], subtrees, MODE_NODE)

# Klasa kao omotač za implementaciju ID3 algoritma
class ID3:
    # Konstruktor koji učitava hiperparametre
    def __init__(self, params=None):
        self.params = params
        self.tree = None

    # Funkcija za treniranje na skupu učitanom iz datoteke
    def fit(self, dataset : str):
        global V

        D, varnames = generate_D(dataset)
        V = count_all(D)
        
        self.tree = id3(D, D, [x for x in range(len(varnames) - 1)], V["y"], 0, self.params)

        decide_default(D)

        print("[BRANCHES]:")
        self.tree.print_tree()

    # Funkcija za testiranje na skupu učitanom iz datoteke
    def predict(self, dataset : str):
        D_test, varnames = generate_D(dataset)
        ctr = 0.
        Y = len(V["y"])
        conf_matr = [[0 for j in range(Y)] for i in range(Y)]
        print("[PREDICTIONS]:", end="")
        for d in D_test:
            pred = self.tree.evaluate(varnames, d[0])
            if(pred == d[1]):
                ctr += 1
            print("", pred, end="")
            conf_matr[V["y"].index(d[1])][V["y"].index(pred)] += 1
        print()
        return ctr / len(D_test), conf_matr

# Main
if __name__ == "__main__":
    path = sys.argv[1]
    t_path = sys.argv[2]

    param = int(sys.argv[3]) if len(sys.argv) > 3 else None
    id3_inst = ID3(param)

    id3_inst.fit(path)
    accuracy, conf_matr = id3_inst.predict(t_path)
    
    print(f"[ACCURACY]: {accuracy:.5f}")
    print("[CONFUSION_MATRIX]:")
    for i in conf_matr:
        for j in i:
            print(j, end=" ")
        print()