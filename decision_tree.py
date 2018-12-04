import numpy as np
import itertools
import copy
import os.path
import os

def popcount (x):
  return bin(x).count("1")

def get_popindex (x):
  y = x
  i = 0
  while True:
    if y == 1:
      return i
    i += 1
    y = y >> 1 

def generate_dt (x, n) :
  nbr_combination = 2 ** len (n) 

  if nbr_combination == 1:
    a = [['no action'], ['new label']]
    o = range (2**len(x))
    o = np.array (o)
    e = np.zeros (((2**len(x)), 2))
    e[0,0] = 1
    for i in range((2**len(x))-1):
      e[i+1,1] = 1
    return a, e, o

  a = [['no action']]
  p = range (nbr_combination)
  for i in p:
    j = popcount(i)
    if j > 1:
      z = i
      b = []
      for k in range (j):
        y = get_popindex (z)
        b.append (len(n)-y-1)
        z = z - 2**y
      a.append (['merge']+list(b))
    elif j == 1:
      a.append (['assign']+[len(n)-get_popindex(i)-1])
    else:
      a.append (['new label'])

  o = range ((2**len(x))*nbr_combination)
  o = np.array (o)

  f = np.identity(nbr_combination)  
  e = np.zeros ((nbr_combination*(2**len(x)), nbr_combination+1))
  for i in range (nbr_combination):
    e[i,0] = 1
  for i in range(2**len(x)-1):
    e[(i+1)*nbr_combination:(i+2)*nbr_combination,1:] = np.copy(f)

  return a, e, o

def connected(s, x, y):
  for z in s:
    _x = [x[0]+z[0], x[1]+z[1], x[2]+z[2]]
    if _x[0] == y[0] and _x[1] == y[1] and _x[2] == y[2]:
      return True
  return False

def extract_connected (s, v):
  label = [0]*len(v)
  count = 1
  q = list()
  for i in range(len(v)):
    if (label[i] == 0):
      label[i] = count
      count += 1
    q.append (v[i])
    while len(q) != 0:
      x = q.pop()
      for j in range(len(v))[i+1:]:
        y = v[j]
        for z in s:
          _x = [x[0]+z[0], x[1]+z[1], x[2]+z[2]]
          if _x[0] == y[0] and _x[1] == y[1] and _x[2] == y[2] and label[j] == 0:
            label[j] = label[i]
            q.append (y)
  return label

def get_vec_coor_from_bin (b, s):
  a = b
  v = []
  j = popcount(a)
  for c in range (j):
    i = get_popindex (a)
    v.append (s[len(s)-i-1])
    a = a - 2**i
  return v

def correction (s, x, n, a, e, o):
  for i in o[2**len(n):]:
    # Keeping only the bits corresponding to the neighbors.
    b1 = i - (i & ~(2**len(n)-1))
    # Those are the x bits.
    b2 = i>>len(n)
    v1 = get_vec_coor_from_bin (b1, n)
    v2 = get_vec_coor_from_bin (b2, x)
    v = v1+v2    
    l = extract_connected (s, v)
    # Replace assign by new label.
    if len (v1) == 1 and max(l) != 1:
      e[o.tolist().index(i)] = 0
      e[o.tolist().index(i)][1] = 1
    # Replace merge by assign or merge.
    if len (v1) > 1:
      # Always only one label on v2
      label_to_find = max (l[len(v1):])
      indices = [tmp1 for tmp1, tmp2 in enumerate(l[0:len(v1)]) if tmp2 == label_to_find]
      # Assign.
      if len(indices) == 1:
        assigning_to = n.index(v1[indices[0]])
        e[o.tolist().index(i)] = 0
        e[o.tolist().index(i)][(1 << (len(n)-assigning_to-1)) + 1] = 1
      # Merges.
      else :
        e[o.tolist().index(i)] = 0
        tmp = []
        for tmp2 in indices:
          tmp.append (n.index(v1[tmp2]))
        tmp = sorted (tmp)
        index = 0
        for tmp2 in tmp:
          index = index + (1 << (len(n)-tmp2-1))
        e[o.tolist().index(i)][index + 1] = 1
  return a, e, o

def compress (s, n, e, o, i, v):
  l = extract_connected (s, v)
  if max(l) == 1:
    # Replacing by equivalent assigns
    e[o.tolist().index(i)] = 0
    for x in v:
      assigning_to = n.index (x)
      e[o.tolist().index(i)][(1 << (len(n)-assigning_to-1)) + 1] = 1
  elif max(l) > 1:
    # Replacing by merges of size the number of components.
    e[o.tolist().index(i)] = 0
    assigns = [[] for k in range(max(l))]
    for j,x in enumerate(v):
      assigns[l[j]-1].append (n.index(x))
    for a in itertools.product(*assigns):
      a = sorted (a)
      index = 0
      for x in a:
        index = index + (1 << (len(n)-x-1))
      e[o.tolist().index(i)][index + 1] = 1
  return e

def merge_connected (s, x, n, a, e, o):
  for i in o[2**len(n):]:
    # Keeping only the bits corresponding to the neighbors.
    b1 = e[o.tolist().index(i)].tolist().index(1) -1
    v1 = []
    if b1 != -1:
      v1 = get_vec_coor_from_bin (b1, n)

    if v1 != []:
      e = compress (s, n, e, o, i, v1)
  return a, e, o

def compress_dt (s, x, n, a, e, o) :
  if len(n) == 0:
    return a, e, o
  a, e, o = merge_connected (s, x, n, a, e, o)
  return a, e, o

def remove_rows (a,e):
  # Now we have multiple equivalent actions to replace the most expensive ones.
  # We can remove all columns that are empty and remove the corresponding actions.

  mask = e == 0
  # Getting the indices of zero valued column
  indices = np.flatnonzero ((~mask).sum(axis=0) == 0)
  e_tmp = np.delete (e, indices, 1)

  for i in reversed (sorted (indices)):
    del a[i]

  return a, e_tmp

# Does not allow optimal trees to be generated.
def action_selection (e) :
  count = e.sum(axis=0)
  for row in e:
    tmp = row * count
    found_it = False
    for i in range(len (tmp)):
      if tmp[i] != np.max(tmp) or found_it:
        tmp[i]=0
      if not found_it and tmp[i] == np.max(tmp):
        found_it = True
    row[tmp==0] = 0
  return e

class iter_ncube:
  def __init__ (self, t, n):
    self.nbr = 0 
    self.iter_dash = 0

    self.t = t
    self.n = n
    self.string_tild = list (bin(sum(subset))[2:].zfill(n) for subset in itertools.combinations ((2**s for s in range(n)), t))
  def __iter__ (self):
    return self
  def next (self):
    if self.iter_dash >= len (self.string_tild):
      raise StopIteration
      self.nbr = 0
      self.iter_dash += 1
    else :
      b1 = map(int, self.string_tild[self.iter_dash])
      b2 = map (int, bin(self.nbr)[2:].zfill(self.n-self.t))      
      j=0
      for i,b in enumerate (b1):
        if b == 0:
          b1[i] = b2[j]
          j+=1
        else:
          b1[i] = '-'
      self.nbr += 1
      if self.nbr >= 2**(self.n-self.t) :
        self.nbr = 0
        self.iter_dash += 1

      return b1

def compute_gain (K, t, R, a):
  # Initialize actions to the set of actions + -1 to test a pixel
  actions = set (range(len(a))) 
  gains = 0

  # Selecting the best combinations to generate the t-cube.
  i_star = 0
  # Initialised as the first occurence of '-'
  while (K[i_star] != '-') :
    i_star+=1
  i = i_star

  for j in range (t):
    while (K[i] != '-') :
      i+=1
    K_1 = list(K)
    K_0 = list(K)
    K_1[i] = 1
    K_0[i] = 0
    action = set(R[t-1][''.join(map(str,K_1))][0]) & set(R[t-1][''.join(map(str,K_0))][0])
    gain = R[t-1][''.join(map(str,K_1))][1] + R[t-1][''.join(map(str,K_0))][1] + int(len(action) != 0)
    if gain > gains:
      i_star = i
    gains = max (gain, gains)
    actions = action & actions
    i+=1
  return actions, gains, i_star


def compute_gains (a, e, o, x):
  n = len (bin(o[len(o)-1])[2:])

  R = []
  # Constructing 0-cubes
  R.append ({})
  for i in range (0,len(o)):
    R[0][bin(o[i])[2:].zfill(n)] = [tuple(np.flatnonzero(e[i]).tolist()), 0, 0]

  for t in  range(1,n+1):
    # New level of t-cubes 
    R.append ({})
    for K in iter_ncube(t, n):
      actions, gains, i_star = compute_gain (K, t, R, a)
      R[t][''.join(map(str,K))] = [tuple(actions), gains, i_star]
  return R

# Defining a simple tree structure
class node (object):
  def __init__ (self, data):
    self.data = data
    self.left = None
    self.right = None

def build_tree (n, rule, level, R, a, x):
  if len(R[level][rule][0]) != 0:
    # We take the first action of the set.
    n = node(a[R[level][rule][0][0]])
  else:
    index = R[level][rule][2]
    if index < len(x):
#      n = node(["test"]+[len(x)-index-1])
      n = node(["test"]+[index])
    else:
#      n = node(["test2"]+[len(rule)-index-1]) 
      n = node(["test2"]+[index-len(x)])
    rule = list(rule)
    rule[index] = '0'
    n.left = build_tree (n.left, ''.join(rule), level-1, R, a, x)
    rule[index] = '1'
    n.right = build_tree (n.right, ''.join(rule), level-1, R, a, x)
    rule[index] = '-'
    rule = ''.join(rule)

  return n

def print_tree (root):
  print root.data
  if (root.right != None):
    print "RIGHT"
    print_tree (root.right)
  if root.left != None:
    print "LEFT"
    print_tree (root.left)

def get_char_subtree_from_str (p, x): 
  names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v', 'w', 'x', 'y', 'z', 'a2', 'b2', 'c2']
  names2 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

  char_subtree = ""
  for i,y in enumerate(p):
    if y == '1':
      if i < len(x):
        char_subtree += names2[i]
      else:
        char_subtree += names[i-len(x)]
  char_subtree += '_'
  for i,y in enumerate(p):
    if y == '0':
      if i < len(x):
        char_subtree += names2[i]
      else:
        char_subtree += names[i-len(x)]
  return char_subtree

def get_nodes_and_states_from_str (p,x):
  names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v', 'w', 'x', 'y', 'z', 'a2', 'b2', 'c2']
  names2 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
  nodes = []
  states = []
  for i,j in enumerate(list(p)[:len(x)]):
    if j != '-':
      nodes += [names2[i]] 
      if j == '1':
        states += [1]
      else:
        states += [0]
  for i,j in enumerate(list(p)[len(x):]):
    if j != '-':
      nodes += [names[i]] 
      if j == '1':
        states += [1]
      else:
        states += [0]
  return nodes, states

def get_str_from_nodes_and_states (nodes, states, x, n):
  p_out = '-'*(len(x)+len(n))
  names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v', 'w', 'x', 'y', 'z', 'a2', 'b2', 'c2']
  names2 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

  for i,j in enumerate(nodes):
    tmp = list(p_out)
    if states[i] == 1:
      if j in names:
        tmp[len(x)+names.index(j)] = '1'
      else:
        tmp[names2.index(j)] = '1'
    else:
      if j in names:
        tmp[len(x)+names.index(j)] = '0'
      else:
        tmp[names2.index(j)] = '0'
    p_out = ''.join(tmp)
  return p_out


def predict_next_subtree (checked_nodes, checked_states, p, x, n, predictable, next_predict, out_predictions, bound, name):

  # From the top-down search in the current binary tree.
  a = checked_nodes
  r = checked_states

  # From the boundary. Those are always valued 0. Invariant by c-translation
  c, t = get_nodes_and_states_from_str (bound, x)

  # From previous prediction.
  b, s = get_nodes_and_states_from_str (p, x)

  # a,b are from current position. We need to translate but first lemme merge without introducing doublons.
  for i,j in enumerate (b):
    if j not in a:
      a += [b[i]]
      r += [s[i]]
  # First, we get the indexes of items that are not predictable. 
  index = []
  for i,j in enumerate(a):
    if j not in predictable:
      index += [i]
  # Then, we eliminate...
  a = [i for j, i in enumerate(a) if j not in index]
  r = [i for j, i in enumerate(r) if j not in index]
  # Finally, we apply the change of name from raster scan.
  a = [next_predict[i] for j, i in enumerate(a)]

  # We can now add the invariant nodes (without introducing repetitions).
  for i,j in enumerate (c):
    if j not in a:
      a += [c[i]]
      r += [t[i]]

  # Here we generate the prediction str.
  out_predictions += [get_str_from_nodes_and_states(a,r,x,n)]

  # Now we generate the flag.
  names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v', 'w', 'x', 'y', 'z', 'a2', 'b2', 'c2']
  names2 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
  indices = [(names2+names).index(x) for x in a]
  checked_nodes = [x for y,x in sorted(zip(indices, a))]
  checked_states = [x for y,x in sorted(zip(indices, r))]

  output = "FLAG_"+name+"_"
  for i,y in enumerate(checked_nodes):
    if checked_states[i] == 1:
      output += y
  output += '_'
  for i,y in enumerate(checked_nodes):
    if checked_states[i] == 0:
      output += y

  return output, out_predictions 

def write_node (x, t, n, checked_nodes, checked_states, p, file_out, indent_level, predictable, next_predict, bound, name, out_predictions):
  names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v', 'w', 'x', 'y', 'z', 'a2', 'b2', 'c2']
  names2 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

  if t.data[0] == 'test':
    file_out.write (' '*indent_level*2 + "if (CONDITION_" + names2[t.data[1]] + ") {\\\n")
    out_predictions = write_node (x, t.right, n, checked_nodes+[names2[t.data[1]]], checked_states+[1], p, file_out, indent_level+1, predictable, next_predict, bound, name, out_predictions)
    file_out.write (' '*indent_level*2 + "} else {\\\n")
    out_predictions = write_node (x, t.left, n, checked_nodes+[names2[t.data[1]]], checked_states+[0], p, file_out, indent_level+1, predictable, next_predict, bound, name, out_predictions)
    file_out.write (' '*indent_level*2 + '}\\\n')
  elif t.data[0] == 'test2':
    file_out.write (' '*indent_level*2 + "if (CONDITION_" + names[n.index(n[t.data[1]])] + ") {\\\n")
    out_predictions = write_node (x, t.right, n, checked_nodes+[names[n.index(n[t.data[1]])]], checked_states+[1], p, file_out, indent_level+1, predictable, next_predict, bound, name, out_predictions)
    file_out.write (' '*indent_level*2 + "} else {\\\n")
    out_predictions = write_node (x, t.left, n, checked_nodes+[names[n.index(n[t.data[1]])]], checked_states+[0], p, file_out, indent_level+1, predictable, next_predict, bound, name, out_predictions)
    file_out.write (' '*indent_level*2 + '}\\\n')
  elif t.data[0] == 'merge':
    file_out.write (' '*indent_level*2 + "BLOCK_x = ")
    nbr_unions = 0
    for i,j in enumerate(t.data[1:]):
      if i < len(t.data)-2:
        nbr_unions+=1
        file_out.write ("UF.Union( ")
      file_out.write ("BLOCK_"+names[n.index(n[j])])
      if i < len(t.data)-2:
        file_out.write (', ')
    for k in range(nbr_unions):
      file_out.write (')')
    file_out.write (';\\\n')
    file_out.write (' '*indent_level*2 +"LOOP_HANDLING\\\n")
    next_subtree, out_predictions = predict_next_subtree (checked_nodes, checked_states, p, x, n, predictable, next_predict, out_predictions, bound, name)
    file_out.write (' '*indent_level*2 + "goto " + next_subtree + ";\\\n")
  elif t.data[0] == 'assign':
    file_out.write (' '*indent_level*2 + "BLOCK_x = BLOCK_" + names[n.index(n[t.data[1]])] + ";\\\n")
    file_out.write (' '*indent_level*2 +"LOOP_HANDLING\\\n")
    next_subtree, out_predictions = predict_next_subtree (checked_nodes, checked_states, p, x, n, predictable, next_predict, out_predictions, bound, name)
    file_out.write (' '*indent_level*2 + "goto " + next_subtree + ";\\\n")
  elif t.data[0] == 'new label':
    file_out.write (' '*indent_level*2 + "BLOCK_x = UF.newLabel ();\\\n")
    file_out.write (' '*indent_level*2 +"LOOP_HANDLING\\\n")
    next_subtree, out_predictions = predict_next_subtree (checked_nodes, checked_states, p, x, n, predictable, next_predict, out_predictions, bound, name)
    file_out.write (' '*indent_level*2 + "goto " + next_subtree + ";\\\n")
  else:
    file_out.write (' '*indent_level*2 + "BLOCK_x = 0;\\\n")
    file_out.write (' '*indent_level*2 +"LOOP_HANDLING\\\n")
    next_subtree, out_predictions = predict_next_subtree (checked_nodes, checked_states, p, x, n, predictable, next_predict, out_predictions, bound, name)
    file_out.write (' '*indent_level*2 + "goto " + next_subtree + ";\\\n")
  return out_predictions

def export_tree_to_c (x, t, n, p, file_out, predictable, next_predict, bound, name):
  # New predictions needed by this subtree 
  out_predictions = []

  name_subtree = name + "_" + get_char_subtree_from_str (p, x)
  file_out.write ("#define " + name_subtree + "\\\n")
  file_out.write ('  FLAG_' + name_subtree + ":\\\n")
#  file_out.write ('  std::cout << \"' + name_subtree + ' at (\" << (int)c << \",\" << (int)r << \")\" << std::endl;\\\n')

  checked_nodes = []
  checked_states = []
  out_predictions = write_node (x, t, n, checked_nodes, checked_states, p, file_out, 1, predictable, next_predict, bound, name, out_predictions) 

  file_out.write ('\n\n')
  return out_predictions

def get_nbr_constraints (p):
  length = 0
  for j,k in enumerate (p):
    if k != '-':
      length += 1
  return length

def build_forest (x,n,s,dt,U,D,H,predictable,next_predict,path,name):
  a,e,o = generate_dt (x,n)
  a,e,o = correction (s,x,n,a,e,o)
  a,e,o = compress_dt (s,x,n,a,e,o)
  a,e = remove_rows (a,e)
  R = compute_gains (a,e,o,x)
  length = len(x)+len(n)

  maj_name = name.upper()

  # Write a source code file
  file_code = open (name+'.hpp', 'w')
  file_code.write ("/*\n * This file was generated\n */\n\n")
  file_code.write ('#ifndef _'+maj_name+'_HPP_\n')
  file_code.write ('#define _'+maj_name+'_HPP_\n\n')

  # For each configuration [start,middle,ends...] 
  for i in range(len(U)):
    
    tree_name = name+'_'+str(i)
    maj_tree_name = maj_name+'_'+str(i)

    file_code.write ("#include \""+path+"/"+tree_name+'.inc\"\n')

    configuration = U[i]
    file_forest = open (path+'/'+tree_name+'.inc','w')
    file_forest.write ("/*\n * This file was generated\n */\n\n")

    file_forest.write ('#ifndef _'+maj_tree_name+'_HPP_\n')
    file_forest.write ('#define _'+maj_tree_name+'_HPP_\n\n')

    start = configuration[0]
    bound = configuration[1]
    ends = configuration[2:]

    stack = []
    stack.append (start)
    processed = []

    while stack:
        c = stack.pop()
        processed.append (c)

        plength = get_nbr_constraints (c)
        print '['+c+'] Total gain : ' + str(R[len(R)-plength-1][c][1]) + "/" + str(2**(length-plength)) + ', Number of leafs : ' + str(2**(length-plength) - R[len(R)-plength-1][c][1])

        # Generate a decision tree from this position
        root = build_tree (None, c, len(R)-plength-1, R, a, x)
        # No repetition
        stack += list(set(export_tree_to_c (x, root, n, c, file_forest, predictable, next_predict, bound, maj_tree_name)))
        # Removing those already processed
        for j in processed:
            while j in stack:
                stack.remove (j)

    end_i = 0
    unused = []
    for c in ends:
      c_already_done = c in processed

      if not c_already_done:
        plength = get_nbr_constraints (c)
        print '['+c+'] Total gain : ' + str(R[len(R)-plength-1][c][1]) + "/" + str(2**(length-plength)) + ', Number of leafs : ' + str(2**(length-plength) - R[len(R)-plength-1][c][1])
        root = build_tree (None, c, len(R)-plength-1, R, a, x)
        unused += list(set(export_tree_to_c(x,root,n,c,file_forest,predictable,next_predict,bound,maj_tree_name)))
        processed += [c]
        for j in processed:
            while j in unused:
                unused.remove(j)

    # Create the handle for a whole line.
    file_forest.write ('#define '+maj_tree_name+'\\\n')
    for tmp_index,tmp in enumerate(processed):
      if tmp in ends:
        file_forest.write('  '+maj_tree_name+'_END_'+str(ends.index(tmp))+' : \\\n')
      name_subtree = maj_tree_name + '_' + get_char_subtree_from_str (tmp, x)
      file_forest.write ('  '+name_subtree+'\\\n')
    file_forest.write ('  '+maj_tree_name+'_UNUSED\n\n')
    unused = list(set(unused))
    for tmp in unused:
        name_subtree = maj_tree_name + '_' + get_char_subtree_from_str (tmp, x)
        file_forest.write ('#define '+name_subtree+'\\\n')         
        file_forest.write ('  FLAG_'+name_subtree+':;\n')
    file_forest.write ("\n#define "+maj_tree_name+'_UNUSED\\\n')
    for tmp in unused:
      name_subtree = maj_tree_name + '_' + get_char_subtree_from_str (tmp, x)
      file_forest.write ('  '+name_subtree+'\\\n')         
    file_forest.write ('\n')

    file_forest.write ('#endif // _'+maj_tree_name+'_HPP_\n')
    file_forest.close ()

  names_x = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
  names_n = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v', 'w', 'x', 'y', 'z', 'a2', 'b2', 'c2']
  p = x+n
  names = names_x[0:len(x)] + names_n[0:len(n)]

  file_code.write ("\n#define BLOCK_x out_s_r[x]")
  for _n,_p in zip(names_n,n):
    file_code.write ("\n#define BLOCK_"+_n+" out_s")
    _z = (_p[0]/dt[0])*dt[0]
    _y = (_p[1]/dt[1])*dt[1]
    _x = (_p[2]/dt[2])*dt[2]
    if _z < 0:
      file_code.write ('p'*(abs(_z)))
    else:
      file_code.write ('n'*_z)
    file_code.write ('_r')
    if _y < 0:
      file_code.write ('p'*(abs(_y)))
    else:
      file_code.write ('n'*_y)
    file_code.write ('[x + ('+str(_x)+')]')

  for _n,_p in zip(names,p):
    file_code.write ("\n#define CONDITION_"+_n+" img_s")
    _z = _p[0]
    _y = _p[1]
    _x = _p[2]
    if _z < 0:
      file_code.write ('p'*(abs(_z)))
    else:
      file_code.write ('n'*_z)
    file_code.write ('_r')
    if _y < 0:
      file_code.write ('p'*(abs(_y)))
    else:
      file_code.write ('n'*_y)
    file_code.write ('[x + ('+str(_x)+')] > 0')


  file_code.write ("\n\ntemplate <class T, class labelT>\n")
  file_code.write ("size_t "+name+" (T* in, labelT* out, union_find<labelT> &UF, size_t w, size_t h, size_t d) {\n\n")

  file_code.write ("  size_t n_labels=0;")
  file_code.write ("  size_t x=0, y=0, z=0;\n")
  if dt[2] > 1:
    file_code.write ("  size_t w_up = (w/"+str(dt[2])+"+1)*"+str(dt[2])+";\n")
  else:
    file_code.write ("  size_t w_up = w;\n")
  if dt[1] > 1:
    file_code.write ("  size_t h_up = (h/"+str(dt[1])+"+1)*"+str(dt[1])+";\n")
  else:
    file_code.write ("  size_t h_up = h;\n")
  if dt[0] > 1:
    file_code.write ("  size_t d_up = (d/"+str(dt[0])+"+1)*"+str(dt[0])+";\n\n")
  else:
    file_code.write ("  size_t d_up = d;\n\n")

  j = 0
  k = 0
  j_max = len(H[0])
  j_start = 0

  file_code.write ("  // First-pass\n")

  # For each configuration [start,middle,ends...] 
  for i in range(len(U)):
    tree_name = name+'_'+str(i)
    maj_tree_name = maj_name+'_'+str(i)

    configuration = U[i]
    domain = D[i]
    domain_block = list(D[i])
    domain_block[0] = set([(w/dt[0])*dt[0] for w in domain[0]])
    domain_block[1] = set([(w/dt[1])*dt[1] for w in domain[1]])
    domain_block[2] = set([(w/dt[2])*dt[2] for w in domain[2]])

    end_w = max(0,len(configuration)-2)
    end_h = max(0,len(H[k])-2)

    # Flag 
    file_code.write ("  "+maj_tree_name+"_BEGIN:\n")

    # Loop handling.
    if i != 0:
      file_code.write ("  #undef LOOP_HANDLING\n\n")
    if j < 2:
      file_code.write ("  #define LOOP_HANDLING\\\n")
      file_code.write ("  x += "+str(dt[2])+";\\\n")
      for e in range(end_w):
        file_code.write ("  if ( w - x ==  " + str(e+1) + ')\\\n')
        file_code.write ("    goto "+maj_tree_name+"_END_"+str(e)+";\\\n")
      file_code.write ("  if ( x == w_up ) {\\\n")
      file_code.write ("    x = 0;\\\n")
      file_code.write ("    y += "+str(dt[1])+";\\\n")
      for e in range(end_h):
        file_code.write ("    if ( h - y ==  " + str(e+1) + ')\\\n')
        file_code.write ("      goto "+maj_name+"_"+str(j_start+2+e)+"_BEGIN;\\\n")
      file_code.write ("    if ( y == h_up ) {\\\n")
      file_code.write ("      y = 0;\\\n")  
      file_code.write ("      z += "+str(dt[0])+";\\\n")
      if k == 0 or k == 1:
        file_code.write ("      if (z == d_up)\\\n")
        file_code.write ("        goto "+maj_name+"_END;\\\n")
        if len(H) == 1:
          file_code.write ("      goto "+maj_name+"_0_BEGIN;\\\n")
        else:
          file_code.write ("      goto "+maj_name+"_"+str(len(H[0]))+"_BEGIN;\\\n")
      else: 
        file_code.write ("      goto "+maj_name+"_END;\\\n")
      file_code.write ("    }\\\n")
      file_code.write ("    goto "+maj_name+"_"+str(j_start+1)+"_BEGIN;\\\n")
      file_code.write ("  }\n\n")
    else:
      file_code.write ("  #define LOOP_HANDLING\\\n")
      file_code.write ("  x += "+str(dt[2])+";\\\n")
      for e in range(end_w):
        file_code.write ("  if ( w - x ==  " + str(e+1) + ')\\\n')
        file_code.write ("    goto "+maj_tree_name+"_END_"+str(e)+";\\\n")
      file_code.write ("  if ( x == w_up ) {\\\n")
      file_code.write ("    x = 0;\\\n")
      file_code.write ("    y = 0;\\\n")  
      file_code.write ("    z += "+str(dt[0])+";\\\n")
      if k == 0 or k == 1:
        file_code.write ("      if (z == d_up)\\\n")
        file_code.write ("        goto "+maj_name+"_END;\\\n")
        if len(H) == 1:
          file_code.write ("      goto "+maj_name+"_0_BEGIN;\\\n")
        else:
          file_code.write ("      goto "+maj_name+"_"+str(len(H[0]))+"_BEGIN;\\\n")
      else: 
        file_code.write ("      goto "+maj_name+"_END;\\\n")
      file_code.write ("  }\n\n")
    file_code.write ("  {\n")

    # Input lines
    file_code.write ("    const T *const img_s_r = in + z*w*h + y*w;\n")
    for _z in domain[0]:
      for _y in domain[1]:
        if _z != 0 or _y != 0:
          file_code.write ("    const T *const img_s")
          if _z < 0:
            file_code.write ('p'*(abs(_z)))
          else:
            file_code.write ('n'*_z)
          file_code.write ('_r')
          if _y < 0:
            file_code.write ('p'*(abs(_y)))
          else:
            file_code.write ('n'*_y)
          file_code.write (" = img_s_r + (" + str(_z) +")*w*h + (" + str(_y)+")*w;\n")
    # Output lines
    file_code.write ("    labelT *const out_s_r = out + z*w*h + y*w;\n")
    for _z in domain_block[0]:
      for _y in domain_block[1]:
        if _z != 0 or _y != 0:
          file_code.write ("    labelT *const out_s")
          if _z < 0:
            file_code.write ('p'*(abs(_z)))
          else:
            file_code.write ('n'*_z)
          file_code.write ('_r')
          if _y < 0:
            file_code.write ('p'*(abs(_y)))
          else:
            file_code.write ('n'*_y)
          file_code.write (" = out_s_r + (" + str(_z) +")*w*h + (" + str(_y)+")*w;\n")

    file_code.write ("    "+maj_tree_name+"\n")
    file_code.write ("  }\n\n")

    j += 1
    if j == j_max:
      j = 0
      k += 1
      if k < len(H):
        j_start += j_max
        j_max = len(H[k])

  # Flag 
  file_code.write ("  "+maj_name+"_END:\n\n")

  file_code.write ("  n_labels = UF.Flatten ();\n\n")

  file_code.write ("  // Second-pass\n\n")

  access_x = []
  for X in x:
      access = "_s"+'n'*X[0]+"_r"+'n'*X[1]+"[x"
      if X[2] != 0:
       access += "+"+str(X[2])
      access += "]"
      access_x.append (access)

  file_code.write ("  z = 0;\n")
  if dt[0]==1:
    file_code.write ("  for (; z < d; z++) {\n")
  else:
    file_code.write ("  for (; z < (d/"+str(dt[0])+")*"+str(dt[0])+"; z+="+str(dt[0])+") {\n")
  file_code.write ("    y = 0;\n")
  if dt[1] == 1:
    file_code.write ("    for (; y < h; y++) {\n")
  else:
    file_code.write ("    for (; y < (h/"+str(dt[1])+")*"+str(dt[1])+"; y+="+str(dt[1])+") {\n")
  file_code.write ("      // get rows pointer\n")
  file_code.write ("      const T *const img_s_r = in + z*w*h + y*w;\n")
  for _z in range(dt[0]):
    for _y in range(dt[1]):
      if _z != 0 or _y != 0:
        file_code.write ("      const T* const img_s")
        file_code.write ('n'*_z)
        file_code.write ('_r')
        file_code.write ('n'*_y)
        file_code.write (' = img_s_r + (' + str(_z) + ")*w*h + ("+ str(_y)+")*w;\n")
  file_code.write ("      labelT *const out_s_r = out + z*w*h + y*w;\n")
  for _z in range(dt[0]):
    for _y in range(dt[1]):
      if _z != 0 or _y != 0:
        file_code.write ("      labelT* const out_s")
        file_code.write ('n'*_z)
        file_code.write ('_r')
        file_code.write ('n'*_y)
        file_code.write (' = out_s_r + (' + str(_z) + ")*w*h + ("+ str(_y)+")*w;\n")
  file_code.write ("      x = 0;\n")
  if dt[2]== 1:
    file_code.write ("      for (; x < w; x++) {\n")
  else:
    file_code.write ("      for (; x < (w/"+str(dt[2])+")*"+str(dt[2])+"; x+="+str(dt[2])+") {\n")
  file_code.write ("        labelT label = out_s_r[x];\n")
  file_code.write ("        if (label > 0) {\n")
  file_code.write ("          label = UF.Find (label);")
  for access in access_x:
    file_code.write ("          if (img"+access+" > 0) \n")
    file_code.write ("            out"+access+" = label;\n")
    file_code.write ("          else\n")
    file_code.write ("            out"+access+" = 0;\n")
  file_code.write ("        } else {\n")
  for access in access_x:
    file_code.write ("          out"+access+" = 0;\n") 
  file_code.write ("        }\n")
  file_code.write ("      }\n")
  for _w in range(1,dt[2]): 
    file_code.write ("      if (w-x == "+str(_w)+") {\n")
    file_code.write ("        labelT label = out_s_r[x];\n")
    file_code.write ("        if (label > 0) {\n")
    file_code.write ("          label = UF.Find (label);")
    for X,access in zip(x,access_x):
      if (X[2] < _w):
        file_code.write ("          if (img"+access+" > 0) \n")
        file_code.write ("            out"+access+" = label;\n")
        file_code.write ("          else\n")
        file_code.write ("            out"+access+" = 0;\n")
    file_code.write ("        } else {\n")
    for X,access in zip(x,access_x):
      if (X[2] < _w):
        file_code.write ("          out"+access+" = 0;\n") 
    file_code.write ("        }\n")
    file_code.write ("      }\n")
  file_code.write ("    }\n")
  for _w in range (1, dt[1]):
    file_code.write ("    if (h-y == "+str(_w)+") {\n")

    file_code.write ("      // get rows pointer\n")
    file_code.write ("      const T *const img_s_r = in + z*w*h + y*w;\n")
    for _z in range(dt[0]):
      for _y in range(_w):
        if _z != 0 or _y != 0:
          file_code.write ("      const T* const img_s")
          file_code.write ('n'*_z)
          file_code.write ('_r')
          file_code.write ('n'*_y)
          file_code.write (' = img_s_r + (' + str(_z) + ")*w*h + ("+ str(_y)+")*w;\n")
    file_code.write ("      labelT *const out_s_r = out + z*w*h + y*w;\n")
    for _z in range(dt[0]):
      for _y in range(_w):
        if _z != 0 or _y != 0:
          file_code.write ("      labelT* const out_s")
          file_code.write ('n'*_z)
          file_code.write ('_r')
          file_code.write ('n'*_y)
          file_code.write (' = out_s_r + (' + str(_z) + ")*w*h + ("+ str(_y)+")*w;\n")
    file_code.write ("      x = 0;\n")
    if dt[2]== 1:
      file_code.write ("      for (; x < w; x++) {\n")
    else:
      file_code.write ("      for (; x < (w/"+str(dt[2])+")*"+str(dt[2])+"; x+="+str(dt[2])+") {\n")
    file_code.write ("        labelT label = out_s_r[x];\n")
    file_code.write ("        if (label > 0) {\n")
    file_code.write ("          label = UF.Find (label);")
    for X,access in zip(x,access_x):
      if (X[1] < _w):
        file_code.write ("          if (img"+access+" > 0) \n")
        file_code.write ("            out"+access+" = label;\n")
        file_code.write ("          else\n")
        file_code.write ("            out"+access+" = 0;\n")
    file_code.write ("        } else {\n")
    for X,access in zip(x,access_x):
      if (X[1] < _w):
        file_code.write ("          out"+access+" = 0;\n") 
    file_code.write ("        }\n")
    file_code.write ("      }\n")
    for _z in range(1,dt[2]): 
      file_code.write ("      if (w-x == "+str(_w)+") {\n")
      file_code.write ("        labelT label = out_s_r[x];\n")
      file_code.write ("        if (label > 0) {\n")
      file_code.write ("          label = UF.Find (label);")
      for X,access in zip(x,access_x):
        if (X[2] < _z and X[1] < _w):
          file_code.write ("          if (img"+access+" > 0) \n")
          file_code.write ("            out"+access+" = label;\n")
          file_code.write ("          else\n")
          file_code.write ("            out"+access+" = 0;\n")
      file_code.write ("        } else {\n")
      for X,access in zip(x,access_x):
        if (X[2] < _z and X[1] < _w):
          file_code.write ("          out"+access+" = 0;\n") 
      file_code.write ("        }\n")
      file_code.write ("      }\n")
    file_code.write ("    }\n")

  file_code.write ("  }\n\n")

  file_code.write ("  return n_labels;\n\n")

  file_code.write ('}\n')

  file_code.write ("\n#undef BLOCK_x")
  for _n,_p in zip(names_n,n):
    file_code.write ("\n#undef BLOCK_"+_n)
  for _n,_p in zip(names,p):
    file_code.write ("\n#undef CONDITION_"+_n)

  file_code.write ('\n\n#endif // _'+maj_name+'_HPP_\n')
  file_code.close ()

def setup (x,n,s):
  names_x = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
  names_n = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v', 'w', 'x', 'y', 'z', 'a2', 'b2', 'c2']

  dt = [0,0,0]
  dt[0] = max ([w[0] for w in x if w[1] == 0 and w[2] == 0])+1
  dt[1] = max ([w[1] for w in x if w[0] == 0 and w[2] == 0])+1
  dt[2] = max ([w[2] for w in x if w[0] == 0 and w[1] == 0])+1

  p = x+n
  names = names_x[0:len(x)] + names_n[0:len(n)]
  p_translated = [[w[0],w[1],w[2]+dt[2]] for w in p]

  predictable = []
  next_predict = {}
  i=0
  j=0
  for v in p:
    j=0
    for w in p_translated:
      if v==w:
        n1 = names[i] 
        n2 = names[j]
        predictable.append (n1)
        next_predict[n1]=n2
      j+=1
    i+=1

  configuration = []
  drange = [min([w[0] for w in p]),max([w[0] for w in p])+1]

  U = []
  D = []

  H = [[]]*(drange[1]-drange[0])

  for i in range (drange[0],drange[1]):
    pp = p
    if i < 0:
      pp = [w for w in p if w[0]>i]
    elif i > 0 :
      pp = [w for w in p if w[0]<i]
    hrange = [min([w[1] for w in pp]),max([w[1] for w in pp])+1] 
    H[i] = [[]]*(hrange[1]-hrange[0])
    for j in range (hrange[0],hrange[1]):
      ppp = pp
      if j < 0:
        ppp = [w for w in pp if w[1]>j]
      elif j > 0:
        ppp = [w for w in pp if w[1]<j]
      wrange = [min([w[2] for w in ppp]),max([w[2] for w in ppp])+1]  
      H[i][j] = [None]*(wrange[1]-wrange[0])
      P = []
      for k in range (wrange[0],wrange[1]):
        pppp = ppp
        if k < 0:
          pppp = [w for w in ppp if w[2] > k]
        elif k > 0:
          pppp = [w for w in ppp if w[2] < k]
        command = ['0']*len(p)
        for w in pppp:
          command[p.index(w)] = '-'
        P.append (''.join(command))
      U.append (P)
      domain_w = range(min([w[2] for w in ppp]),max([w[2] for w in ppp])+1)
      domain_h = range(min([w[1] for w in ppp]),max([w[1] for w in ppp])+1)
      domain_d = range(min([w[0] for w in ppp]),max([w[0] for w in ppp])+1)
      D.append ([domain_d,domain_h,domain_w])

  return dt, predictable, next_predict, U, D, H

def connex_8_block ():
  x = []
  n = []

  s = [[0,-1,-1],
       [0,-1, 0],
       [0,-1, 1],
       [0, 0,-1],
       [0, 0, 1],
       [0, 1,-1],
       [0, 1, 0],
       [0, 1, 1]
      ]

  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  x.append ([0,0,1])
  x.append ([0,1,0])
  x.append ([0,1,1])
  # neighbors.
  n.append ([0,-1,-1])
  n.append ([0,-1,0])
  n.append ([0,-1,1])
  n.append ([0,-1,2])
  n.append ([0,0,-1])
  n.append ([0,1,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_8_block')

def connex_4 ():
  x = []
  n = []

  s = [[0,-1, 0],
       [0, 0,-1],
       [0, 0, 1],
       [0, 1, 0]
      ]

  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  # neighbors.
  n.append ([0,-1,0])
  n.append ([0,0,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_4')

def connex_4_hor ():
  x = []
  n = []

  s = [[0,-1, 0],
       [0, 0,-1],
       [0, 0, 1],
       [0, 1, 0]
      ]

  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  x.append ([0,0,1])
  # neighbors.
  n.append ([0,-1,0])
  n.append ([0,-1,1])
  n.append ([0,0,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_4_hor')

def connex_4_ver ():
  x = []
  n = []

  s = [[0,-1, 0],
       [0, 0,-1],
       [0, 0, 1],
       [0, 1, 0]
      ]

  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  x.append ([0,1,0])
  # neighbors.
  n.append ([0,-1,0])
  n.append ([0,0,-1])
  n.append ([0,1,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_4_ver')

def connex_8 ():
  x = []
  n = []

  s = [[0,-1,-1],
       [0,-1, 0],
       [0,-1, 1],
       [0, 0,-1],
       [0, 0, 1],
       [0, 1,-1],
       [0, 1, 0],
       [0, 1, 1]
      ]

  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  # neighbors.
  n.append ([0,-1,-1])
  n.append ([0,-1,0])
  n.append ([0,-1,1])
  n.append ([0,0,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_8')

def connex_6 ():
  x = []
  n = []

  s = [[-1, 0, 0],
       [ 0,-1, 0],
       [ 0, 0,-1],
       [ 0, 0, 1],
       [ 0, 1, 0],
       [ 1, 0, 0]
      ]

  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  # neighbors.
  n.append ([-1,0,0])
  n.append ([0,-1,0])
  n.append ([0,0,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_6')

def connex_6_hor ():
  x = []
  n = []

  s = [[-1, 0, 0],
       [ 0,-1, 0],
       [ 0, 0,-1],
       [ 0, 0, 1],
       [ 0, 1, 0],
       [ 1, 0, 0]
      ]

  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  x.append ([0,0,1])
  # neighbors.
  n.append ([-1,0,0])
  n.append ([-1,0,1])
  n.append ([0,-1,0])
  n.append ([0,-1,1])
  n.append ([0,0,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_6_hor')

def connex_6_ver ():
  x = []
  n = []

  s = [[-1, 0, 0],
       [ 0,-1, 0],
       [ 0, 0,-1],
       [ 0, 0, 1],
       [ 0, 1, 0],
       [ 1, 0, 0]
      ]


  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  x.append ([0,1,0])
  # neighbors.
  n.append ([-1,0,0])
  n.append ([-1,1,0])
  n.append ([0,-1,0])
  n.append ([0,0,-1])
  n.append ([0,1,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_6_ver')

#takes around 8 minutes to compute.
def connex_26 ():
  x = []
  n = []

  s = [
       [-1,-1,-1],
       [-1,-1, 0],
       [-1,-1, 1],
       [-1, 0,-1],
       [-1, 0, 0],
       [-1, 0, 1],
       [-1, 1,-1],
       [-1, 1, 0],
       [-1, 1, 1],
       [0,-1,-1],
       [0,-1, 0],
       [0,-1, 1],
       [0, 0,-1],
       [0, 0, 1],
       [0, 1,-1],
       [0, 1, 0],
       [0, 1, 1],
       [1,-1,-1],
       [1,-1, 0],
       [1,-1, 1],
       [1, 0,-1],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1,-1],
       [1, 1, 0],
       [1, 1, 1]
      ]

  if not os.path.exists ('decision_trees'):
    os.makedirs('decision_trees')

  # points to label.
  x.append ([0,0,0])
  # neighbors.
  n.append ([-1,-1,-1])
  n.append ([-1,-1, 0])
  n.append ([-1,-1, 1])
  n.append ([-1, 0,-1])
  n.append ([-1, 0, 0])
  n.append ([-1, 0, 1])
  n.append ([-1, 1,-1])
  n.append ([-1, 1, 0])
  n.append ([-1, 1, 1])
  n.append ([0,-1,-1])
  n.append ([0,-1,0])
  n.append ([0,-1,1])
  n.append ([0,0,-1])

  dt, predictable, next_predict, U, D, H = setup (x,n,s)
  build_forest (x,n,s,dt,U,D,H,predictable,next_predict,'decision_trees','connex_26')

import time
def st_time(func):
    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print "Function=%s, Time=%s" % (func.__name__, t2 - t1)
        return r

    return st_func

def generate_all ():
  print "\nGenerating Connexity 4"
  st_time(connex_4)()
  print "\nGenerating Connexity 4 horizontally optimised"
  st_time(connex_4_hor)()
  print "\nGenerating Connexity 4 vertically optimised"
  st_time(connex_4_ver)()
  print "\nGenerating Connexity 8"
  st_time(connex_8)()
  print "\nGenerating Connexity 8 block optimised"
  st_time(connex_8_block)()
#  print "\nGenerating Connexity hexagonal"
#  st_time(connex_hex)()
#  print "\nGenerating Connexity hexagonal down triangle optimised"
#  st_time(connex_hex_block_down)()
#  print "\nGenerating Connexity hexagonal up triangle optimised"
#  st_time(connex_hex_block_up)()
  print "\nGenerating Connexity Cross 3D"
  st_time(connex_6)()
  print "\nGenerating Connexity Cross 3D horizontally optimised"
  st_time(connex_6_hor)()
  print "\nGenerating Connexity Cross 3D vertically optimised"
  st_time(connex_6_ver)()
  print "\nGenerating Connexity 26"
  st_time(connex_26)()
#  print "\nGenerating Connexity 26 horizontally optimised"
#  st_time(connex_26_hor())
#  print "\nGenerating Connexity 26 block optimised"
#  st_time(connex_26_block())

