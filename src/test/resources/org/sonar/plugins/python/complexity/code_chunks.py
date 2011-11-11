def if_else(): # 3
    if 0 and 0:
        pass
    else:
        pass

def if_elif_else(): # 5
    if 0 and 0:
        pass
    elif 0 and 0:
        pass
    else:
        pass
                
def if_compl_cond1(): # 5
    if (0 and 0) or (0 and 0):
        pass
    else:
        pass

def if_compl_cond2(): # 4
    if 0 and 0 or 0:
        pass
    else:
        pass
    
def for_else(): # 2
    for x in xrange(10):
        pass
    else:
        pass
        
def while_comp_cond(): # 3
    while 0 > 0 and 0 < 0:
        pass
        
def while_else(): # 2
    while 0 < 100:
        pass
    else:
        pass
        
def while_else_compl_cond1(): # 3
    while 0 > 0 and 0 < 0:
        pass
    else:
        pass

def while_else_compl_cond2(): # 5
    while (0 and 0) or (0 and 0):
        pass
    else:
        pass

def while_else_compl_cond3(): # 4
    while 0 and 0 or 0:
        pass
    else:
        pass

def list_compr(): # 2
    [x for x in []]
    
def list_compr_filter(): # 3
    [x for x in [] if True]
    
def gen_expr(): # 2
    (x for x in [])
    
def gen_expr_filter(): # 3
    (x for x in [] if True)
