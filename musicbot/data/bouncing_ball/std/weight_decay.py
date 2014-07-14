
def l_p(x,p):
    return sum(x.ravel()**p)
def dl_p(x,p):
    return p*sum(x.ravel()**(p-1))
