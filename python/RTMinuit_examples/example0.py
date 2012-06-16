from RTMinuit import *

################################################################################

class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class F:
    def __init__(self,data,params):
        self.data = data
        self.params = params
        varnames = ['%s'%i for i in params]

        self.func_code = Struct(co_argcount=len(params),co_varnames=varnames)
        self.func_defaults = None # Optional but makes vectorize happy

    def __call__(self,*arg):
        flag = arg[0]
        if flag==0:
            return (arg[1]-self.data[0])**2 + (arg[2]-self.data[1])**2 + (arg[3]-self.data[2])**2 -1.
        elif flag==1:
            return (arg[1]-self.data[0])**4 + (arg[2]-self.data[1])**4 + (arg[3]-self.data[2])**4 -1.

################################################################################

myparams = ['flag','x','y','z']
f = F([1,2,3],myparams)

kwd = {}
kwd['flag']=1
kwd['fix_flag']=True
kwd['x']=3.0
#kwd['fix_x']=False
kwd['y']=1.0
#kwd['fix_y']=False
kwd['z']=4.0
#kwd['fix_z']=False
kwd['printlevel']=0

m = Minuit(f,**kwd)
#m = Minuit(f)

m.release_all_params()

print "List of free parameters:"
print m.list_of_vary_param()
print m.free_param
print "List of fixed parameters:"
print m.list_of_fixed_param()
print m.fix_param

print "Values: "
print m.values
print "Args: "
print m.args

m.migrad()
print m.values,m.errors

