import sys
import numpy as np

from anytree import Node, RenderTree, AsciiStyle, Walker, PreOrderIter

infilename = sys.argv[1]
infile = open(infilename,'r')

def traverse2(node):
    nodename = 'FILE'
    if type(node.name)==str:
        nodename = node.name
        print("open " + nodename + " -> " + nodename + " -> ")
    for child in node.children:
         traverse2(child)
    print("close " + nodename + " -> ")

'''
def traverse2(node):
    nodename = 'FILE'
    if type(node.name)==str:
        nodename = node.name
        print("open " + nodename + " -> " + nodename + " -> ")
    for child in node.children:
         traverse2(child)

    if type(node.name)==str:
        print("close " + nodename + " -> ")
'''


def traverse(node, nodenames_traversed=[], total=0):
    nodename = 'BLANK'
    if type(node.name) == str: # Directory
        nodename = node.name
        print("open " + nodename + " -> " + nodename + " -> ")

        fullnodename = ""
        for n in node.path:
            fullnodename += n.name + "/"

        if fullnodename not in nodenames_traversed:
            print("RESETTING TOTAL!")
            total = 0
            nodenames_traversed.append(fullnodename)

        dirs = []
        files = []
        for child in node.children:
            if type(child.name) == str:
                dirs.append(child)
            else:
                files.append(child)
        #print("Collection of children and files")
        #print(dirs)
        #print(files)

        for child in dirs:
            print("child: ", child)
            return_value,nodenames_traversed = traverse(child,nodenames_traversed=nodenames_traversed, total=total)
            total += return_value
        for child in files:
            print("child: ", child)
            total += int(child.name[1])

        print("close " + nodename + " ->     size: " + str(total))
        if total <= 100000:
            print("THIS ONE close " + nodename + " ->     size: " + str(total))
        return total,nodenames_traversed
        


# Part A


level = 0

fs = []

total_gt_10000 = 0

#directories = {'root': []}
root = Node('root')
current_node = root
print("STARTING: ")
print(root.name)

icount = 0
for line in infile:
    print(current_node)
    icount += 1
    print(f"LINE: {icount}", line)
    if line.find('$')>=0:
        if line[2:4] == 'ls':
            1
        elif line[2:4] == 'cd':
            if line.find('/')>=0:
                level = 0
                current_node = root
                print("At the top!")
            elif line.find('..')>=0:
                level -= 1
                print("HERE: ", line)
                #print(line)
                #'''
                print("Node   name: ", current_node.name)
                print("Parent name: ", current_node.parent.name)
                print(f"Going up from {current_node.name} to {current_node.parent.name}")
                current_node = current_node.parent
                #'''
                '''
                if current_node is not None:
                    print("Node   name: ", current_node.name)
                    if current_node.parent is not None:
                        print("Parent name: ", current_node.parent.name)
                        print(f"Going up from {current_node.name} to {current_node.parent.name}")
                        current_node = current_node.parent
                '''
            else:
                #print("Going down!")
                name = line.split()[2]
                print(f"Going down from {current_node.name} to {name}")
                print(f'name: ', name)
                x = Node(name, parent=current_node)
                current_node = x
                level += 1
    else:
        if line.find('dir')<0:
            print(line)
            size = int(line.split()[0])
            name = line.split()[1]
            x = Node([name, size], parent=current_node)

    #for pre, fill, node in RenderTree(root):
        #print("%s%s" % (pre, node.name))
    #print(RenderTree(root, style=AsciiStyle()))

    print(f"LEVEL LINE: {level} ",line)

#for pre, fill, node in RenderTree(root):
#    print("%s%s" % (pre, node.name))
print()
print()
print(RenderTree(root, style=AsciiStyle()))


#print()
#traverse2(root)
print()
print()

#'''
nodenames_traversed = []
test = traverse(root, nodenames_traversed = nodenames_traversed, total=0)
if test is not None:
    total_size,nodenames_traversed = test
    print(total_size)
    print(nodenames_traversed)
#'''


# Part B
rootsize =  50216456
totaldisk = 70000000

need_to_free = 30000000 - (totaldisk - rootsize)
print(f"Need to free: {need_to_free}")
# 10216456
