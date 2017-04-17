import pydot

graph = pydot.Dot(graph_type='digraph')

# creating nodes is as simple as creating edges!
node_a = pydot.Node(name="nodea", style="filled", fillcolor="white",fontcolor='blue',color='blue',texlbl=r'$\omega$',label='Initial state')
node_b = pydot.Node(name="nodeb", style="filled", fillcolor="white",texlbl=r'$\omega$',label='Spin coin by\na small angle')
#node_b = pydot.Node("Node B", style="filled", fillcolor="green")
#node_c = pydot.Node("Node C", style="filled", fillcolor="#0000ff")
#node_d = pydot.Node("Node D", style="filled", fillcolor="#976856")

graph.add_node(node_a)
graph.add_node(node_b)
#graph.add_node(node_c)
#graph.add_node(node_d)

graph.add_edge(pydot.Edge(node_a, node_b,label='START',fontcolor='red'))
#graph.add_edge(pydot.Edge(node_b, node_c))
#graph.add_edge(pydot.Edge(node_c, node_d))
# but, let's make this last edge special, yes?
#graph.add_edge(pydot.Edge(node_d, node_a, label="and back we go again", labelfontcolor="#009933", fontsize="10.0", color="blue"))

# and we are done
graph.write_png('complicated.png',prog='dot')

