# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:02:46 2018

@author: gvch48
"""

import random
import matplotlib.pyplot as plt

def make_coauth_graph():        #1559 vertices, 40016 edges
    
    coauth_graph = {}
    for vertex in range(1559): coauth_graph[vertex] = []
    
    with open('coauthorship.txt') as file:          
        for num, line in enumerate(file, 0):
            if '*Edges' in line:
                edgesIterStart = int(num)

    with open('coauthorship.txt') as file:   
        for num, line in enumerate(file, 0):    
            if edgesIterStart < num:
                lineText = line.split()
                vertexA = int(lineText[0])
                vertexB = int(lineText[1])
                
                if vertexA == 1559:
                    vertexA = 0
                if vertexB == 1559:
                    vertexB = 0
                
                if vertexA == vertexB or vertexA in coauth_graph[vertexB]:
                    continue
                else:
                    coauth_graph[vertexA] += [vertexB]
                    coauth_graph[vertexB] += [vertexA]
    return coauth_graph


def make_ring_group_graph(m,k,p,q):             #copied from question 1 code
    #initialize empty grqph
#    ring_group_graph = [[] for i in range(m*k)]
    ring_group_graph = {}
    for vertex in range(m*k): ring_group_graph[vertex] = []
    for vertex in range(m*k):
        for other_vertex in range(vertex+1,m*k):
            groupDiff = (other_vertex % m) - (vertex % m) 
            random_number = random.random()
            if groupDiff in [-1,0,1,m-1] and random_number < p and other_vertex not in ring_group_graph[vertex]:
                ring_group_graph[vertex] += [other_vertex]
                ring_group_graph[other_vertex] += [vertex]
            elif random_number < q and other_vertex not in ring_group_graph[vertex]:
                ring_group_graph[vertex] += [other_vertex]
                ring_group_graph[other_vertex] += [vertex]
 #       ring_group_graph[vertex] = dict(set(ring_group_graph[vertex]))
    return ring_group_graph

class PATrial:          #taken from lecture 3
    """
    Used when each new node is added in creation of a PA graph.
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are in proportion to the
    probability that it is linked to.
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a PATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities
        
        Returns:
        Set of nodes
        """       
        #compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors
    
def make_complete_graph(num_nodes):         #taken from lecture 3
    """Takes the number of nodes num_nodes and returns a dictionary
    corresponding to a complete directed graph with the specified number of
    nodes. A complete graph contains all possible edges subject to the
    restriction that self-loops are not allowed. The nodes of the graph should
    be numbered 0 to num_nodes - 1 when num_nodes is positive. Otherwise, the
    function returns a dictionary corresponding to the empty graph."""
    #initialize empty graph
    complete_graph = {}
    #consider each vertex
    for vertex in range(num_nodes):
        #add vertex with list of neighbours
        complete_graph[vertex] = set([j for j in range(num_nodes) if j != vertex])
    return complete_graph
    
def make_PA_Graph(total_nodes, out_degree):         #taken from lecture 3
    """creates a PA_Graph on total_nodes where each vertex is iteratively
    connected to a number of existing nodes equal to out_degree"""
    #initialize graph by creating complete graph and trial object
    PA_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)
    for vertex in range(out_degree, total_nodes):
        PA_graph[vertex] = trial.run_trial(out_degree)
        for neighbour in PA_graph[vertex]:                  #makes graph undirected
            PA_graph[neighbour].add(vertex)
    for vertex in PA_graph:                               #makes sure no duplicates
        PA_graph[vertex] = list(set(PA_graph[vertex]))
    return PA_graph


def compute_brilliance(graph,vertex):
    starpoints = []
    for neighbour in graph[vertex]:
        starpoints += [neighbour]
    worstNeighbour = -1
    worstScore = -1
    if len(starpoints) > 0:
        while worstScore != 0:
            for neighbour1 in starpoints:
                nScore = 0
                for neighbour2 in starpoints:
                    if neighbour2 in graph[neighbour1]:
                        nScore += 1
                if nScore > worstScore:
                    worstScore = nScore
                    worstNeighbour = neighbour1
            if worstScore > 0:
                starpoints.remove(worstNeighbour)
                worstNeighbour = -1
                worstScore = -1
    return len(starpoints)

def compute_brilliance_V2(graph,vertex):
    selectionpool = []
    starpoints = []
    for neighbour in graph[vertex]:
        selectionpool.append(neighbour)
    while selectionpool != []:
     #   print(len(graph[vertex]))                        #for debugging
        minConnections = len(selectionpool) #initialise
        bestChoice = -1     #initialse
        for selection in selectionpool:
            connections= 0      #initialise
            for otherselection in selectionpool:
                if otherselection in graph[selection]:
                    connections += 1
            if connections < minConnections:
                minConnections = connections
                bestChoice = selection
        starpoints.append(bestChoice)
        selectionpool.remove(bestChoice)
        for v in selectionpool:
            if v in graph[bestChoice]:
                selectionpool.remove(v)
    return len(starpoints)
    


def compute_All_Brilliances(graph):
    brilliances = {}
    for vertex in graph:
        print('Vertex '+str(vertex)+' of '+str(len(graph)))       #for checking progress. can be removed
        brilliances[vertex] = compute_brilliance_V2(graph,vertex)
    return brilliances

def brilliance_distribution(graph):
    brilliances = compute_All_Brilliances(graph)
    brilliance_distribution = {}
    for vertex in brilliances:
        if brilliances[vertex] in brilliance_distribution:
            brilliance_distribution[brilliances[vertex]] += 1
        else:
            brilliance_distribution[brilliances[vertex]] = 1
    return brilliance_distribution

def make_Brilliance_distrib_image(graph, graphType):
    distrib = brilliance_distribution(graph)
   
    normalized_distribution = {}
    for degree in distrib:
        normalized_distribution[degree] = distrib[degree] / len(graph)
   
    #create arrays for plotting
    xdata = []
    ydata = []
    for degree in distrib:
        xdata += [degree]
        ydata += [normalized_distribution[degree]]
    
    #plot degree distribution
    plt.clf() #clears plot
    plt.xlabel('Brilliance')
    plt.ylabel('Normalized Rate')
    plt.title('Normalized brilliance Distribution of Graph')
    plt.semilogy(xdata, ydata, marker='.', linestyle='None', color='b')
    plt.savefig('V2 normalized brilliance distrib '+str(graphType)+' .png',dpi=300)


"""
####CODE TO PRODUCE IMAGES####
"""

"""
COAUTHORSHIP GRAPH
"""
#graph = make_coauth_graph()
#make_Brilliance_distrib_image(graph,'coauthorship graph 1559')

"""
RING GROUP GRAPH
each vertex will on average have (3k-1)*p + (m-3)*k*q edges
so total expected number of edges, E = m*k*((3k-1)*p + (m-3)*k*q)
fixing E=40016,m=100,k=16, we have 47*p+1552*q = 40016/1600
also fixing p=0.2, we get q= 0.010058
 """
#graph = make_ring_group_graph(100,16,0.2,0.010058)
#make_Brilliance_distrib_image(graph, 'ring grp graph 100 16 0.2 0.010058')

"""
PA GRAPH
m = total vertices, n = out degree
expected number of edges, E = n(n-1)+(m-n)*n
simplify: E = n*(m-1)
fixing m = 1559
get n = 26
"""
graph = make_PA_Graph(1559,26)
make_Brilliance_distrib_image(graph,'PA graph 1559 26')
    


