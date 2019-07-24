# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:40:32 2018

@author: gvch48
"""

import random
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt

def make_random_graph(num_nodes, prob):
    """Returns a dictionary to a random graph with the specified number of nodes
    and edge probability.  The nodes of the graph are numbered 0 to
    num_nodes - 1.
    """
    #initialize empty graph
    random_graph = {}
    for vertex in range(num_nodes):
        random_graph[vertex] = []
    #consider each vertex
    for vertex in range(num_nodes):
        for neighbour in range(num_nodes):
            if vertex < neighbour:
                random_number = random.random()
                if random_number < prob and neighbour not in random_graph[vertex]:
                    random_graph[vertex] += [neighbour]
                    random_graph[neighbour] += [vertex]
    return random_graph
    

def make_ring_group_graph(m,k,p,q):     #assumes p > q
    #initialize empty grqph
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
    
def make_PA_graph(total_nodes, out_degree):         #taken from lecture 3
    """creates a PA_Graph on total_nodes where each vertex is iteratively
    connected to a number of existing nodes equal to out_degree"""
    #initialize graph by creating complete graph and trial object
    PA_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)
    for vertex in range(out_degree, total_nodes):
        PA_graph[vertex] = trial.run_trial(out_degree)
        for neighbour in PA_graph[vertex]:                  #makes graph undirected
            PA_graph[neighbour].add(vertex)
    for vertex in PA_graph:
        PA_graph[vertex] = list(PA_graph[vertex])
    return PA_graph

#not needed anymore as shuffle happens within the algorithms
def shuffle_neighbours(graph):          #shuffles neighbours so as to not affect search time
    for vertex in graph:
        random.shuffle(graph[vertex])
        
def search_time_random_graph(graph, source, target):
    """Since all edges are random the only info the id gives is whether the 
    neighbour is the target or not, so the algorithm runs through each neighbour
    and checks if they are the target. If none of them are then the algorithm
    jumps to a new vertex at random. (It would be quicker to choose a vertex to 
    jump to that we havent visited before, but the vertex doesnt have that 
    information available"""
    searches = 0
    currentvertex = source
    nextvertex = -1
    while currentvertex != target:
  #      print(currentvertex, graph[currentvertex])          #for debugging
        random.shuffle(graph[currentvertex])
        for neighbour in graph[currentvertex]:
            searches += 1
            if neighbour == target:
                nextvertex = target
                break
        if nextvertex != target:
            nextvertex = random.choice(graph[currentvertex])
        currentvertex = nextvertex
    return searches


def average_search_time_random_graph(graph, samples):
    sum_search_time = 0
    sampledPairs = []
    for _ in itertools.repeat(None, samples):
        while True:
            vertexA = random.randint(0,len(graph)-1)
            vertexB = random.randint(0,len(graph)-1)
            if vertexA != vertexB and (vertexA, vertexB) not in sampledPairs:   #dont sample same pair multiple times
                break
        sampledPairs.append((vertexA,vertexB))
        sum_search_time += search_time_random_graph(graph, vertexA, vertexB)
    return int(sum_search_time/samples)



"""OLD VERSION OF FUNCTION
def choose_best_next_neighbour(m,neighbourIDList, targetID):
    bestchoice = (-1,-1)
    mindist = m
    nextbest = (-1,-1)
    secondbestdist = m
    for neighbourID in neighbourIDList:
        dist = min_dist_between_groups(m,neighbourID[1],targetID[1])
        if dist < mindist:
            mindist = dist
            bestchoice = neighbourID
        elif dist < secondbestdist:
            secondbestdist = dist
            nextbest = neighbourID
        if mindist == 0:
            break
    if nextbest != (-1,-1):
        return random.choice([bestchoice,nextbest])
    else:
        return bestchoice     
"""

def choose_best_next_neighbour(m, neighbourIDList, targetID):
    """
    old version had problem of creating deadlocks where a few vertices kept moving between each other.
    this version orders the neighbours in terms of likeliness to be close to target, but wont always pick
    the best option. it picks the best option 50% of the time, second best 25%, 3rd 12.5% etc.
    This introduces some suboptimality i order to prevent the worst case scenario of deadlock
    """
    orderedBest = []
    for neighbourID in neighbourIDList:
        dist = min_dist_between_groups(m,neighbourID[1],targetID[1])
        orderedBest.append((neighbourID[0],neighbourID[1],dist))
    orderedBest = sorted(orderedBest,key=itemgetter(2))
    for neighbour in orderedBest:
        if random.random() < 0.5:
            return(neighbour[0],neighbour[1])
        
            
def min_dist_between_groups(m, grp1, grp2):
    if grp1 > grp2:
        temp = grp1
        grp1 = grp2
        grp2 = temp    
    dist1 = grp2 - grp1
    dist2 = m+grp1-grp2
    return min(dist1,dist2)


def search_time_ring_group_graph(graph, source, target, m, k, q):
    searches = 0
    currentID = (source, source%m)
    nextID = (-1,-1)
    targetID = (target, target%m)
    while currentID[0] != targetID[0]:
  #      print(str(currentID)+'______'+str(targetID))            #for debugging
        random.shuffle(graph[currentID[0]])                     #help prevent deadlocks
        neighbourIDList = []
        if currentID[1]-targetID[1] in [-1,0,1,m-1]:    #adjacent or same group. high chance of target being neighbour. check every neighbour
            for neighbour in graph[currentID[0]]:
                searches += 1
                neighbourID = (neighbour, neighbour%m)
                neighbourIDList.append(neighbourID)
                if neighbourID[0] == targetID[0]:
                    nextID = targetID
                    break
            if nextID[0] != targetID[0]:
                nextID = choose_best_next_neighbour(m,neighbourIDList, targetID)
        else:
            goodChoiceFound = False
            for neighbour in graph[currentID[0]]:
                searches += 1
                neighbourID = (neighbour, neighbour%m)
                neighbourIDList.append(neighbourID)
                unchecked = len(graph[currentID[0]])-len(neighbourIDList) #number of neighbours not yet queried
                neighbourDist = min_dist_between_groups(m,neighbourID[1],targetID[1])   #distance from neighbour to target
                prob = 1-(1-q)**(unchecked*k*neighbourDist) #prob of one of the unchecked neighbours being closer to target than currently queried one
                if 0.5 > prob:      #if lower than 50% chance of finding better neighbour to jump to
                    nextID = neighbourID
                    goodChoiceFound = True
                    break
            if not goodChoiceFound:
                nextID = choose_best_next_neighbour(m, neighbourIDList, targetID)
        currentID = nextID
    return searches
                
def average_search_time_ring_group_graph(graph, m, k, q, samples):
    sum_search_time = 0
    sampledPairs = []
    for _ in itertools.repeat(None, samples):
        while True:
            vertexA = random.randint(0,len(graph)-1)
            vertexB = random.randint(0,len(graph)-1)
            if vertexA != vertexB and (vertexA, vertexB) not in sampledPairs:   #dont sample same pair multiple times
                break
        sampledPairs.append((vertexA,vertexB))
#        print(vertexA, vertexB)                           #for debugging
        sum_search_time += search_time_ring_group_graph(graph, vertexA, vertexB, m, k, q)
    return int(sum_search_time/samples)



def search_time_PA_graph(graph, out_deg, source, target): #assumes out degree >= 5
    searches = 0
    currentID = source+1    #id goes from 1 to n, graph goes from 0 to n-1
    nextID = -1
    targetID = target+1
    while currentID != targetID:
 #       print(currentID)                            #for debugging
        random.shuffle(graph[currentID-1])
        neighbourIDList = []
        if currentID < out_deg + 2:      #part of initial complete graph
            for neighbour in graph[currentID-1]:
                searches += 1
                neighbourID = neighbour+1
                neighbourIDList.append(neighbourID)
                if neighbourID == targetID:
                    nextID = targetID
                    break
            else:
   #             neighbourIDList.sort()  #not needed since we know current vertex is connected to vertices 1 to m
                while True:
                    nextID = random.randint(1,out_deg+2)   #picks another vertex from original complete graph
                    if nextID != currentID:                 #makes sure nextID isnt current one
                        break
        else:
            for neighbour in graph[currentID-1]:
                searches += 1
                neighbourID = neighbour +1
                neighbourIDList.append(neighbourID)
                if neighbourID < out_deg+2:
                    nextID = neighbourID
                    break
            else:
                neighbourIDList.sort()
                nextID = random.choice(neighbourIDList[:5])
        currentID = nextID
    return searches

def average_search_time_PA_graph(graph, out_deg, samples):
    sum_search_time = 0
    sampledPairs = []
    for _ in itertools.repeat(None, samples):
        while True:
            vertexA = random.randint(0,len(graph)-1)
            vertexB = random.randint(0,len(graph)-1)
            if vertexA != vertexB and (vertexA, vertexB) not in sampledPairs:   #dont sample same pair multiple times
                break
        sampledPairs.append((vertexA,vertexB))
        sum_search_time += search_time_PA_graph(graph, out_deg, vertexA, vertexB)
    return int(sum_search_time/samples)

def plot_search_time_random(num_nodes,prob,num_samples,num_graphs):
    search_time_distrib={}
    for x in range(1,num_graphs+1):
        graph = make_random_graph(num_nodes,prob)
        search_time = average_search_time_random_graph(graph,num_samples)
        if search_time in search_time_distrib:
            search_time_distrib[search_time] += 1
        else:
            search_time_distrib[search_time] = 1
            
        if x%10 == 0:       #update plot every 10 graphs so i can leave running for a long time, stop at any time and still have data
            xdata = []
            ydata = []
            for value in search_time_distrib:
                xdata += [value]
                ydata += [search_time_distrib[value]]
            
            #plot degree distribution
            plt.clf() #clears plot
            plt.xlabel('Search Time')
            plt.ylabel('Number of instances')
            plt.title('Search Time Distribution of '+str(x)+' Random Graphs')
            plt.semilogy(xdata, ydata, marker='.', linestyle='None', color='b')
            plt.savefig('Search Time Distribution of '+str(num_graphs)+' Random Graphs '+str(num_nodes)+' '+str(prob)+' '+str(num_samples)+' .png',dpi=300,bbox_inches='tight')
    #recreate plot after all graphs done
    xdata = []
    ydata = []
    for value in search_time_distrib:
        xdata += [value]
        ydata += [search_time_distrib[value]]
    
    #plot degree distribution
    plt.clf() #clears plot
    plt.xlabel('Search Time')
    plt.ylabel('Number of instances')
    plt.title('Search Time Distribution of '+str(x)+' Random Graphs')
    plt.semilogy(xdata, ydata, marker='.', linestyle='None', color='b')
    plt.savefig('Search Time Distribution of '+str(num_graphs)+' Random Graphs '+str(num_nodes)+' '+str(prob)+' '+str(num_samples)+' .png',dpi=300,bbox_inches='tight')

def plot_search_time_ring_group(m,k,p,q,num_samples,num_graphs):
    search_time_distrib={}
    for x in range(1,num_graphs+1):
        graph = make_ring_group_graph(m,k,p,q)
        search_time = average_search_time_ring_group_graph(graph, m, k, q, num_samples)
        if search_time in search_time_distrib:
            search_time_distrib[search_time] += 1
        else:
            search_time_distrib[search_time] = 1
            
        if x%10 == 0:       #update plot every 10 graphs so i can leave running for a long time, stop at any time and still have data
            xdata = []
            ydata = []
            for value in search_time_distrib:
                xdata += [value]
                ydata += [search_time_distrib[value]]
            
            #plot degree distribution
            plt.clf() #clears plot
            plt.xlabel('Search Time')
            plt.ylabel('Number of instances')
            plt.title('Search Time Distribution of '+str(x)+' Ring Group Graphs')
            plt.semilogy(xdata, ydata, marker='.', linestyle='None', color='b')
            plt.savefig('Search Time Distribution of '+str(num_graphs)+' Ring Group Graphs '+str(m)+' '+str(k)+' '+str(p)+' '+str(q)+' '+str(num_samples)+' .png',dpi=300,bbox_inches='tight')
    #recreate plot after all graphs done
    xdata = []
    ydata = []
    for value in search_time_distrib:
        xdata += [value]
        ydata += [search_time_distrib[value]]
    
    #plot degree distribution
    plt.clf() #clears plot
    plt.xlabel('Search Time')
    plt.ylabel('Number of instances')
    plt.title('Search Time Distribution of '+str(x)+' Ring Group Graphs')
    plt.semilogy(xdata, ydata, marker='.', linestyle='None', color='b')
    plt.savefig('Search Time Distribution of '+str(num_graphs)+' Ring Group Graphs '+str(m)+' '+str(k)+' '+str(p)+' '+str(q)+' '+str(num_samples)+' .png',dpi=300,bbox_inches='tight')


def plot_search_time_PA(total_nodes,out_degree,num_samples,num_graphs):
    search_time_distrib={}
    for x in range(1,num_graphs+1):
        graph = make_PA_graph(total_nodes, out_degree)
        search_time = average_search_time_PA_graph(graph,out_degree,num_samples)
        if search_time in search_time_distrib:
            search_time_distrib[search_time] += 1
        else:
            search_time_distrib[search_time] = 1
            
        if x%10 == 0:       #update plot every 10 graphs so i can leave running for a long time, stop at any time and still have data
            xdata = []
            ydata = []
            for value in search_time_distrib:
                xdata += [value]
                ydata += [search_time_distrib[value]]
            
            #plot degree distribution
            plt.clf() #clears plot
            plt.xlabel('Search Time')
            plt.ylabel('Number of instances')
            plt.title('Search Time Distribution of '+str(x)+' PA Graphs')
            plt.semilogy(xdata, ydata, marker='.', linestyle='None', color='b')
            plt.savefig('Search Time Distribution of '+str(num_graphs)+' PA Graphs '+str(total_nodes)+' '+str(out_degree)+' '+str(num_samples)+' .png',dpi=300,bbox_inches='tight')
    #recreate plot after all graphs done
    xdata = []
    ydata = []
    for value in search_time_distrib:
        xdata += [value]
        ydata += [search_time_distrib[value]]
    
    #plot degree distribution
    plt.clf() #clears plot
    plt.xlabel('Search Time')
    plt.ylabel('Number of instances')
    plt.title('Search Time Distribution of '+str(x)+' PA Graphs')
    plt.semilogy(xdata, ydata, marker='.', linestyle='None', color='b')
    plt.savefig('Search Time Distribution of '+str(num_graphs)+' PA Graphs '+str(total_nodes)+' '+str(out_degree)+' '+str(num_samples)+' .png',dpi=300,bbox_inches='tight')


"""
####CODE TO PRODUCE IMAGES####
"""

#plot_search_time_random(1000,0.05,1000,3000)

#plot_search_time_ring_group(100,16,0.5,0.01,1000,3000)
    
#plot_search_time_PA(1000,20,2500,3000)
