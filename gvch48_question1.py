# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:36:22 2018

@author: gvch48
"""


import random
import queue
import matplotlib.pyplot as plt


def make_ring_group_graph(m,k,p,q):
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

        
def compute_degrees(graph):
    """Takes a graph and computes the degrees for the nodes in the
    graph. Returns a dictionary with the same set of keys (nodes) and the
    values are the degrees."""
    #initialize degrees dictionary with zero values for all vertices
    degree = {}
    for vertex in range( len(graph) ):
        degree[vertex] = 0
    #consider each vertex
    for vertex in range( len(graph) ):
        for neighbour in graph[vertex]:
            degree[vertex] += 1
    return degree

def degree_distribution(graph):
    """Takes a graph and computes the unnormalized distribution of the
    degrees of the graph.  Returns a dictionary whose keys correspond to
    degrees of nodes in the graph and values are the number of nodes with
    that degree. Degrees with no corresponding nodes in the graph are not
    included in the dictionary."""
    #find in_degrees
    degree = compute_degrees(graph)
    #initialize dictionary for degree distribution
    degree_distribution = {}
    #consider each vertex
    for vertex in degree:
        #update degree_distribution
        if degree[vertex] in degree_distribution:
            degree_distribution[degree[vertex]] += 1
        else:
            degree_distribution[degree[vertex]] = 1
    return degree_distribution

def max_dist(graph, source):
    """finds the distance (the length of the shortest path) from the source to
    every other vertex in the same component using breadth-first search, and
    returns the value of the largest distance found"""
    q = queue.Queue()
    found = {}
    distance = {}
    for vertex in range(len(graph)): 
        found[vertex] = 0
        distance[vertex] = -1
    max_distance = 0
    found[source] = 1
    distance[source] = 0
    q.put(source)
    while q.empty() == False:
        current = q.get()
        for neighbour in graph[current]:
            if found[neighbour] == 0:
                found[neighbour] = 1
                distance[neighbour] = distance[current] + 1
                max_distance = distance[neighbour]
                q.put(neighbour)
    return max_distance

def diameter(graph):
    """returns the diameter of a graph, by finding, for each vertex, the maximum
    length of a shortest path starting at that vertex, and returning the overall
    maximum"""
    distances = []
    for vertex in graph:                            #look at each vertex
        distances += [max_dist(graph, vertex)]      #find the distance to the farthest other vertex
    return max(distances)                           #return the maximum value found

def local_clustering_coefficient(graph, vertex):
    """returns ratio of edges to possible edges in neighbourhood of vertex"""
    edges = 0
    #look at each pair of neighbours of vertex
    for neighbour1 in graph[vertex]:
        for neighbour2 in graph[vertex]:
    #look at whether neighbour pair are joined by an edge
            if (neighbour1 < neighbour2) and (neighbour2 in graph[neighbour1]):
                edges += 1
    #divide number of edges found by number of pairs considered
    return 2*edges/(len(graph[vertex]) * (len(graph[vertex]) - 1))

def average_clustering_coefficient(graph):
    sum = 0
    for vertex in range(len(graph)):
        sum += local_clustering_coefficient(graph, vertex)
    return sum/len(graph)


def make_distrib_image(m,k,p):
#    m= int(input("Number of groups: "))
#    k= int(input("Number of vertices per group: "))
#    p= float(input("Probability of edge between vertices in same or adjacent group: "))
#    print("Computing...")
    graph = make_ring_group_graph(m,k,p,0.5-p)
    distrib = degree_distribution(graph)
    
    normalized_distribution = {}
    for degree in distrib:
        normalized_distribution[degree] = distrib[degree] / (m*k)
    
    #create arrays for plotting
    xdata = []
    ydata = []
    for degree in distrib:
        xdata += [degree]
        ydata += [normalized_distribution[degree]]
    
    #plot degree distribution
    plt.clf() #clears plot
    plt.xlabel('Degree')
    plt.ylabel('Normalized Rate')
    plt.title('Normalized Degree Distribution of Graph (m='+str(m)+', k='+str(k)+', p='+str(p)+')')
    plt.loglog(xdata, ydata, marker='.', linestyle='None', color='b')
    plt.savefig('normalized degree distrib '+str(m)+' '+str(k) + ' '+str(p)+' .png',dpi=300)
   
def make_diameter_image(m,k,q,minp,maxp,mult_p, num_of_graphs):
    #points is number of probabilities to compute diameter for. evenly distributed between minp and maxp
    xdata = []
    ydata = []
    
    p = minp
    while p <= maxp:
        print(p)
        graph = make_ring_group_graph(m,k,p,q)
        xdata += [p]
        ydata += [diameter(graph)/(m*k)]
        p *= mult_p
        
    if num_of_graphs > 1:
        for graph_tracker in range(1,num_of_graphs):
            print('graph: '+str(graph_tracker+1))
            graphYdata = []               
            for p in xdata:
                graph = make_ring_group_graph(m,k,p,q)
                graphYdata += [diameter(graph)/(m*k)]
            for yValue in range(len(graphYdata)):
                ydata[yValue] += graphYdata[yValue]
        ydata = [y/num_of_graphs for y in ydata]
    
    plt.clf() #clears plot
    plt.xlabel('Probability')
    plt.xlim(max(minp-0.05,0),maxp+0.05)
    plt.ylabel('Normalised Diameter')
    plt.title('Average Normalised Diameter over '+str(num_of_graphs)+' Graphs (m='+str(m)+', k='+str(k)+', q='+str(q)+')')
    plt.plot(xdata,ydata,marker='.', linestyle='None', color='b')
    plt.savefig('normalised diameter vs probability '+str(m)+' '+str(k)+' '+str(q)+' '+str(num_of_graphs)+' .png',dpi=300)
    
def make_clustering_image(m,k,q,minp,maxp,precision, num_of_graphs):
    #points is number of probabilities to compute diameter for. evenly distributed between minp and maxp
    xdata = []
    ydata = []
    p = minp
    while p < maxp:
        graph = make_ring_group_graph(m,k,p,q)
        avrg_cluster_coeff = average_clustering_coefficient(graph)
        xdata += [p]
        ydata += [avrg_cluster_coeff]
        p += precision
    xdata += maxp
    ydata += [average_clustering_coefficient(make_ring_group_graph(m,k,maxp,q))]
    
    if num_of_graphs > 1:
        for graph_tracker in range(1,num_of_graphs):
            print('graph: '+str(graph_tracker+1))
            graphYdata = []               
            for p in xdata:
                graph = make_ring_group_graph(m,k,p,q)
                avrg_cluster_coeff = average_clustering_coefficient(graph)
                graphYdata += [avrg_cluster_coeff]
            for yValue in range(len(graphYdata)):
                ydata[yValue] += graphYdata[yValue]
        ydata = [y/num_of_graphs for y in ydata]

    plt.clf() #clears plot    

    plt.xlabel('Probability')
    plt.xlim(minp-precision,maxp+precision)
    plt.ylabel('Average Clustering Coefficient')
    plt.title('Average Clustering coeff over '+str(num_of_graphs)+' Graphs (m='+str(m)+', k='+str(k)+', q='+str(q)+')')
    plt.plot(xdata,ydata,marker='.', linestyle='None', color='b')
    plt.savefig('average clustering vs probability '+str(m)+' '+str(k)+' '+str(q)+' '+str(num_of_graphs)+' .png',dpi=300,bbox_inches='tight')


####TEST CODE####
#graph = make_ring_group_graph(4,4,0.3,0.2)
print("running: make_diameter_image")
make_diameter_image(100,10,1/1000,1/200,1,1.2,5)
