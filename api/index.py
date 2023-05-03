from flask import Flask, request, jsonify
import heapq
import networkx as nx
from queue import Queue
from typing_extensions import NewType
from collections import defaultdict
import numpy as np

class AStarGraph:
    def __init__(self, node_golds, node_obstcls, node_empty):
        self.edges = defaultdict(list)
        self.weights = {}
        self.node_golds = node_golds
        self.node_obstcls = node_obstcls
        self.node_empty = node_empty

    def add_edge(self, from_node, to_node):
        counter = 0
        while counter < (len(self.node_golds) + len(self.node_empty) + len(self.node_obstcls)):
            parent = from_node
            # f_n_cost=[]
            # g_n_cost=[]
            # node=[]
            # path=[]
            g_n_cost = []
            for i, j, k in self.get_edges_from_node(parent):
                g_n_cost.append(k)
            # first_cost_node=g_n_cost[0]
            # in this loop we iterate in
            # for i in range(g_n_cost):
            #   j=i
            #   for j in range(g_n_cost):
            #     if(g_n_cost[i]==g_n_cost[j] or g_n_cost.all()==1000):
            #       to_node=parent
            if to_node in self.node_golds:
                self.edges[from_node].append(to_node)
                self.weights[(from_node, to_node)] = 1
                counter += 1
            elif to_node in self.node_empty:
                self.edges[from_node].append(to_node)
                self.weights[(from_node, to_node)] = 10
                counter += 1
            else:
                self.edges[from_node].append(to_node)
                self.weights[(from_node, to_node)] = 1000000
                counter += 1
            for i in range(len(g_n_cost)):
                j = i
                for j in range(len(g_n_cost)):
                    if (g_n_cost[i] == g_n_cost[j] or all(cst == 1000 for cst in (g_n_cost))):
                        to_node = parent

    def add_undirected_edge(self, node1, node2):
        self.add_edge(node1, node2)
        self.add_edge(node2, node1)

    # return edges as alist of tuples
    def get_edges_from_node(self, start_node):
        edges = []
        for neighbor in self.find_neighbors(start_node):
            edge_weight = self.weights[(start_node, neighbor)]
            edges.append((start_node, neighbor, edge_weight))
            # [(from,to,weight),(from,to,weight),(from,to,weight),(from,to,weight)]
        return edges

    def find_neighbors(self, node):
        return self.edges[node]

    def heuristic(self, node, goal):
        """
        Returns the MST heuristic value for A* algorithm.
        """
        mst_cost = 0
        visited = set()
        # The tuple contains two values: the first is 0, which represents the initial cost or distance to reach node. The second value, node,
        # is an object or identifier that represents a node in a graph or some other data structure.
        # ------------------------
        # a heap is often used to maintain a priority queue of nodes to be explored, with the node with the lowest cost or priority being removed first
        # -----------------------
        heap = [(0, node)]

        while heap:
            (cost, n) = heapq.heappop(heap)
            if n not in visited:
                visited.add(n)
                mst_cost += cost
                for neighbor in self.edges[n]:
                    heapq.heappush(heap, (self.weights[(n, neighbor)], neighbor))

        return mst_cost

    def a_star_mst(self, start, goal):
        # Create a priority queue to store nodes to be explored
        # each element in this heap queue(periority queue) has tuple (path_cost+h(node,goal),path_cost,node,path_list[])
        frontier = [(0 + self.heuristic(start, goal), 0, start, [])]
        # Create a set to store explored nodes//visited nodes
        explored = set()

        # Loop until the frontier is empty
        while frontier:
            # Get the node with the smallest total cost
            (f_cost, g_cost, current_node, path) = heapq.heappop(frontier)

            # If the current node is the goal, return the path
            if current_node == goal:
                path.append(current_node)
                return (path, g_cost)  # g_cost weights

            # Add the current node to the explored set
            explored.add(current_node)

            # Loop through the neighbors of the current node
            for neighbor in self.find_neighbors(current_node):

                # If the neighbor has already been explored, skip it
                if neighbor in explored:
                    continue

                # Calculate the cost to the neighbor node
                new_cost = g_cost + self.weights[(current_node, neighbor)]

                # Calculate the heuristic cost for the neighbor node
                h_cost = self.heuristic(neighbor, goal)

                # Add the neighbor node to the frontier
                heapq.heappush(frontier, (new_cost + h_cost, new_cost, neighbor, path + [current_node]))

        # If the goal is not found, return None
        return None


class AStarModGraph:
    def __init__(self, node_golds, node_obstcls, node_empty, explored=set()):
        self.counter = len(node_golds)
        self.explored = explored
        self.edges = defaultdict(list)
        self.weights = {}
        self.node_golds = node_golds
        self.node_obstcls = node_obstcls
        self.node_empty = node_empty
        self.path = list()
        self.cost = 0

    def add_edge(self, from_node, to_node):
        if to_node in self.node_golds:
            self.edges[from_node].append(to_node)
            self.weights[(from_node, to_node)] = 1

        elif to_node in self.node_empty:
            self.edges[from_node].append(to_node)
            self.weights[(from_node, to_node)] = 10

        elif to_node in self.node_obstcls:
            self.edges[from_node].append(to_node)
            self.weights[(from_node, to_node)] = 1000000

    def add_undirected_edge(self, node1, node2):
        self.add_edge(node1, node2)
        self.add_edge(node2, node1)

    # return edges as alist of tuples
    def get_edges_from_node(self, start_node):
        edges = []
        for neighbor in self.find_neighbors(start_node):
            edge_weight = self.weights[(start_node, neighbor)]
            edges.append((start_node, neighbor, edge_weight))
            # [(from,to,weight),(from,to,weight),(from,to,weight),(from,to,weight)]
        return edges

    def find_neighbors(self, node):
        return self.edges[node]

    def heuristic(self, node, goal, counter=0, visited=set()):
        neighbors = self.find_neighbors(node)
        if goal in neighbors:
            return counter + self.weights[(node, goal)]
        else:
            if len(neighbors) > 1:
                neighbors = [n for n in neighbors if n not in visited and n is not None]
            if not neighbors:
                return 0
            costs = [self.weights[(node, n)] for n in neighbors]
            neighbors = neighbors[costs.index(sorted(list(set(costs)))[0])]
            return self.heuristic(neighbors, goal, counter + self.weights[(node, neighbors)],
                                  visited=visited.union({neighbors}))
            # return min([self.heuristic(n, goal,counter=counter+self.weights[(node, n)], visited=visited.union({n})) for n in neighbors])

    def searchGold(self, start, path=list(), cost=0):
        # print('path  ' , path)
        if start in self.node_golds:
            return path + [start], cost
        neighbors = self.find_neighbors(start)
        for neighbor in neighbors:
            if neighbor in self.node_golds:
                return path + [start, neighbor], cost + self.heuristic(start, neighbor)
        if len(neighbors) > 1:
            neighbors = [n for n in neighbors if n not in path and n is not None]
            # print('neighbors   ',neighbors)
            if not neighbors:
                return [], 0
            costs = [self.heuristic(start, n) for n in neighbors]
            neighbors = neighbors[costs.index(sorted(list(set(costs)))[0])]
            # ne = [[i, j] for i,j in zip(neighbors, costs)]
            # ne.sort(key = lambda t: t[1])
            # neighbors = ne[0][0] #[nei[0] for nei in ne[:2]]
        # print('neighbors   ',neighbors)
        paths = [self.searchGold(n, path + [start], cost + self.heuristic(start, n)) for n in neighbors]
        paths = [p for p in paths if p is not None]
        paths.sort(key=lambda t: t[1])
        # print('paths  ' , paths)
        if paths:
            return paths[0]
        else:
            return [], 0

    def searchGoal(self, start, goal, path=list(), cost=0):
        # print('path  ' , path)
        if start == goal:
            return path + [start], cost
        neighbors = self.find_neighbors(start)
        for neighbor in neighbors:
            if neighbor in self.node_golds:
                return path + [start, neighbor], cost + self.heuristic(start, neighbor)
        if len(neighbors) > 1:
            neighbors = [n for n in neighbors if n not in path and n is not None]
            # print('neighbors   ',neighbors)
            if not neighbors:
                return [], 0
            costs = [self.heuristic(start, n) for n in neighbors]
            neighbors = neighbors[costs.index(sorted(list(set(costs)))[0])]
        # print('neighbors   ',neighbors)
        paths = [self.searchGoal(n, goal, path + [start], cost + self.heuristic(start, n)) for n in neighbors]
        paths = [p for p in paths if p is not None]
        paths.sort(key=lambda t: t[1])
        if paths:
            return paths[0]
            # else:
            return

    # def a_star_mst(self, start, goal):
    #   while self.node_golds:
    #     nextPath, nextCost = self.searchGold(start)
    #     # print(nextPath)
    #     self.path += nextPath[:-1]
    #     self.cost += nextCost
    #     # print(nextPath[-1])
    #     start = nextPath[-1]
    #     self.node_golds.remove(start)
    #   self.path += start
    #   goalPath, goalCost = self.searchGoal(self.path[-1], goal)
    #   self.path += goalPath[1:]
    #   self.cost += goalCost
    #   return self.path, self.cost

    def a_star_mst(self, start, goal):
        if start == goal and not self.node_golds:
            return
        neighbors = self.find_neighbors(start)
        haveGold = [n for n in neighbors if n in self.node_golds]
        if haveGold:
            self.path.append(haveGold[0])
            self.cost += self.heuristic(start, haveGold[0])
            self.node_golds.remove(haveGold[0])
            self.weights[(start, haveGold[0])] += 10
            self.a_star_mst(haveGold[0], goal)
        elif not self.node_golds and goal in neighbors:
            self.path.append(goal)
            self.cost += self.heuristic(start, goal)
            return
        else:
            fneighbors = [n for n in neighbors if n not in self.path]
            if fneighbors:
                neighbors = fneighbors
            else:
                pass;
            neighborsCost = [self.heuristic(start, n) for n in neighbors]
            minNeighbor = neighbors[
                neighborsCost.index(sorted(list(set(neighborsCost)))[0])]  ## cost.index(sorted(list(set(cost)))[0])
            self.path.append(minNeighbor)
            self.weights[(start, minNeighbor)] += 1
            self.cost += self.heuristic(start, minNeighbor)
            self.a_star_mst(minNeighbor, goal)

G = nx.Graph()
nodes = range(64)
G.add_nodes_from(nodes)

# Add edges to connect the nodes
for i in range(8):
    for j in range(8):
        if i < 7:
            G.add_edge(i * 8 + j, (i + 1) * 8 + j)
        if j < 7:
            G.add_edge(i * 8 + j, i * 8 + j + 1)

def bfs(graph, start,gold,obst):
    parent = {start: None}
    visited = set()  # set of visited nodes
    queue = Queue()  # queue to store nodes to visit
    visited.add(start)
    queue.put(start)
    while not queue.empty():
        node = queue.get()

        if node in gold:  # path found
            gold.remove(node)
            path = []
            while True:
                path.append(node)
                node = parent[node]
                if node == parent[start]:
                    break
            return path[::-1]  # reverse path to get from start to end
        a = list(graph.neighbors(node))
        w = [i for i in a if i in gold]
        neighbors = [i for i in a if i not in gold]
        neighbors.extend(w)
        neighbors = neighbors[::-1]

        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in obst:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.put(neighbor)

    return None

app = Flask(__name__)

@app.route('/')
def my_endpoint():
    return 'A* \nModified A* \nBFS'

@app.route('/astar/', methods=['POST'])
def my_endpoint_astar():
    data = request.get_json();
    golds = data['golds']
    rocks = data['rocks']
    print(golds,rocks)
    empty = []

    i = 0
    for i in range(64):
        empty.insert(i, str(i))

    i = 0
    for i in range(len(golds)):
        empty.remove(golds[i])

    i = 0
    for i in range(len(rocks)):
        empty.remove(rocks[i])

    print(empty)

    graph = AStarGraph(golds, rocks, empty)

    i = 0
    for i in range(63):
        if i % 8 == 7:
            graph.add_undirected_edge(str(i), str(i + 8))
        else:
            if i > 55:
                graph.add_undirected_edge(str(i), str(i + 1))
            else:
                graph.add_undirected_edge(str(i), str(i + 8))
                graph.add_undirected_edge(str(i), str(i + 1))

    path, cost = graph.a_star_mst('0', '63')

    result = {'path': path}
    return jsonify(result)

@app.route('/astarmod/', methods=['POST'])
def my_endpoint_astar_mod():
    data = request.get_json();
    golds = data['golds']
    rocks = data['rocks']
    print(golds,rocks)
    empty = []

    i = 0
    for i in range(64):
        empty.insert(i, str(i))

    i = 0
    for i in range(len(golds)):
        empty.remove(golds[i])

    i = 0
    for i in range(len(rocks)):
        empty.remove(rocks[i])

    print(empty)

    graph = AStarModGraph(golds, rocks, empty)

    i = 0
    for i in range(63):
        if i % 8 == 7:
            graph.add_undirected_edge(str(i), str(i + 8))
        else:
            if i > 55:
                graph.add_undirected_edge(str(i), str(i + 1))
            else:
                graph.add_undirected_edge(str(i), str(i + 8))
                graph.add_undirected_edge(str(i), str(i + 1))

    graph.a_star_mst('0', '63')
    path= list()+['0']
    path +=  graph.path


    result = {'path': path}
    return jsonify(result)

@app.route('/bfs/', methods=['POST'])
def my_endpoint_bfs():
    data = request.get_json();
    input_gold = data['golds']
    input_obst = data['rocks']
    gold = [int(num) for num in input_gold]
    obst = [int(num) for num in input_obst]
    print(gold,obst)

    for i in gold:
        if i in obst:
            result = {'error': f'cell {i} is repeated as a gold and a rock'}
            return jsonify(result)

    if len(gold)==0:
        final = bfs(G, 0, [63], obst)
        final_string = []
        for i in final:
            final_string.append(str(i))
        result = {'path': final_string}
        return jsonify(result)

    path = bfs(G, 0,gold,obst)

    while gold:

        start = path[-1]

        path.extend(bfs(G, start,gold,obst))

    gold = [63]
    print(type(path[-1]))
    path.extend(bfs(G, path[-1],gold,obst))

    final = []
    for i in range(len(path)):
        if path[i] != path[i - 1]:
            final.append(path[i])
    #print(final)
    final_string=[]
    for i in final:
        final_string.append(str(i))
    result = {'path': final_string}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False)

