import numpy as np
import pandas as pd
import gudhi as gd
import sys
import matplotlib.pyplot as plt
import networkx as nx
from ripser import Rips
import pickle
import os
import persim
from sklearn.manifold import TSNE
from matplotlib.pyplot import figure
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)


# Perform DFS on the graph and set the departure time of all
# vertices of the graph
def DFS(graph, v, discovered, departure, time):
    discovered[v] = True
 
    # set arrival time – not needed
    # time = time + 1
 
    # do for every edge (v, u)
    for (u, w) in graph.adjList[v]:
        # if `u` is not yet discovered
        if not discovered[u]:
            time = DFS(graph, u, discovered, departure, time)
 
    # ready to backtrack
    # set departure time of vertex `v`
    departure[time] = v
    time = time + 1
 
    return time
 
def findLongestDistance(graph, source, n):
    # `departure` stores vertex number having its departure
    # time equal to the index of it
    departure = [-1] * n
 
    # to keep track of whether a vertex is discovered or not
    discovered = [False] * n
    time = 0
 
    # perform DFS on all undiscovered vertices
    for i in range(n):
        if not discovered[i]:
            time = DFS(graph, i, discovered, departure, time)
 
    cost = [sys.maxsize] * n
    cost[source] = 0
 
    # Process the vertices in topological order, i.e., in order
    # of their decreasing departure time in DFS
    added = []
    
    for i in reversed(range(n)):
     
        # for each vertex in topological order,
        # relax the cost of its adjacent vertices
        v = departure[i]
        # edge from `v` to `u` having weight `w`
        for (u, w) in graph.adjList[v]:
            w = -w     # make edge weight negative
            # if the distance to destination `u` can be shortened by
            # taking edge (v, u), then update cost to the new lower value
            if cost[v] != sys.maxsize and cost[v] + w < cost[u]:
                cost[u] = cost[v] + w
                
            
    
    
    dist = dict()
    for i in range(n):
        dist[i] = {-cost[i]}
        
    return dist

class Graph:
    # Constructor
    def __init__(self, edges, n):
 
        # A list of lists to represent an adjacency list
        self.adjList = [[] for _ in range(n)]
 
        # add edges to the directed graph
        for (source, dest, weight) in edges:
            self.adjList[source].append((dest, weight))

class Topo:
    
    def __init__(self, G, edgs, delay_dict, s_inst, sn_lst=None):
        self.G = G
        self.delay_dict = delay_dict
        self.s_inst = s_inst
        self.sn_lst = sn_lst
        self.Vn = G.number_of_nodes()
        self.n = np.max(edgs.flatten()) + 1
        self.nodelist = [node for node in G]
        self.edgs = edgs
        self.edgs_d = [(lst[0], lst[1], 1) for lst in edgs] 
        
        
    def degree_distribution(self, in_degree=True, log=True):
        if in_degree:
            in_d = list(dict(self.G.in_degree()).values())
            plt.xlabel("degree")
            plt.ylabel("frequency")
            if log:
                plt.scatter(np.log10(list(set(in_d))[1:]), np.log10([in_d.count(val) for val in set(in_d)][1:]))
                plt.title("in_degree")
            else:
                plt.scatter(list(set(in_d)), [in_d.count(val) for val in set(in_d)])
                plt.title("in_degree")

        else:
            out_d = list(dict(self.G.out_degree()).values())
            plt.xlabel("degree")
            plt.ylabel("frequency")
            if log:
                plt.scatter(np.log10(list(set(out_d))[1:]), np.log10([out_d.count(val) for val in set(out_d)][1:]))
                plt.title("out_degree")
                
            else:
                plt.scatter(list(set(out_d)), [out_d.count(val) for val in set(out_d)])
                plt.title("out_degree")

    def global_persistent(self):
        d_lst = []
        for delay in range(max(self.delay_dict.values()) + 1):
            delay_level_lst = []
            for node in self.delay_dict.keys():
                if self.delay_dict[node] == delay:
                    delay_level_lst.append(node)

            if len(delay_level_lst) == 0:
                continue
            d_lst.append(delay_level_lst)
        to_level = len(d_lst) 
        st = gd.SimplexTree()

        for edg in self.edgs:
            st.insert(edg)

        for delay in range(to_level):
            nodes = d_lst[delay]
            for node in nodes:
                st.assign_filtration([node], delay)

        st.extend_filtration()
        dgms = st.extended_persistence()

        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        axs[0,0].scatter([dgms[0][i][1][0] for i in range(len(dgms[0]))], [dgms[0][i][1][1] for i in range(len(dgms[0]))])
        axs[0,0].plot([0, to_level],[0,to_level])
        axs[0,0].set_title("Ordinary PD")
        axs[0,1].scatter([dgms[1][i][1][0] for i in range(len(dgms[1]))], [dgms[1][i][1][1] for i in range(len(dgms[1]))])
        axs[0,1].plot([0,to_level],[0,to_level])
        axs[0,1].set_title("Relative PD")
        axs[1,0].scatter([dgms[2][i][1][0] for i in range(len(dgms[2]))], [dgms[2][i][1][1] for i in range(len(dgms[2]))])
        axs[1,0].plot([0,to_level],[0,to_level])
        axs[1,0].set_title("Extended+ PD")
        axs[1,1].scatter([dgms[3][i][1][0] for i in range(len(dgms[3]))], [dgms[3][i][1][1] for i in range(len(dgms[3]))], color="orange")
        axs[1,1].plot([0,to_level],[0,to_level])
        axs[1,1].set_title("Extended- PD")
        plt.show()
        return dgms
    
    def distance_expansion(self, shortest=True, remove_excep=True):
        if self.sn_lst is None:
            print("The source nodes are not provided, it will be randomly generated.")
            self.sn_lst = np.random.choice(nodelist, size=10, replace=False)
        
        if shortest:
            all_data = []
            for sn in self.sn_lst:
                pre_sub_Vn = 9
                sub_Vn = 0
                r = 1
                r_lst = []
                Vn_lst = []
                while sub_Vn < self.Vn - 1: 
                    sub_g = nx.ego_graph(self.G, sn, radius=r)
                    sub_Vn = sub_g.number_of_nodes()
                    if sub_Vn == pre_sub_Vn:
                        break
                    r_lst.append(r)
                    Vn_lst.append(sub_Vn)
                    r += 1
                    pre_sub_Vn = sub_Vn

                all_data.append((r_lst, Vn_lst))
            
            figure(figsize=(10, 8), dpi=80)
            for tp in all_data:
                r_lst = [0] + tp[0]
                Vn_lst = [0] + tp[1]

                plt.plot(r_lst, Vn_lst)
                plt.legend(self.sn_lst)

            plt.show()
                
            return all_data
        
        else:
            all_data = []
            for sn in self.sn_lst:
                all_dist_raw = []
                n = self.n
                graph = Graph(self.edgs_d, n)
                p_input = [sn]
                for node in p_input: all_dist_raw.append(findLongestDistance(graph, node, n))

                # building the delay based dictionary
                loc_delay_dict = all_dist_raw[0]
                loc_delay_dict = {key: list(loc_delay_dict[key]) for key in loc_delay_dict.keys()}
                for dist in all_dist_raw[1:]:
                    for key in dist.keys(): 
                        val = dist[key]
                        if key in loc_delay_dict.keys():
                            loc_delay_dict[key] = loc_delay_dict[key] + list(val)
                        else:
                            loc_delay_dict[key] = list(val)

                loc_delay_dict = {key: max(loc_delay_dict[key]) for key in loc_delay_dict.keys()}

                d_lst = []
                for delay in range(max(loc_delay_dict.values()) + 1):
                    delay_level_lst = []
                    for node in loc_delay_dict.keys():
                        if loc_delay_dict[node] == delay:
                            delay_level_lst.append(node)

                    if len(delay_level_lst) == 0:
                        continue
                    d_lst.append(delay_level_lst)
                
                r_lst = [idx for idx in range(len(d_lst))]
                l_lst = [len(lst) for lst in d_lst]

                all_data.append((r_lst, l_lst))
            
            figure(figsize=(10, 8), dpi=80)
            final_all_data = []
            for tp in all_data:
                r_lst = tp[0]
                Vn_lst = tp[1]

                cum_Vn = 0
                final_Vn = []
                for Vn in Vn_lst:
                    cum_Vn += Vn
                    final_Vn.append(cum_Vn)
                
                final_all_data.append((r_lst, final_Vn))
                plt.plot(r_lst, final_Vn)
                plt.legend(self.sn_lst)

            
            return final_all_data
    
    def distance_expansion_degrees(self):
        o_degree = []
        i_degree = []
        for sn in self.sn_lst:
            o_degree.append(self.G.out_degree[sn])
            i_degree.append(self.G.in_degree[sn])

        plt.figure(figsize=(10,5))
        plt.bar([str(val) for val in sn_lst], o_degree, align='edge', width=0.3) 
        plt.bar([str(val) for val in sn_lst], i_degree, align='center', width=0.3, color="orange") 
        plt.legend(["out_degree", "in_degree"])
        
        
    def gen_local_pd(self, s_n, radius, k, save_dir, longest=True):
        print("This method will only apply local pd to a sub graph.")
        sub_G = nx.ego_graph(self.G, s_n, radius)
        print(f"Number of nodes in the sub graph: {sub_G.number_of_nodes()}")
        if longest:
            i = 0
            n_nodelist = []
            print(f"Now start generating local pd to the saving directory")

            for s_n in sub_G:
                sub_g = nx.ego_graph(sub_G, s_n, radius=k)
                edgs = list(sub_g.edges)
                if len(edgs) <= 4:
                    continue
                n_nodelist.append(s_n)
                col = []
                row = []
                for edg in edgs:
                    col.append(edg[0])
                    row.append(edg[1])

                p_input = set()
                p_output = set()

                p_input.add(s_n)

                o_d = sub_g.out_degree
                i_d = sub_g.in_degree

                for node in sub_g:
                    if o_d[node] == 0 and i_d[node] == 0:
                        continue

                    if o_d[node] == 0:
                        p_output.add(node)

                    if i_d[node] == 0:
                        p_input.add(node)

                #####################################
                nn = sub_g.number_of_nodes()
                idx_dict = {list(sub_g.nodes)[idx]:idx for idx in range(nn)}
                all_dist_raw = []
                edgs_d = [(idx_dict[lst[0]], idx_dict[lst[1]], 1) for lst in edgs] 
                n = nn
                graph = Graph(edgs_d, n)

                if len(p_input) == 1:
                    loc_delay_dict = findLongestDistance(graph, idx_dict[s_n], n)
                    loc_delay_dict = {key: loc_delay_dict[key].pop() for key in loc_delay_dict.keys()}

                else:
                    for node in p_input: all_dist_raw.append(findLongestDistance(graph, idx_dict[node], n))

                    # building the delay based dictionary
                    loc_delay_dict = all_dist_raw[0]
                    loc_delay_dict = {key: list(loc_delay_dict[key]) for key in loc_delay_dict.keys()}
                    for dist in all_dist_raw[1:]:
                        for key in dist.keys(): 
                            val = dist[key]
                            if key in loc_delay_dict.keys():
                                loc_delay_dict[key] = loc_delay_dict[key] + list(val)
                            else:
                                loc_delay_dict[key] = list(val)

                    loc_delay_dict = {key: max(loc_delay_dict[key]) for key in loc_delay_dict.keys()}
                ##########################################

                d_lst = []
                for delay in range(max(loc_delay_dict.values()) + 1):
                    delay_level_lst = []
                    for node in loc_delay_dict.keys():
                        if loc_delay_dict[node] == delay:
                            delay_level_lst.append(node)

                    if len(delay_level_lst) == 0:
                        continue
                    d_lst.append(delay_level_lst)
                to_level = len(d_lst) 
                st = gd.SimplexTree()

                edgs = [(idx_dict[edg[0]], idx_dict[edg[1]]) for edg in edgs]
                for edg in edgs:
                    st.insert(edg)


                for delay in range(to_level):
                    nodes = d_lst[delay]
                    for node in nodes:
                        st.assign_filtration([node], delay)

                st.extend_filtration()
                dgms = st.extended_persistence()
                np.save(f"{save_dir}/pd_{s_n}.npy", dgms)
                i += 1

                if i//1000 > 0 and i%1000 == 0:
                    print("now processing node number: " + str(i))
        
        else:
            i = 0
            n_nodelist = []
            print(f"Now start generating local pd to the saving directory")
            for s_n in sub_G:
                sub_g = nx.ego_graph(sub_G, s_n, radius=k)
                edgs = list(sub_g.edges)
                if len(edgs) <= 4:
                    continue
                n_nodelist.append(s_n)
                col = []
                row = []
                for edg in edgs:
                    col.append(edg[0])
                    row.append(edg[1])

                p_input = set()
                p_output = set()

                p_input.add(s_n)

                o_d = sub_g.out_degree
                i_d = sub_g.in_degree

                for node in sub_g:
                    if o_d[node] == 0 and i_d[node] == 0:
                        continue

                    if o_d[node] == 0:
                        p_output.add(node)

                    if i_d[node] == 0:
                        p_input.add(node)

                # building the delay based dictionary
                all_raw = []
                for v in p_input:
                    all_raw.append(nx.shortest_path_length(sub_g, v))    

                loc_delay_dict = all_raw[0]
                loc_delay_dict = {key: [loc_delay_dict[key]] for key in loc_delay_dict.keys()}
                for dist in all_raw[1:]:
                    for key in dist.keys(): 
                        val = dist[key]
                        if key in loc_delay_dict.keys():
                            loc_delay_dict[key] = loc_delay_dict[key] + [val]
                        else:
                            loc_delay_dict[key] = [val]

                loc_delay_dict = {key: min(loc_delay_dict[key]) for key in loc_delay_dict.keys()}


                d_lst = []
                for delay in range(max(loc_delay_dict.values()) + 1):
                    delay_level_lst = []
                    for node in loc_delay_dict.keys():
                        if loc_delay_dict[node] == delay:
                            delay_level_lst.append(node)

                    if len(delay_level_lst) == 0:
                        continue
                    d_lst.append(delay_level_lst)
                to_level = len(d_lst) 
                st = gd.SimplexTree()

                edgs = [(edg[0], edg[1]) for edg in edgs]
                for edg in edgs:
                    st.insert(edg)


                for delay in range(to_level):
                    nodes = d_lst[delay]
                    for node in nodes:
                        st.assign_filtration([node], delay)

                st.extend_filtration()
                dgms = st.extended_persistence()
                np.save(f"{save_dir}/s_pd_{s_n}.npy", dgms)
                i += 1

                if i//1000 > 0 and i%1000 == 0:
                    print("now processing node number: " + str(i))
        
        print("all finish")
        np.save("nodelist.npy", n_nodelist)
        
        
    def vis_local_graph(self, s_n, radius, save_path="graph.png"):
        print("Attention: Please run gen_local_pd before visualize the sub graph.")
        sub_G = nx.ego_graph(self.G, s_n, radius)
        color = []
        i = 0 
        for node in sub_G:
            if node in self.s_inst:
                color.append("red")
                i += 1
            else:
                color.append("blue")

        print(f"The number of sequential instances in the given sub graph is: {i}")
        print(f"Total size of the sub graph is: {sub_G.number_of_nodes()}")
        
        nodelist = [node for node in sub_G.nodes]
        
        sub_delay = {node:self.delay_dict[node] for node in nodelist}

        d_lst = []
        for delay in range(max(sub_delay.values()) + 1):
            delay_level_lst = []
            for node in sub_delay.keys():
                if sub_delay[node] == delay:
                    delay_level_lst.append(node)

            if len(delay_level_lst) == 0:
                continue
            d_lst.append(delay_level_lst)
            
            
        mid = np.max([len(lst) for lst in d_lst])/2
        pos = {}
        for delay in range(len(d_lst)):
            lst = d_lst[delay]
            for in_pos in range(len(lst)):
                node = lst[in_pos]
                pos[node] = ((mid - len(lst)) + in_pos*2, (len(d_lst) - delay)*2)
                
                
        labels = {}
        for lst in d_lst:
            for node in lst:
                if node in self.s_inst:
                    val = str(sub_delay[node]) + " " + str(node)
                    labels[node] = val
                    
                    
        f = plt.figure(figsize=(10,50), dpi=100) 
        nx.draw(sub_G, node_size=20, pos=pos, width=0.3, arrowsize=5, node_color=color, labels=labels, font_size=8)
        f.savefig(save_path)
        
    def process_sym(self, w_d_lst_path, save_path):
        w_d_lst = np.load(w_d_lst_path, allow_pickle=True)
        N = len(w_d_lst)
        b_symm = np.random.random(size=(N,N))
        for idx in range(len(w_d_lst)):
            lst = w_d_lst[idx]
            
            for in_idx in range(len(lst)):
                val = lst[in_idx]
                b_symm[idx][(len(w_d_lst)-len(lst))+in_idx] = val
                b_symm[(len(w_d_lst)-len(lst))+in_idx][idx] = val

        np.save(save_path, b_symm)
    
    
    def clustering_analysis(self, sym_path, th, save_path):
        X = np.load(sym_path)
        N = X.shape[0]
        
        to_fil = np.mean(np.percentile(X, th, axis=1))
        
        fil_idx = []
        for idx in range(len(X)):
            if np.mean(X[idx]) <= to_fil:
                fil_idx.append(idx)
                
        X = X[fil_idx, :][:, fil_idx]
        
        fig, axes = plt.subplots(2, 1, figsize=(25, 35), dpi=100, gridspec_kw={'height_ratios': [1, 5]})
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')
        model = model.fit(X)
        plot = plot_dendrogram(model, ax=axes[0])
        idx_lst = [int(idx) for idx in (plot.get('ivl'))]
        X_fil = X[idx_lst, :][:, idx_lst]
        im = axes[1].matshow(X_fil, cmap=plt.cm.hot_r)
        print("Generating Hierarchical Clustering Dendrogram and Similarity Matrix")
        fig.savefig("b_symm_0.png")
        
        return idx_lst

    def show_matrix(self, sym_path, pre_compute, normalize_path=None):
        X = np.load(sym_path)
        N = X.shape[0]
        X = X[pre_compute, :][:, pre_compute]
        
        if normalize_path is not None:
            o_lst = np.load(normalize_path)
            for i in range(len(X)):
                for j in range(len(X)):
                    distance = X[i][j]
                    size_diff = abs(o_lst[i] - o_lst[j])
                    if size_diff == 0:
                        continue
                    X[i][j] = distance/size_diff
        
        plt.matshow(X, cmap=plt.cm.hot_r)
        plt.colorbar()
        return X
    
    
    def idx_dict(nodelist):
        idx_dict = dict()
        for idx in range(len(nodelist)):
            idx_dict[nodelist[idx]] = idx
            
        return idx_dict