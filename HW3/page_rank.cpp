#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <float.h>
#include <utility>
#include <vector>
using namespace std;

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
	vector<double> pre(numNodes), cur(numNodes);
  for (int i = 0; i < numNodes; ++i)
  {
    //solution[i] = equal_prob;
    pre[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
	
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }
	*/
	
  	vector<const Vertex*> incomingBegin(numNodes);
  	vector<int> incomingSize(numNodes), outgoingSize(numNodes);
	for(int i=0; i<numNodes; i++){
		incomingBegin[i] = incoming_begin(g, i);
		incomingSize[i] = incoming_size(g, i);
		outgoingSize[i] = outgoing_size(g, i);
	}
  
  	double global_diff = DBL_MAX;
	while(global_diff >= convergence){
		global_diff = 0.0;
		for(int i = 0; i < numNodes; i++){
			cur[i] = 0.0;

			const Vertex* vj = incomingBegin[i];
			int len = incomingSize[i];
			for(int j=0; j<len; j++){
				cur[i] += pre[vj[j]] /  outgoingSize[vj[j]];
			}

       		cur[i] = (damping * cur[i]) + (1.0-damping) / numNodes;
			for(int j=0; j<numNodes; j++){
				if(outgoingSize[j] != 0) continue;
				cur[i] += damping * pre[j] / numNodes;
			}
			global_diff += abs(cur[i] - pre[i]);
		}
		cur.swap(pre);
	}

	for (int i = 0; i < numNodes; ++i)
	{
		solution[i] = pre[i];
	}
}
