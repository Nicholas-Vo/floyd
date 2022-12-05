#include <stdio.h>
#include <mpi.h>
#include <math.h>

#define N 5 //number of vertices
#define INF 9999 //define infinity

//function prototypes
void floydWarshall(int graph[N][N]);
void printSolution(int dist[N][N]);

int main(int argc, char** argv) {
    int rank, numProcs;
    int graph[N][N] = { {0, 5, INF, 10}, 
                        {INF, 0, 3, INF}, 
                        {INF, INF, 0, 1}, 
                        {INF, INF, INF, 0} 
                      }; 
  
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  
    // Process 0 sends the matrix to all other processes
    if(rank == 0) {
        for(int i = 1; i < numProcs; i++) {
            MPI_Send(&graph, N*N, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&graph, N*N, MPI_INT, 0, 0, MPI_COMM_WORLD, 
                    MPI_STATUS_IGNORE);
    }
  
    // Each process does its own part of the Floyd-Warshall algorithm
    floydWarshall(graph);
  
    // Process 0 receives the result from all other processes
    if(rank == 0) {
        int result[N][N];
        for(int i = 0; i < numProcs; i++) {
            MPI_Recv(&result, N*N, MPI_INT, i, 0, MPI_COMM_WORLD, 
                        MPI_STATUS_IGNORE);
            for(int j = 0; j < N; j++) {
                for(int k = 0; k < N; k++) {
                    graph[j][k] = result[j][k];
                }
            }
        }
        printSolution(graph);
    } else {
        MPI_Send(&graph, N*N, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  
    MPI_Finalize();
    return 0;
}

// A function to implement the Floyd-Warshall algorithm 
void floydWarshall(int graph[N][N]) {
    int dist[N][N], i, j, k; 
  
    /* Initialize the solution matrix same as input graph matrix. 
       Or we can say the initial values of shortest distances 
       are based on shortest paths considering no intermediate 
       vertex. */
    for (i = 0; i < N; i++) 
        for (j = 0; j < N; j++) 
            dist[i][j] = graph[i][j]; 
  
    /* Add all vertices one by one to the set of intermediate 
       vertices. 
      ---> Before start of a iteration, we have shortest 
           distances between all pairs of vertices such that 
           the shortest distances consider only the vertices in 
           set {0, 1, 2, .. k-1} as intermediate vertices. 
      ----> After the end of a iteration, vertex no. k is added 
            to the set of intermediate vertices and the set 
            becomes {0, 1, 2, .. k} */
    for (k = 0; k < N; k++) { 
        // Pick all vertices as source one by one 
        for (i = 0; i < N; i++) { 
            // Pick all vertices as destination for the 
            // above picked source 
            for (j = 0; j < N; j++) { 
                // If vertex k is on the shortest path from 
                // i to j, then update the value of dist[i][j] 
                if (dist[i][k] + dist[k][j] < dist[i][j]) 
                    dist[i][j] = dist[i][k] + dist[k][j]; 
            } 
        } 
    } 
  
    // Print the shortest distance matrix 
    printSolution(dist); 
} 
  
/* A utility function to print solution */
void printSolution(int dist[N][N]) { 
    printf ("The following matrix shows the shortest distances"
            " between every pair of vertices \n"); 
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++) { 
            if (dist[i][j] == INF) 
                printf("%7s", "INF"); 
            else
                printf ("%7d", dist[i][j]); 
        } 
        printf("\n"); 
    } 
}