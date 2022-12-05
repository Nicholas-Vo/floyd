#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define INF INT_MAX
#define V 4

int min(int a, int b) {
    return (a < b) ? a : b;
}

void floyd_warshall(int dist[V][V]) {
    int i, j, k;
    for (k = 0; k < V; k++) {
        for (i = 0; i < V; i++) {
            for (j = 0; j < V; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int dist[V][V];
    int i, j;

    // Initialize MPI
    int my_rank, num_procs;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initialize matrix and calculate distances
    if (my_rank == 0) {
        for (i = 0; i < V; i++) {
            for (j = 0; j < V; j++) {
                if (i == j)
                    dist[i][j] = 0;
                else
                    dist[i][j] = INF;
            }
        }
        dist[0][1] = 1;
        dist[2][3] = 1;
        dist[3][0] = 1;
    }
    int local_data[V][V];
    MPI_Scatter(&dist, V * V / num_procs, MPI_INT, &local_data, V * V / num_procs, MPI_INT, 0, MPI_COMM_WORLD);

    // Run Floyd-Warshall on each processor
    floyd_warshall(local_data);

    // Gather data
    MPI_Gather(&local_data, V * V / num_procs, MPI_INT, &dist, V * V / num_procs, MPI_INT, 0, MPI_COMM_WORLD);

    // Print results
    if (my_rank == 0) {
        printf("Following matrix shows the shortest distances between every pair of vertices \n");
        for (i = 0; i < V; i++) {
            for (j = 0; j < V; j++) {
                if (dist[i][j] == INF)
                    printf("INF ");
                else
                    printf("%d   ", dist[i][j]);
            }
            printf("\n");
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}