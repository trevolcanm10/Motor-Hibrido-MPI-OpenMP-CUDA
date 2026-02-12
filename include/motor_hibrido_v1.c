#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <omp.h>
#include <mpi.h>
#include <limits.h>
#include <unistd.h>

#define MAX_LINE 1000

// Función para contar palabras de manera thread-safe
int contar_palabras(char *linea) {
    int count = 0;
    char *token;
    char *saveptr;  // necesario para strtok_r
    token = strtok_r(linea, " \t\n.,;:!?\"()", &saveptr);
    while (token != NULL) {
        count++;
        token = strtok_r(NULL, " \t\n.,;:!?\"()", &saveptr);
    }
    return count;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    char hostname[256];
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    gethostname(hostname, sizeof(hostname));

    char *folder = "/home/ubuntu/PCyP/textos";
    DIR *d = opendir(folder);
    if (!d) {
        if (world_rank == 0) perror("No se pudo abrir la carpeta");
        MPI_Finalize();
        return 1;
    }

    // Guardar todos los nombres de archivos .log
    struct dirent *dir;
    char **archivos = NULL;
    int num_archivos = 0;

    while ((dir = readdir(d)) != NULL) {
        if (strstr(dir->d_name, ".log")) {
            archivos = realloc(archivos, (num_archivos + 1) * sizeof(char *));
            archivos[num_archivos] = malloc(PATH_MAX);
            snprintf(archivos[num_archivos], PATH_MAX, "%s/%s", folder, dir->d_name);
            num_archivos++;
        }
    }
    closedir(d);

    if (num_archivos == 0) {
        if (world_rank == 0) printf("No se encontraron archivos .log en %s\n", folder);
        MPI_Finalize();
        return 1;
    }

    // Información inicial de cada nodo
    #pragma omp critical
    printf("[%s MPI %d] Procesando %d archivos con %d hilos OpenMP\n",
           hostname, world_rank, (num_archivos + world_size - 1) / world_size, omp_get_max_threads());

    int palabras_totales_nodo = 0;

    // Repartir archivos entre nodos MPI
    for (int i = world_rank; i < num_archivos; i += world_size) {

        int palabras_archivo = 0;

        #pragma omp parallel
        {
            int local_count = 0;
            FILE *local_fp = fopen(archivos[i], "r");
            if (!local_fp) {
                #pragma omp critical
                printf("[%s MPI %d Thread %d] No se pudo abrir %s\n",
                       hostname, world_rank, omp_get_thread_num(), archivos[i]);
            } else {
                char linea[MAX_LINE];
                while (fgets(linea, MAX_LINE, local_fp)) {
                    local_count += contar_palabras(linea);
                }
                fclose(local_fp);
            }

            #pragma omp atomic
            palabras_archivo += local_count;

            #pragma omp critical
            printf("[%s MPI %d Thread %d] Contó %d palabras en %s\n",
                   hostname, world_rank, omp_get_thread_num(), local_count, archivos[i]);
        }

        palabras_totales_nodo += palabras_archivo;

        #pragma omp critical
        printf("[%s MPI %d] Total palabras en %s: %d\n",
               hostname, world_rank, archivos[i], palabras_archivo);
    }

    // Mostrar resumen por nodo
    printf("[%s MPI %d] TOTAL palabras procesadas por este nodo: %d\n",
           hostname, world_rank, palabras_totales_nodo);

    // Calcular total global usando MPI_Reduce
    int palabras_totales_global = 0;
    MPI_Reduce(&palabras_totales_nodo, &palabras_totales_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("\n===== RESUMEN GLOBAL =====\n");
        printf("Total de archivos: %d\n", num_archivos);
        printf("Total de palabras en todos los nodos: %d\n", palabras_totales_global);
        printf("===========================\n");
    }

    // Liberar memoria
    for (int i = 0; i < num_archivos; i++) free(archivos[i]);
    free(archivos);

    MPI_Finalize();
    return 0;
}
