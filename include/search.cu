#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// Definir delimitadores simples para el conteo de palabras
#define IS_DELIMITER(c) (c == ' ' || c == '\n' || c == '\t' || c == ',' || c == '.' || c == ';')

// Kernel de CUDA: Se ejecuta en la GPU
__global__ void countWordsKernel(const char* d_text, int length, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        // Lógica para detectar inicio de palabra:
        // 1. El caracter actual NO es un delimitador
        // 2. El caracter ANTERIOR SÍ era un delimitador (o es el inicio del archivo)
        bool is_char = !IS_DELIMITER(d_text[idx]);
        bool prev_is_delim = (idx == 0) ? true : IS_DELIMITER(d_text[idx - 1]);

        if (is_char && prev_is_delim) {
            atomicAdd(d_count, 1);
        }
    }
}

// Función auxiliar para leer archivo a memoria
std::vector<char> readFileToBuffer(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: No se pudo abrir el archivo " << filepath << std::endl;
        return {};
    }

    std::streamsize sizes = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(sizes);
    if (file.read(buffer.data(), sizes)) {
        return buffer;
    }
    return {};
}

int main() {
    // 1. Definir archivo a procesar
    std::string filepath = "/home/jparrales/paralela/Motor-Hibrido-MPI-OpenMP-CUDA/data/access.log.txt";
    
    std::cout << "Leyendo archivo: " << filepath << "..." << std::endl;

    // 2. Leer archivo en CPU
    std::vector<char> h_text = readFileToBuffer(filepath);
    int text_len = h_text.size();

    if (text_len == 0) {
        std::cerr << "El archivo esta vacio o no existe." << std::endl;
        return 1;
    }

    std::cout << "Tamano del archivo: " << text_len << " bytes." << std::endl;

    // 3. Reservar memoria en GPU
    char* d_text;
    int* d_count;
    int h_count = 0; // Resultado en CPU

    cudaMalloc((void**)&d_text, text_len * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    // 4. Copiar datos de CPU a GPU
    cudaMemcpy(d_text, h_text.data(), text_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice); // Inicializar contador en 0

    // 5. Configurar ejecución del kernel
    int blockSize = 256;
    int numBlocks = (text_len + blockSize - 1) / blockSize;

    std::cout << "Lanzando kernel CUDA con " << numBlocks << " bloques de " << blockSize << " hilos..." << std::endl;
    
    // Llamada al kernel
    countWordsKernel<<<numBlocks, blockSize>>>(d_text, text_len, d_count);
    
    // Sincronizar para asegurar que termine
    cudaDeviceSynchronize();

    // 6. Copiar resultado de vuelta
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "========================================" << std::endl;
    std::cout << "Total de palabras contadas (GPU): " << h_count << std::endl;
    std::cout << "========================================" << std::endl;

    // 7. Liberar memoria
    cudaFree(d_text);
    cudaFree(d_count);

    return 0;
}