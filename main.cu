#include <iostream>
#include <cuda_runtime.h>

const int GAP_PENALTY = -2;
const int MATCH_SCORE = 2;
const int MISMATCH_SCORE = -1;

__global__ void smithWatermanKernel(char* sequence1, char* sequence2, int* scoreMatrix, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width) {
        int match = (sequence1[i] == sequence2[j]) ? MATCH_SCORE : MISMATCH_SCORE;

        int diagonal = scoreMatrix[(i - 1) * width + (j - 1)] + match;
        int up = scoreMatrix[(i - 1) * width + j] + GAP_PENALTY;
        int left = scoreMatrix[i * width + (j - 1)] + GAP_PENALTY;

        int maxScore = max(0, max(diagonal, max(up, left)));

        scoreMatrix[i * width + j] = maxScore;
    }
}

void smithWatermanParallel(char* sequence1, char* sequence2, int* scoreMatrix, int width, int height, dim3 blockDim, dim3 gridDim) {
    char* d_sequence1, *d_sequence2;
    int* d_scoreMatrix;

    // Allocate device memory
    cudaMalloc((void**)&d_sequence1, height * sizeof(char));
    cudaMalloc((void**)&d_sequence2, width * sizeof(char));
    cudaMalloc((void**)&d_scoreMatrix, width * height * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_sequence1, sequence1, height * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sequence2, sequence2, width * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scoreMatrix, scoreMatrix, width * height * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    smithWatermanKernel<<<gridDim, blockDim>>>(d_sequence1, d_sequence2, d_scoreMatrix, width, height);

    // Copy results back to host
    cudaMemcpy(scoreMatrix, d_scoreMatrix, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_sequence1);
    cudaFree(d_sequence2);
    cudaFree(d_scoreMatrix);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <blockDimX> <blockDimY>" << std::endl;
        return 1;
    }

    const int width = 8;
    const int height = 8;
    char sequence1[height + 1] = "AGTACGTA";
    char sequence2[width + 1] = "TATAGCGA";
    int scoreMatrix[width * height];

    // Initialize score matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            scoreMatrix[i * width + j] = 0;
        }
    }

    // Extract block dimensions from command-line arguments
    int blockDimX = std::stoi(argv[1]);
    int blockDimY = std::stoi(argv[2]);

    // Define grid dimensions based on width, height, and block dimensions
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((width + blockDimX - 1) / blockDimX, (height + blockDimY - 1) / blockDimY);

    // Perform parallel Smith-Waterman
    smithWatermanParallel(sequence1, sequence2, scoreMatrix, width, height, blockDim, gridDim);

    // Output the resulting score matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << scoreMatrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
