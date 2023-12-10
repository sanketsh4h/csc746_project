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

void smithWatermanParallel(char* sequence1, char* sequence2, int* scoreMatrix, int width, int height) {
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

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    smithWatermanKernel<<<gridDim, blockDim>>>(d_sequence1, d_sequence2, d_scoreMatrix, width, height);

    // Copy results back to host
    cudaMemcpy(scoreMatrix, d_scoreMatrix, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_sequence1);
    cudaFree(d_sequence2);
    cudaFree(d_scoreMatrix);
}

int main() {
    const int width = 8;
    const int height = 8;
    char sequence1[height+1] = "AGTACGTA";
    char sequence2[width+1] = "TATAGCGA";
    int scoreMatrix[width * height];

    // Initialize score matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            scoreMatrix[i * width + j] = 0;
        }
    }

    // Perform parallel Smith-Waterman
    smithWatermanParallel(sequence1, sequence2, scoreMatrix, width, height);

    // Output the resulting score matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << scoreMatrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
