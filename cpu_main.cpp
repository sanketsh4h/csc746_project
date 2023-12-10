#include <iostream>
#include <omp.h>
#include <chrono>

const int GAP_PENALTY = -2;
const int MATCH_SCORE = 2;
const int MISMATCH_SCORE = -1;

void smithWaterman(char* sequence1, char* sequence2, int* scoreMatrix, int width, int height) {
    printf("Test 3\n");
#pragma omp parallel

    // Adjust loop conditions and array indices
for (int i = 1; i <= height; ++i) {
    #pragma omp parallel
        for (int j = 1; j <= width; ++j) {
            // Adjust array indices to start from 0
            int match = (sequence1[i - 1] == sequence2[j - 1]) ? MATCH_SCORE : MISMATCH_SCORE;

            // Adjust array indices for scoreMatrix
            int diagonal = scoreMatrix[(i - 1) * width + (j - 1)] + match;
            int up = scoreMatrix[(i - 1) * width + j] + GAP_PENALTY;
            int left = scoreMatrix[i * width + (j - 1)] + GAP_PENALTY;

            // Adjust array indices for scoreMatrix
            int maxScore = std::max(0, std::max(diagonal, std::max(up, left)));

            // Adjust array indices for scoreMatrix
            scoreMatrix[i * width + j] = maxScore;
        }
    }

}

int main(int argc, char* argv[]) {
    // if (argc != 2) {
    //     std::cerr << "Usage: " << argv[0] << " <numThreads>" << std::endl;
    //     return 1;
    // }
    printf("Test 0-1\n");
    // Convert the command-line argument to an integer
    int desiredNumThreads = std::stoi(argv[1]);
    printf("Test 0-2\n");
    // Set the number of threads at runtime
    omp_set_num_threads(desiredNumThreads);
    // Disable nested parallelism
    // omp_set_nested(0);
    printf("Test 0-3\n");
    const int width = 625;
    const int height = 625;
    char sequence1[height + 1] = "GCCTCAACTTACTTTTGCGCTTAGCTGGAGTTGCCCCTCTTATTACTACCGGATGATCGCAACATAACCGGTCGTCAAATAAGGAGGTTGATATCCCGACAAAGTGGAGCATTAGCGAGGATCGACATTTTTCAGGGAAGAATACAATCTTTAGCCTCAGTATCTATTCCGAGCTTTGTCCGTGGCGCCCGTTCTAATAGCTAAGCACTCATTCTAGGCCAAAATCGGGTTTGGGGAATCTTCACTCTCTCTGAGTCGCAGGATGCGCGCCTCGTACTCAGATCTTTATATTGGTCTTAATTTATGTAGCCAAATGGGGCACGGATACGGTTAGGACTGGCAGGAGTAGTATTGTCCTGTGCCTAGACCCTTGCTTGGACACTCACTATTATGGATGTGTACCGCAATTCACCCAGCGGAAAAGTAAAGCACAAAAGGGAGGATACCTCCCCGTGAACGACCCGTTGCAGTGTGACTCGTCAAACCAGCCGCCAGTGTCATGGATAAGCCAGCACAACGACTACTTGAAAACGGATCTCTGCGACCGTGAAGCTCGGTGCGGCTGATCCGATTTCAGTCCCACATGACACCAATAGATCGAGGTCATATAGCCGTCTATAAGTTC";
    char sequence2[width + 1] = "CCGCTCTTTTCTGTTCACTAAGTGGGTTCCGGGCGCTGACGGTTCCGGAATCTGTTTCGTAATTAGAGCTTGCAAGCTCGAAGAAAGGTAATCGAAGGGGGCTACGATTCAGCCCCTACACGGGATTGGCTGGACAACAAACGAGCCAGCTATGGAAGCCCCGCAGGATAGCGGTTCCGGATGGTAAGGCTTTACTGTCAGAAGCAAGCAGGGGGAGGGTAATGCATCTGAGGGGTTGATGTATGGTGGTGACGTTATGTGGTACCTGTTAGAGTCCTCGATCAGACCTATGTGAGTTCTACGATGGTTGTCTTTTTTTCAGTTACCGTACTTTGAATAGCGAAAGACGTTGTTTCCGAACCATGCTGGAGGTAATGGTTAAAGACAACCTGTATTTCTTTCAAATTATGTTTAACTTAATCTCCTAGTAGAGCCGATATGTGTAAGGGTTATCCCATTCGCATGTATCGATGGACCTTAACACCGACTACTTGCGCCCTAAAGTAGTTGTACGCCAACGTATGGAGCGCATGACAACGGATAAATGGTTTTGATATGCTGGACTCGGTGGGCCAAATGACGATTGACGCCATGCAGCCAGGGGGGATGCGCTATACGCCGGGGG";
    int* scoreMatrix = new int[width * height];

    printf("Test 1\n");
    // Initialize score matrix
    for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
        scoreMatrix[i * width + j] = 0;
    }
}

    printf("Test 2\n");
    auto start = std::chrono::high_resolution_clock::now();
    // Perform Smith-Waterman on CPU
    smithWaterman(sequence1, sequence2, scoreMatrix, width, height);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Duration: " << duration.count() << " microseconds" << std::endl;
    printf("Test 7\n");
    // Output the resulting score matrix
    // for (int i = 0; i <= height; ++i) {
    //     for (int j = 0; j <= width; ++j) {
    //         std::cout << scoreMatrix[i * (width + 1) + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    delete[] scoreMatrix;

    return 0;
}
