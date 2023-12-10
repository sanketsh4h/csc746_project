#include <iostream>
#include <omp.h>

const int GAP_PENALTY = -2;
const int MATCH_SCORE = 2;
const int MISMATCH_SCORE = -1;

int desiredNumThreads = std::stoi(argv[1]);

// Set the number of threads at runtime
omp_set_num_threads(desiredNumThreads);


void smithWaterman(char* sequence1, char* sequence2, int* scoreMatrix, int width, int height) {
#pragma omp parallel
    for (int i = 1; i <= height; ++i) {
#pragma omp parallel
        for (int j = 1; j <= width; ++j) {
            int match = (sequence1[i - 1] == sequence2[j - 1]) ? MATCH_SCORE : MISMATCH_SCORE;

            int diagonal = scoreMatrix[(i - 1) * (width + 1) + (j - 1)] + match;
            int up = scoreMatrix[(i - 1) * (width + 1) + j] + GAP_PENALTY;
            int left = scoreMatrix[i * (width + 1) + (j - 1)] + GAP_PENALTY;

            int maxScore = std::max(0, std::max(diagonal, std::max(up, left)));

            scoreMatrix[i * (width + 1) + j] = maxScore;
        }
    }
}

int main() {
    const int width = 25;
    const int height = 25;
    char sequence1[height+1] = "TGATATAGCATTAGTCAGCGGAGAA";
    char sequence2[width+1] = "GCATGTATTCCTGCATGTATACAAC";
    int scoreMatrix[(width + 1) * (height + 1)];

    // Initialize score matrix
    for (int i = 0; i <= height; ++i) {
        for (int j = 0; j <= width; ++j) {
            scoreMatrix[i * (width + 1) + j] = 0;
        }
    }

    // Perform Smith-Waterman on CPU
    smithWaterman(sequence1, sequence2, scoreMatrix, width, height);

    // Output the resulting score matrix
    for (int i = 0; i <= height; ++i) {
        for (int j = 0; j <= width; ++j) {
            std::cout << scoreMatrix[i * (width + 1) + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}