#pragma once
#include <chrono>
#include <iostream>

using namespace std::chrono;

#define START_TIMER(BLOCK_NAME) \
    auto BLOCK_NAME##_start = high_resolution_clock::now();

#define END_TIMER(BLOCK_NAME) \
    auto BLOCK_NAME##_end = high_resolution_clock::now(); \
    duration<double> BLOCK_NAME##_elapsed = BLOCK_NAME##_end - BLOCK_NAME##_start; \
    if(rank==0) \
        std::cout << #BLOCK_NAME " time = " << BLOCK_NAME##_elapsed.count() << " sec" << std::endl;
