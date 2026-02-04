// include/Timer.h（新建文件）
#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <iostream>
#include <fstream>

class Timer {
public:
    // 开始计时
    void Start(const std::string& tag) {
        mTag = tag;
        mStart = std::chrono::high_resolution_clock::now();
    }

    // 结束计时并返回耗时（毫秒），同时打印到控制台
    double End(bool print = true) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - mStart;
        double cost = duration.count();
        
        if (print) {
            std::cout << "[TIMER] " << mTag << " cost: " << cost << " ms" << std::endl;
        }
        
        // 可选：将结果写入文件（追加模式）
        static std::ofstream log_file("orbslam_timing_delect.log", std::ios::app);
        if (log_file.is_open()) {
            log_file << mTag << "," << cost << "\n";
            log_file.flush();
        }
        
        return cost;
    }

private:
    std::string mTag;
    std::chrono::high_resolution_clock::time_point mStart;
};

#endif // TIMER_H

