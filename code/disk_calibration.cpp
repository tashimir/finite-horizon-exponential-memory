// Simulate retained-edge costs for fixed memory policies on uniform disk trajectories.
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

struct Policy { double alpha, delta, factor; };
struct Point { double x, y; };

uint64_t mix64(uint64_t z) {
    z += UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
uint64_t trajectory_seed(uint64_t master, uint64_t n, uint64_t i) {
    return mix64(mix64(mix64(master) + n) + i);
}
double uniform53(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}
Point disk_point(std::mt19937_64& rng) {
    const double radius = std::sqrt(uniform53(rng));
    const double angle = 6.283185307179586476925286766559 * uniform53(rng);
    return {radius * std::cos(angle), radius * std::sin(angle)};
}

int main(int argc, char** argv) {
    try {
        std::signal(SIGUSR1, SIG_IGN);
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            args[argv[i]] = argv[i + 1];
        }
        const uint64_t n = std::stoull(args.at("--N"));
        const uint64_t start = std::stoull(args.at("--start"));
        const uint64_t count = std::stoull(args.at("--count"));
        const uint64_t master = std::stoull(args.at("--seed"));
        const int threads = std::stoi(args.at("--threads"));
        if (!n || !count || threads < 1) throw std::runtime_error("Invalid size");
        std::ifstream parameters(args.at("--parameters"));
        if (!parameters) throw std::runtime_error("Cannot read parameters");
        std::vector<Policy> policies;
        double alpha, delta;
        while (parameters >> alpha >> delta) {
            if (!(alpha > 0 && delta >= 0 && delta <= 1))
                throw std::runtime_error("Invalid policy");
            policies.push_back({alpha, delta, std::pow(delta, alpha) +
                (delta == 1 ? 0.0 : std::exp(alpha * std::log1p(-delta)))});
        }
        if (policies.empty()) throw std::runtime_error("Empty policy set");
        const size_t p = policies.size();
        std::vector<Point> supplied;
        if (args.count("--points")) {
            supplied.resize(count * (n + 1));
            std::ifstream input(args.at("--points"), std::ios::binary);
            input.read(reinterpret_cast<char*>(supplied.data()), supplied.size()*sizeof(Point));
            if (!input || input.peek() != EOF) throw std::runtime_error("Point data size mismatch");
        }
        std::vector<Point> dump;
        if (args.count("--dump-points")) dump.resize(count * (n + 1));
        std::vector<double> output(count * p);
        omp_set_num_threads(threads);
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < count; ++i) {
            std::mt19937_64 rng(trajectory_seed(master, n, start + i));
            auto point = [&](uint64_t step) {
                Point result = supplied.empty() ? disk_point(rng) : supplied[i*(n+1)+step];
                if (!dump.empty()) dump[i*(n+1)+step] = result;
                return result;
            };
            Point initial = point(0);
            std::vector<double> x(p, initial.x), y(p, initial.y), costs(p, 0), correction(p, 0);
            for (uint64_t step = 1; step <= n; ++step) {
                const Point input = point(step);
                for (size_t j = 0; j < p; ++j) {
                    const Policy& policy = policies[j];
                    const double dx = input.x - x[j], dy = input.y - y[j];
                    const double squared = dx*dx + dy*dy;
                    const double distance_power = policy.alpha == 1.0 ? std::sqrt(squared) :
                        (policy.alpha == 2.0 ? squared : std::pow(squared, policy.alpha/2));
                    const double increment = policy.factor * distance_power;
                    const double adjusted = increment - correction[j];
                    const double total = costs[j] + adjusted;
                    correction[j] = (total - costs[j]) - adjusted;
                    costs[j] = total;
                    x[j] += policy.delta * dx;
                    y[j] += policy.delta * dy;
                }
            }
            std::copy(costs.begin(), costs.end(), output.begin() + i*p);
        }
        for (double value : output)
            if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid output cost");
        std::ofstream file(args.at("--output"), std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.data()), output.size()*sizeof(double));
        file.close();
        if (!file) throw std::runtime_error("Cost output failed");
        if (!dump.empty()) {
            std::ofstream points(args.at("--dump-points"), std::ios::binary);
            points.write(reinterpret_cast<const char*>(dump.data()), dump.size()*sizeof(Point));
            points.close();
            if (!points) throw std::runtime_error("Point output failed");
        }
        std::cout << "{\"N\":" << n << ",\"start\":" << start << ",\"count\":" << count
                  << ",\"policies\":" << p << ",\"threads\":" << threads << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
