#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

class RandomStreamGen {
public:
    explicit RandomStreamGen(uint64_t seed = 42) : rng_(seed) {}

    std::string randomToken(std::size_t max_len = 30) {
        static const std::string alphabet =
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-";

        std::uniform_int_distribution<int> len_dist(1, static_cast<int>(max_len));
        std::uniform_int_distribution<int> ch_dist(0, static_cast<int>(alphabet.size() - 1));

        int len = len_dist(rng_);
        std::string s;
        s.reserve(static_cast<std::size_t>(len));
        for (int i = 0; i < len; ++i) {
            s.push_back(alphabet[static_cast<std::size_t>(ch_dist(rng_))]);
        }
        return s;
    }


    std::vector<std::string> generateStream(std::size_t n, double reuse_prob = 0.75, std::size_t max_len = 30) {
        std::vector<std::string> pool;
        pool.reserve(n / 2);

        std::vector<std::string> stream;
        stream.reserve(n);

        std::uniform_real_distribution<double> prob(0.0, 1.0);

        for (std::size_t i = 0; i < n; ++i) {
            bool reuse = (!pool.empty() && prob(rng_) < reuse_prob);
            if (reuse) {
                std::uniform_int_distribution<std::size_t> id_dist(0, pool.size() - 1);
                stream.push_back(pool[id_dist(rng_)]);
            } else {
                std::string token = randomToken(max_len);
                pool.push_back(token);
                stream.push_back(std::move(token));
            }
        }
        return stream;
    }

    static std::vector<std::size_t> buildCheckpoints(std::size_t n, double step_percent) {
        std::vector<std::size_t> cps;
        if (n == 0) return cps;
        std::size_t step = static_cast<std::size_t>(std::llround(n * (step_percent / 100.0)));
        if (step == 0) step = 1;

        for (std::size_t pos = step; pos <= n; pos += step) {
            cps.push_back(pos);
        }
        if (cps.empty() || cps.back() != n) cps.push_back(n);
        return cps;
    }

private:
    std::mt19937_64 rng_;
};

class HashFuncGen {
public:
    struct HashFunc32 {
        uint64_t a = 0;
        uint64_t b = 0;

        static uint64_t fnv1a64(const std::string& s) {
            constexpr uint64_t FNV_OFFSET = 14695981039346656037ull;
            constexpr uint64_t FNV_PRIME  = 1099511628211ull;
            uint64_t h = FNV_OFFSET;
            for (unsigned char c : s) {
                h ^= static_cast<uint64_t>(c);
                h *= FNV_PRIME;
            }
            return h;
        }

        uint32_t operator()(const std::string& s) const {
            uint64_t x = fnv1a64(s);
            uint64_t y = a * x + b;
            return static_cast<uint32_t>(y & 0xFFFFFFFFull);
        }
    };

    static std::vector<HashFunc32> generate(std::size_t count, uint64_t seed = 2026) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());

        std::vector<HashFunc32> funcs;
        funcs.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            HashFunc32 hf;
            hf.a = dist(rng) | 1ull;
            hf.b = dist(rng);
            funcs.push_back(hf);
        }
        return funcs;
    }
};

class HyperLogLog {
public:
    using Hash32 = HashFuncGen::HashFunc32;

    HyperLogLog(uint8_t p, Hash32 h)
        : p_(p), m_(1u << p), hash_(std::move(h)), reg_(m_, 0) {
        if (p_ < 4 || p_ > 20) {
            throw std::invalid_argument("p must be in [4,20]");
        }
    }

    void add(const std::string& x) {
        uint32_t hv = hash_(x);

        uint32_t idx = hv >> (32 - p_);

        uint32_t w = hv << p_;

        uint8_t rank;
        if (w == 0) {
            rank = static_cast<uint8_t>((32 - p_) + 1);
        } else {
            rank = static_cast<uint8_t>(__builtin_clz(w) + 1);
            uint8_t max_rank = static_cast<uint8_t>((32 - p_) + 1);
            if (rank > max_rank) rank = max_rank;
        }

        if (rank > reg_[idx]) reg_[idx] = rank;
    }

    double estimate() const {
        double sum = 0.0;
        uint32_t zeros = 0;

        for (uint8_t r : reg_) {
            sum += std::ldexp(1.0, -static_cast<int>(r));
            if (r == 0) ++zeros;
        }

        double alpha = alphaM(m_);
        double raw = alpha * static_cast<double>(m_) * static_cast<double>(m_) / sum;

        if (raw <= 2.5 * m_ && zeros > 0) {
            raw = m_ * std::log(static_cast<double>(m_) / zeros);
        }

        constexpr double TWO32 = 4294967296.0;
        if (raw > TWO32 / 30.0) {
            raw = -TWO32 * std::log(1.0 - raw / TWO32);
        }

        return raw;
    }

    void clear() {
        std::fill(reg_.begin(), reg_.end(), 0);
    }

    uint32_t m() const { return m_; }
    uint8_t p() const { return p_; }

private:
    static double alphaM(uint32_t m) {
        if (m == 16) return 0.673;
        if (m == 32) return 0.697;
        if (m == 64) return 0.709;
        return 0.7213 / (1.0 + 1.079 / m);
    }

private:
    uint8_t p_;
    uint32_t m_;
    Hash32 hash_;
    std::vector<uint8_t> reg_;
};

struct Aggregated {
    std::vector<double> sum_exact;
    std::vector<double> sum_est;
    std::vector<double> sum_est_sq;
    std::vector<double> sum_sq_err;
    std::size_t streams = 0;
};

int main() {
    const std::size_t NUM_STREAMS = 30;
    const std::size_t STREAM_SIZE = 200000;
    const double STEP_PERCENT = 5.0;
    const double REUSE_PROB = 0.78;
    const uint8_t P = 12;
    const uint64_t STREAM_SEED_BASE = 1000;
    const uint64_t HASH_SEED = 2026;

    auto funcs = HashFuncGen::generate(1, HASH_SEED);
    auto hfunc = funcs.front();

    auto checkpoints = RandomStreamGen::buildCheckpoints(STREAM_SIZE, STEP_PERCENT);
    const std::size_t T = checkpoints.size();

    Aggregated agg;
    agg.sum_exact.assign(T, 0.0);
    agg.sum_est.assign(T, 0.0);
    agg.sum_est_sq.assign(T, 0.0);
    agg.sum_sq_err.assign(T, 0.0);

    std::ofstream details("run_details.csv");
    details << "stream_id,step_id,processed,exact,estimate,abs_err,rel_err\n";
    details << std::fixed << std::setprecision(6);

    for (std::size_t s = 0; s < NUM_STREAMS; ++s) {
        RandomStreamGen gen(STREAM_SEED_BASE + s);
        auto stream = gen.generateStream(STREAM_SIZE, REUSE_PROB, 30);

        HyperLogLog hll(P, hfunc);
        std::unordered_set<std::string> exact_set;
        exact_set.reserve(static_cast<std::size_t>(STREAM_SIZE * 1.3));

        std::size_t step_id = 0;

        for (std::size_t i = 0; i < stream.size(); ++i) {
            hll.add(stream[i]);
            exact_set.insert(stream[i]);

            if (step_id < T && (i + 1) == checkpoints[step_id]) {
                double exact = static_cast<double>(exact_set.size());
                double est = hll.estimate();
                double abs_err = std::fabs(est - exact);
                double rel_err = (exact > 0.0) ? abs_err / exact : 0.0;

                agg.sum_exact[step_id] += exact;
                agg.sum_est[step_id] += est;
                agg.sum_est_sq[step_id] += est * est;
                agg.sum_sq_err[step_id] += (est - exact) * (est - exact);

                details << s << "," << step_id << "," << (i + 1) << ","
                        << exact << "," << est << "," << abs_err << "," << rel_err << "\n";

                ++step_id;
            }
        }
    }

    agg.streams = NUM_STREAMS;
    details.close();

    std::ofstream stats("stats.csv");
    stats << "step_id,processed,mean_exact,mean_est,std_est,bias_rel,rmse_rel,theory_104,theory_13\n";
    stats << std::fixed << std::setprecision(8);

    const double m = static_cast<double>(1u << P);
    const double theory_104 = 1.04 / std::sqrt(m);
    const double theory_13  = 1.30 / std::sqrt(m);

    for (std::size_t j = 0; j < T; ++j) {
        double mean_exact = agg.sum_exact[j] / agg.streams;
        double mean_est = agg.sum_est[j] / agg.streams;
        double var_est = agg.sum_est_sq[j] / agg.streams - mean_est * mean_est;
        if (var_est < 0) var_est = 0;
        double std_est = std::sqrt(var_est);

        double bias_rel = (mean_exact > 0.0) ? (mean_est - mean_exact) / mean_exact : 0.0;
        double rmse = std::sqrt(agg.sum_sq_err[j] / agg.streams);
        double rmse_rel = (mean_exact > 0.0) ? rmse / mean_exact : 0.0;

        stats << j << "," << checkpoints[j] << ","
              << mean_exact << "," << mean_est << ","
              << std_est << "," << bias_rel << "," << rmse_rel << ","
              << theory_104 << "," << theory_13 << "\n";
    }
    stats.close();

    std::cout << "HLL params: p=" << static_cast<int>(P)
              << ", m=" << (1u << P)
              << ", theory RSE ~ 1.04/sqrt(m) = " << theory_104
              << ", relaxed 1.3/sqrt(m) = " << theory_13 << "\n";

    return 0;
}
