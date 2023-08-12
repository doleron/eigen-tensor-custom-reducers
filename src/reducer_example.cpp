#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"
using namespace std::chrono;

inline float basic_sum(const float& a, const float& b) {
    return a + b;
}

template <typename T>
class SumIfReducer
{

public:

    SumIfReducer(std::function<bool(T)> _test_op): test_op(_test_op) {}

    void reduce(const T val, T* acc) const {
        if (this->test_op(val)) {
            *acc = *acc + val;
        }
    }

    template <typename Packet>
    void reducePacket(const Packet& packet, Packet* acc) const {
        (*acc) = Eigen::internal::padd<Packet>(*acc, packet);
    }

    float initialize() const {
        return T(0);
    }

    template <typename Packet>
    Packet initializePacket() const {
        float init = initialize();
        return Eigen::internal::pset1<Packet>(init);
    }

    T finalize(const T acc) const {
        return acc;
    }

    template <typename Packet>
    Packet finalizePacket(const Packet& acc) const {
        return acc;
    }

    template <typename Packet>
    T finalizeBoth(const T acc_val, const Packet& acc_packet) const {
        auto packet = Eigen::internal::predux(acc_packet);
        return this->custom_op(acc_val, packet);
    }

private:
    std::function<bool(T)> test_op;
};

void example1() {

    std::cout << "Example 1\n\n";

    Eigen::Tensor<int, 3> X(2, 2, 3);
    X.setValues({
        {{1, 2, 3},{4, 5, 6}},
        {{7, 8, 9},{10, 11, 12}},
    });

    const auto test = [](int val) {
        return !(val & 0x01);
    };

    SumIfReducer<int> evenReducer(test);

    std::cout << "X is\n\n"<< X << "\n\n";


    Eigen::array<Eigen::Index, 1> dims0({0});
    std::cout << "X.reduce(dims0):\n" << X.reduce(dims0, evenReducer) << "\n\n";

    Eigen::array<Eigen::Index, 2> dims1({0, 1});
    std::cout << "X.reduce(dims1):\n" << X.reduce(dims1, evenReducer) << "\n\n";

    Eigen::array<Eigen::Index, 3> dims2({0, 1, 2});
    std::cout << "X.reduce(dims2):\n" << X.reduce(dims2, evenReducer) << "\n\n";

}

void example2() {

    std::cout << "Example 2\n\n";

    Eigen::Tensor<float, 3> X(2, 2, 3);
    X = X.random() - X.constant(0.5);

    const auto test = [](float val) {
        return val >= 0.f;
    };

    SumIfReducer<float> evenReducer(test);

    std::cout << "X is\n\n"<< X << "\n\n";


    Eigen::array<Eigen::Index, 1> dims0({0});
    std::cout << "X.reduce(dims0):\n" << X.reduce(dims0, evenReducer) << "\n\n";

    Eigen::array<Eigen::Index, 2> dims1({0, 1});
    std::cout << "X.reduce(dims1):\n" << X.reduce(dims1, evenReducer) << "\n\n";

    Eigen::array<Eigen::Index, 3> dims2({0, 1, 2});
    std::cout << "X.reduce(dims2):\n" << X.reduce(dims2, evenReducer) << "\n\n";

}

int main(int, char **)
{
    example1();
    example2();
    return 0;
}