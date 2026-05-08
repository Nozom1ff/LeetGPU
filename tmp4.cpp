#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace std;
#define INT8_MAX 127
#define INT8_MIN -128
vector<float> nums({5.47, 6.08, -7.59, 0, -1.95, -4.57, 10.08});

int main() {
  float a_max = *max_element(nums.begin(), nums.end());
  float a_min = *min_element(nums.begin(), nums.end());
  float s = (a_max - a_min) / (INT8_MAX - INT8_MIN);
  for (const auto v : nums) {
    int8_t x = static_cast<int8_t>((v - a_min) / s + INT8_MIN);
    cout << static_cast<int>(x) << endl;
  }
  return 0;
}
