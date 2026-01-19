#pragma once

#include <string>
#include <vector>

std::vector<float> readEigenvalues(const std::string& filepath);
std::vector<float> readCoarseEigenvalues(const std::string& eigenpack_location);
std::vector<float> readFineEigenvalues(const std::string& eigenpack_location);
