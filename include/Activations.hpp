#pragma once

#include <memory>
#include <string>

class IActivation {
public:
    virtual ~IActivation() = default;
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual std::string name() const = 0;
};

class Sigmoid : public IActivation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
    std::string name() const override;
};

class ReLU : public IActivation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
    std::string name() const override;
};

class Tanh : public IActivation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
    std::string name() const override;
};

std::shared_ptr<IActivation> createActivation(const std::string& typeName);
