using System;
using System.Collections.Generic;
using System.Linq;

namespace KarpathyAI;

/// <summary>
/// Stores a scalar value and its gradient, tracking computation graph for reverse-mode automatic differentiation.
/// Corresponds to the Value class in engine.py.
/// </summary>
public class Value
{
    public double Data { get; set; }
    public double Grad { get; set; }
    public string Op { get; set; }
    public string Label { get; set; }

    internal HashSet<Value> Prev { get; }
    internal Action BackwardAction { get; set; }

    public Value(double data, IEnumerable<Value>? children = null, string op = "", string label = "")
    {
        Data = data;
        Grad = 0.0;
        Op = op;
        Label = label;
        Prev = children != null ? new HashSet<Value>(children) : new HashSet<Value>();
        BackwardAction = () => { };
    }

    public override string ToString() => $"Value(data={Data}, grad={Grad})";

    #region Operator Overloads

    // Implicit conversion allows writing: Value v = 4.0; or passing raw doubles to functions expecting Value.
    public static implicit operator Value(double value) => new(value);

    // Operator overload for +
    public static Value operator +(Value self, Value other)
    {
        var outVal = new Value(self.Data + other.Data, new[] { self, other }, "+");
        outVal.BackwardAction = () =>
        {
            // Accumulate gradient if the same object is used twice (e.g. x = a + a)
            self.Grad += 1.0 * outVal.Grad;
            other.Grad += 1.0 * outVal.Grad;
        };
        return outVal;
    }

    public static Value operator +(Value self, double other) => self + new Value(other);
    public static Value operator +(double self, Value other) => new Value(self) + other;

    // Operator overload for *
    public static Value operator *(Value self, Value other)
    {
        var outVal = new Value(self.Data * other.Data, new[] { self, other }, "*");
        outVal.BackwardAction = () =>
        {
            self.Grad += other.Data * outVal.Grad;
            other.Grad += self.Data * outVal.Grad;
        };
        return outVal;
    }

    public static Value operator *(Value self, double other) => self * new Value(other);
    public static Value operator *(double self, Value other) => new Value(self) * other;

    // Operator overload for negation: -Value
    public static Value operator -(Value self) => self * -1.0;

    // Operator overload for subtraction: Value - other
    public static Value operator -(Value self, Value other) => self + (-other);
    public static Value operator -(Value self, double other) => self - new Value(other);
    public static Value operator -(double self, Value other) => new Value(self) - other;

    // Operator overload for division: Value / other
    public static Value operator /(Value self, Value other) => self * other.Pow(-1.0);
    public static Value operator /(Value self, double other) => self / new Value(other);
    public static Value operator /(double self, Value other) => new Value(self) / other;

    #endregion

    #region Activation and Math Functions

    // Exponential function: e^x
    public Value Exp()
    {
        var x = Data;
        var outVal = new Value(Math.Exp(x), new[] { this }, "exp");
        outVal.BackwardAction = () =>
        {
            // d/dx e^x = e^x
            Grad += outVal.Data * outVal.Grad;
        };
        return outVal;
    }

    // Power function: x^other
    public Value Pow(double other)
    {
        var outVal = new Value(Math.Pow(Data, other), new[] { this }, $"**{other}");
        outVal.BackwardAction = () =>
        {
            // d/dx (a^b) = b * a^(b-1)
            Grad += other * Math.Pow(Data, other - 1.0) * outVal.Grad;
        };
        return outVal;
    }

    // Hyperbolic tangent activation function
    public Value Tanh()
    {
        var x = Data;
        var t = (Math.Exp(2 * x) - 1.0) / (Math.Exp(2 * x) + 1.0);
        var outVal = new Value(t, new[] { this }, "tanh");
        outVal.BackwardAction = () =>
        {
            // d/dx tanh(x) = 1 - tanh^2(x)
            Grad += (1.0 - t * t) * outVal.Grad;
        };
        return outVal;
    }

    #endregion

    #region Backpropagation

    // Backpropagation implementation:
    // Builds the topological graph order, then calls backward from output to inputs.
    public void Backward()
    {
        var topo = new List<Value>();
        var visited = new HashSet<Value>();

        void BuildTopo(Value v)
        {
            if (visited.Add(v))
            {
                foreach (var child in v.Prev)
                {
                    BuildTopo(child);
                }
                topo.Add(v);
            }
        }

        BuildTopo(this);

        Grad = 1.0;
        for (int i = topo.Count - 1; i >= 0; i--)
        {
            topo[i].BackwardAction();
        }
    }

    #endregion
}

/// <summary>
/// A single Neuron with incoming weights and a bias.
/// </summary>
public class Neuron
{
    public List<Value> W { get; }
    public Value B { get; }

    public Neuron(int nin, Random? rng = null)
    {
        rng ??= Random.Shared;
        // Random uniform distribution in [-1.0, 1.0]
        W = Enumerable.Range(0, nin)
                      .Select(_ => new Value(rng.NextDouble() * 2.0 - 1.0))
                      .ToList();
        B = new Value(rng.NextDouble() * 2.0 - 1.0);
    }

    // Forward pass: sum(w * x) + b -> Tanh
    public Value Forward(IReadOnlyList<Value> x)
    {
        if (x.Count != W.Count)
            throw new ArgumentException($"Expected {W.Count} inputs, got {x.Count}");

        Value act = B;
        for (int i = 0; i < W.Count; i++)
        {
            act += W[i] * x[i];
        }
        return act.Tanh();
    }

    public Value Forward(IReadOnlyList<double> x) => Forward(x.Select(v => new Value(v)).ToList());

    public List<Value> Parameters()
    {
        var list = new List<Value>(W.Count + 1);
        list.AddRange(W);
        list.Add(B);
        return list;
    }
}

/// <summary>
/// A single neural network layer containing multiple neurons.
/// </summary>
public class Layer
{
    public List<Neuron> Neurons { get; }

    public Layer(int nin, int nout, Random? rng = null)
    {
        Neurons = Enumerable.Range(0, nout)
                            .Select(_ => new Neuron(nin, rng))
                            .ToList();
    }

    public List<Value> Forward(IReadOnlyList<Value> x)
    {
        return Neurons.Select(n => n.Forward(x)).ToList();
    }

    public List<Value> Forward(IReadOnlyList<double> x)
    {
        return Forward(x.Select(v => new Value(v)).ToList());
    }

    public List<Value> Parameters()
    {
        return Neurons.SelectMany(n => n.Parameters()).ToList();
    }
}

/// <summary>
/// Multi-Layer Perceptron (MLP) neural network.
/// </summary>
public class MLP
{
    public List<Layer> Layers { get; }

    public MLP(int nin, IReadOnlyList<int> nouts, Random? rng = null)
    {
        var sz = new List<int> { nin };
        sz.AddRange(nouts);

        Layers = new List<Layer>(nouts.Count);
        for (int i = 0; i < nouts.Count; i++)
        {
            Layers.Add(new Layer(sz[i], sz[i + 1], rng));
        }
    }

    public List<Value> Forward(IReadOnlyList<Value> x)
    {
        var current = x;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current.ToList();
    }

    public List<Value> Forward(IReadOnlyList<double> x) =>
        Forward(x.Select(v => new Value(v)).ToList());

    /// <summary>
    /// Helper for networks with a single scalar output node.
    /// </summary>
    public Value ForwardScalar(IReadOnlyList<Value> x)
    {
        var outs = Forward(x);
        return outs.Count == 1 ? outs[0] : throw new InvalidOperationException($"Expected 1 output, got {outs.Count}");
    }

    public Value ForwardScalar(IReadOnlyList<double> x) =>
        ForwardScalar(x.Select(v => new Value(v)).ToList());

    public List<Value> Parameters()
    {
        return Layers.SelectMany(l => l.Parameters()).ToList();
    }
}
