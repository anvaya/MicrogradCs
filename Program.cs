using KarpathyAI;

        // Define neural network: 3 inputs, layers of [4, 4, 1]
        var rng = new Random(42);
        var n = new MLP(3, new[] { 4, 4, 1 }, rng);

        // Inputs (X)
        double[][] xs =
        [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
        ];

        // Desired targets (Y)
        double[] ys = [1.0, -1.0, -1.0, 1.0];

        // 50 training iterations
        for (int k = 0; k < 50; k++)
        {
            // Forward pass
            var ypred = xs.Select(x => n.ForwardScalar(x)).ToList();

            // Squared loss: sum((yout - ygt)^2)
            Value loss = 0.0;
            for (int i = 0; i < ys.Length; i++)
            {
                var diff = ypred[i] - ys[i];
                loss += diff.Pow(2);
            }

            // Reset gradients before backpropagation
            foreach (var p in n.Parameters())
            {
                p.Grad = 0.0;
            }

            // Backpropagation
            loss.Backward();

            // Update parameters (learning rate: -0.05)
            foreach (var p in n.Parameters())
            {
                p.Data += -0.05 * p.Grad;
            }

            Console.WriteLine($"Iteration {k + 1,2}  Loss: {loss.Data}");
        }
