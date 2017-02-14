class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value
        self.value = bias
        for x, w in zip(inputs, weights):
            self.value += x * w
			
In the solution, I set self.value to the bias and then loop through the inputs and weights, adding each weighted input to self.value. Notice calling .value on self.inbound_nodes[0] or self.inbound_nodes[1] gives us a list.