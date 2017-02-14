#This becomes a base (or say abstract) class to define base set of properties
#that every node holds
class Node(object):
    def __init__(self, inbound_nodes=[]):

        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes

        # Node(s) to which this Node passes values
        self.outbound_nodes = []

        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        # A calculated value
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented

#Creating child class of Node class which is "Input" node.    
class Input(Node):
    def __init__(self):
        #An input node has no inbound nodes,
        #so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    #Note: Input node is the only node where the value
    #may be passed as an argument to forward().
    #All other node implementations should get the value
    #of the previous node from self.inbound_nodes
    #
    #E.g. val0=self.inbound_nodes[0].value
    def forward(self, value=None):
        #Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

#Add subclass performs addition calculation
class Add(Node):
    def __init__(self,x,y):
        Node.__init__(self,[x,y])

    def forward(self):
        """
        You'll be writing code here in the next quiz!
        """

