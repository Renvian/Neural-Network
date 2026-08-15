#training data
hours = 3
attendance = 0.8

actual_output = 1

#weights
w1 = 0.5
w2 = 1.0
w3 = 2.0
learning_rate = 0.01

#NN

for epoch in range(100):
#Forward pass
#N1
    print(f"epoch #{epoch}")
    z1 = w1 * hours+ w2 * attendance
    print(f"z1: {z1:.2f}")

#ReLU
    def relu(x):
        if x > 0:
            return x
        else:
            return 0
    '''Introduces non linearity in the nueral network, thereby being able to
                    1. Ignore weights that do not add real meaning, by giving a negatiove result, hence preventing their propogation
                    2. Resembles real world classification systems by having non linearity, thereby acknowledging the actual complexities of classification
    '''  

#activation
    a1 = relu(z1)
    print(f"activation: {a1:.2f}")


#Output neuron
    prediction = w3 * a1
    print(f"prediction: {prediction:.2f}") 

#Loss
    loss = (actual_output - prediction) ** 2
    print(f"Loss : {loss:.2f}")

#Back propogation

#gradient of loss
    ''' how loss changes wrt prediction '''

    dJ_dy = 2 * (prediction - actual_output)

#gradient for w3
    ''' how prediction changes with w3 '''

    dJ_dw3 = dJ_dy * a1 # dy_dw3 = a1

#ReLU derivative
    if z1 > 0:
        relu_grad = 1 #relu changes at the same rate as positive quantites do
    else:
        relu_grad = 0

#Hidden gradients leading upto gradients of w1 and w2
    dJ_dw2 = dJ_dy * w3 * relu_grad * attendance
    dJ_dw1 = dJ_dy * w3 * relu_grad * hours

    ''' dJ_dw1 = dJ_dy * dy_da1 * da1_dz1 * dz1_dw1 '''

#Gradient descent
    w1 = w1 - learning_rate * dJ_dw1
    w2 = w2 - learning_rate * dJ_dw2
    w3 = w3 - learning_rate * dJ_dw3

    ''' Weights are adjusted based on their gradient
    Learning rate is set appropriately, therefore not to skip the lowest loss region
    '-' indicates attempting to go in the direction opposite to that of the increasing cost
    '''

    print(f"{w1:.2f}")  
    print(f"{w2:.2f}")  
    print(f"{w3:.2f}")  
    print("\n")








