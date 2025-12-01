# Mathematical Formulation of the Backprop Engine

## 1. Weighted Input (Pre-activation)
For a neuron receiving inputs \( x_1, x_2 \) with weights \( w_1, w_2 \) and bias \( b \):

\[
n = x_1 w_1 + x_2 w_2 + b
\]

This is the scalar weighted sum before applying the activation function.

---

## 2. Sigmoid Activation (with λ parameter)
The sigmoid activation used in the code is:

\[
O = \sigma(\lambda n) = \frac{1}{1 + e^{-\lambda n}}
\]

### Derivative of sigmoid:
\[
\frac{dO}{dn} 
= \lambda \, O (1 - O)
\]

This derivative appears in backpropagation when computing \(\frac{\partial L}{\partial n}\).

---

## 3. Loss Function (Squared Error)
The loss used is:

\[
L = \frac{1}{2}(T - O)^2
\]

where  
- \(T\) is the target output  
- \(O\) is the predicted output

### Loss derivative with respect to output \(O\):

\[
\frac{\partial L}{\partial O} = -(T - O) = O - T
\]

---

## 4. Backpropagation of Output Error
Using the chain rule:

\[
\frac{\partial L}{\partial n}
= 
\frac{\partial L}{\partial O}
\cdot
\frac{dO}{dn}
\]

Substituting derivatives:

\[
\frac{\partial L}{\partial n}
=
(O - T)\,\lambda\,O(1 - O)
\]

This value is stored as the node’s “delta”.

---

## 5. Gradients of Parameters (Weights and Bias)
For the pre-activation:

\[
n = x_1 w_1 + x_2 w_2 + b
\]

The derivatives of \(n\) are:

\[
\frac{\partial n}{\partial w_1} = x_1,
\qquad
\frac{\partial n}{\partial w_2} = x_2,
\qquad
\frac{\partial n}{\partial b} = 1
\]

By chain rule:

\[
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial n} \cdot x_1
\]

\[
\frac{\partial L}{\partial w_2}
=
\frac{\partial L}{\partial n} \cdot x_2
\]

\[
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial n} \cdot 1
\]

These expressions determine weight and bias gradients.

---

## 6. Gradient of Inputs (Optional, but computed by engine)
Similarly:

\[
\frac{\partial n}{\partial x_1} = w_1
\]

\[
\frac{\partial n}{\partial x_2} = w_2
\]

Thus:

\[
\frac{\partial L}{\partial x_1}
=
\frac{\partial L}{\partial n} \cdot w_1
\]

\[
\frac{\partial L}{\partial x_2}
=
\frac{\partial L}{\partial n} \cdot w_2
\]

---

## 7. Weight Update Rule (Gradient Descent)
Given learning rate \( \eta \):

\[
w_1 \leftarrow w_1 - \eta \frac{\partial L}{\partial w_1}
\]

\[
w_2 \leftarrow w_2 - \eta \frac{\partial L}{\partial w_2}
\]

\[
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]

This reduces the loss in the next forward pass.

---

## 8. Chain Rule Summary for Any Node
If a node \(z\) depends on predecessors \(a, b\):

\[
z = f(a, b)
\]

Then for the loss \(L\):

\[
\frac{\partial L}{\partial a}
=
\frac{\partial z}{\partial a}
\cdot
\frac{\partial L}{\partial z}
\]

\[
\frac{\partial L}{\partial b}
=
\frac{\partial z}{\partial b}
\cdot
\frac{\partial L}{\partial z}
\]

This is exactly what the engine performs using `+=` as gradients may accumulate from multiple downstream uses.

---

## 9. Derivatives of Basic Operations Used in the Engine

### Addition
\[
z = a + b
\]

\[
\frac{\partial z}{\partial a} = 1,
\qquad
\frac{\partial z}{\partial b} = 1
\]

### Multiplication
\[
z = a b
\]

\[
\frac{\partial z}{\partial a} = b,
\qquad
\frac{\partial z}{\partial b} = a
\]

### Negation
\[
z = -a
\]

\[
\frac{\partial z}{\partial a} = -1
\]

### Power (squared case)
\[
z = a^2
\]

\[
\frac{d z}{d a} = 2a
\]

These combine via chain rule to propagate deltas backward through the graph.

---

# End of Mathematical Summary
