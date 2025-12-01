# Mathematical Formulation of the Backprop Engine


## 1. Weighted Input (Pre-activation)
$$
\[
n = x_1 w_1 + x_2 w_2 + b
\]
$$


## 2. Sigmoid Activation (with \lambda parameter)

$$
\[
O = \sigma(\lambda n) = \frac{1}{1 + e^{-\lambda n}}
\]
$$

Derivative:

$$
\[
\frac{dO}{dn} = \lambda\, O(1 - O)
\]
$$


## 3. Loss Function (Squared Error)

$$
\[
L = \frac{1}{2}(T - O)^2
\]
$$

Derivative:

$$
\[
\frac{\partial L}{\partial O} = O - T
\]
$$


## 4. Backpropagation to Pre-activation 

$$
\[
\frac{\partial L}{\partial n}=\frac{\partial L}{\partial O}
\cdot
\frac{dO}{dn}
\]
$$

$$
\[
\frac{\partial L}{\partial n}=(O - T)\,\lambda\,O(1 - O)
\]
$$


## 5. Gradients of Parameters (Weights & Bias)

Given:

$$
\[
n = x_1 w_1 + x_2 w_2 + b
\]
$$

Derivatives:

$$
\[
\frac{\partial n}{\partial w_1} = x_1,
\qquad
\frac{\partial n}{\partial w_2} = x_2,
\qquad
\frac{\partial n}{\partial b} = 1
\]
$$

Thus:

$$
\[
\frac{\partial L}{\partial w_1} = x_1 \, \frac{\partial L}{\partial n}
\]
$$

$$
\[
\frac{\partial L}{\partial w_2} = x_2 \, \frac{\partial L}{\partial n}
\]
$$

$$
\[
\frac{\partial L}{\partial b} = 1 \cdot \frac{\partial L}{\partial n}
\]
$$


## 6. Input Gradients

$$
\[
\frac{\partial n}{\partial x_1} = w_1
\]
$$

$$
\[
\frac{\partial n}{\partial x_2} = w_2
\]
$$

Thus:

$$
\[
\frac{\partial L}{\partial x_1} = w_1 \frac{\partial L}{\partial n}
\]
$$

$$
\[
\frac{\partial L}{\partial x_2} = w_2 \frac{\partial L}{\partial n}
\]
$$


## 7. Weight Update Rule (Gradient Descent)

$$
\[
w_1 \leftarrow w_1 - \eta \frac{\partial L}{\partial w_1}
\]
$$

$$
\[
w_2 \leftarrow w_2 - \eta \frac{\partial L}{\partial w_2}
\]
$$

$$
\[
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]
$$


## 8. General Chain Rule for Any Node

If:

$$
\[
z = f(a, b)
\]
$$

Then:

$$
\[
\frac{\partial L}{\partial a}=\frac{\partial z}{\partial a}
\cdot
\frac{\partial L}{\partial z}
\]
$$

$$
\[
\frac{\partial L}{\partial b}=\frac{\partial z}{\partial b}
\cdot
\frac{\partial L}{\partial z}
\]
$$


## 9. Derivatives of Basic Operations

### Addition
$$
\[
z = a + b
\]
$$

$$
\[
\frac{\partial z}{\partial a} = 1,
\qquad
\frac{\partial z}{\partial b} = 1
\]
$$

### Multiplication
$$
\[
z = ab
\]
$$

$$
\[
\frac{\partial z}{\partial a} = b,
\qquad
\frac{\partial z}{\partial b} = a
\]
$$

### Negation
$$
\[
z = -a
\]
$$

$$
\[
\frac{\partial z}{\partial a} = -1
\]
$$

### Power (Square)
$$
\[
z = a^2
\]
$$
$$
\[
\frac{dz}{da} = 2a
\]
$$
